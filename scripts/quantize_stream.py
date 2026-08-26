"""Quantize GLM-5.3-Flash one decoder layer at a time, never holding the 643 GB source.

Why per layer rather than per shard: the release stores the 288 routed experts of a layer as
separate tensors, and the runtime's sanitize stacks them into one `switch_mlp` tensor per
projection — it needs all 288 present at once. So the index is walked by layer: the tensors of
layer N are gathered lazily from whichever shards hold them (MLX memory-maps, nothing is read
until evaluation), sanitized as a group, quantized tensor by tensor, and written out.

Which modules are quantized is *derived* from the runtime: a full-size `Model` is built lazily
and every leaf that defines `to_quantized` — the question `nn.quantize` asks — is recorded, then
filtered by the recipe. Output keys follow the runtime's own (mlx-vlm) naming, so the checkpoint
reloads through `mlx_vlm`-style loaders and through this package's `load.py` alike.

Recipe:
  * routed experts (`switch_mlp`, 97% of parameters)      --bits / --expert-bits, group 64
  * everything else quantizable                            --bits / --other-bits, group 64
    (KDA projections, MLA low-rank projections, absorbed kv_b, shared experts, dense MLPs,
    embeddings, lm_head). The six KDA input projections must share one width — the runtime
    fuses them into a single matmul at load — which the recipe guarantees.
  * lightning-indexer projections: always 8-bit (block selection errors compound; ~0.2%)
  * kept as stored: the MoE router and its correction bias, mHC arrays (fp32 `base`/`scale`),
    KDA `A_log`/`dt_bias` (fp32), convolutions, norms, the vision tower (bf16)
  * dropped: the multi-token-prediction layer (layer 45)

    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4 --other-bits 8
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_flash_mlx.glm5_next import Model
from glm53_flash_mlx.load import make_config

AUX_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "processor_config.json", "preprocessor_config.json", "LICENSE")


def recipe(path: str, args) -> dict | None:
    if path.startswith("vision_model"):
        return None
    if ".indexer." in path:
        return {"group_size": args.group_size, "bits": 8}
    if ".switch_mlp." in path:
        return {"group_size": args.group_size, "bits": args.expert_bits}
    return {"group_size": args.group_size, "bits": args.other_bits}


def quantizable_paths(cfg, args) -> dict[str, dict]:
    model = Model(cfg)  # lazy: nothing allocated
    out = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized"):
            continue
        params = recipe(path, args)
        if params is None:
            continue
        if module.weight.shape[-1] % params["group_size"]:
            print(f"  skip {path}: in-dim {module.weight.shape[-1]} not divisible by {params['group_size']}")
            continue
        out[path] = params
    return out


def materialise(x):
    with mx.stream(mx.cpu):
        mx.eval(x)
    return x


def quantize(w, group_size, bits):
    try:
        out = mx.quantize(materialise(w), group_size=group_size, bits=bits); mx.eval(out); return out
    except RuntimeError as err:
        if "Timeout" not in str(err):
            raise
        with mx.stream(mx.cpu):
            out = mx.quantize(w, group_size=group_size, bits=bits); mx.eval(out); return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--expert-bits", type=int); ap.add_argument("--other-bits", type=int)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--shard-gb", type=float, default=10.0)
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()
    args.expert_bits = args.expert_bits or args.bits
    args.other_bits = args.other_bits or args.bits

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.load(open(src / "config.json"))
    cfg = make_config(raw_cfg)
    model = Model(cfg)
    qpaths = quantizable_paths(cfg, args)
    print(f"quantizable modules: {len(qpaths)} (experts {args.expert_bits}b, other {args.other_bits}b, indexer 8b)", flush=True)

    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    n_layers = cfg.text_config.num_hidden_layers
    groups: dict[str, list[str]] = defaultdict(list)
    for key in index:
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
        if m:
            groups[f"layer{int(m.group(1)):03d}"].append(key)
        elif key.startswith("model.visual."):
            groups["vision"].append(key)
        else:
            groups["top"].append(key)
    order = ["top"] + [f"layer{i:03d}" for i in range(n_layers)] + ["vision"]
    if args.limit_layers:
        order = ["top"] + [f"layer{i:03d}" for i in range(args.limit_layers)] + ["vision"]
    dropped = sorted(g for g in groups if g not in order)
    print(f"groups: {len(order)} (dropping {dropped}: multi-token-prediction layer)", flush=True)

    opened: dict[str, dict] = {}
    def fetch(keys):
        out = {}
        for k in keys:
            shard = index[k]
            if shard not in opened:
                opened[shard] = mx.load(str(src / shard))
            out[k] = opened[shard][k]
        return out

    target = args.shard_gb * 1e9
    out_index, pending, pending_bytes = {}, {}, 0
    out_n = total_out = 0
    counts = {"quantized": 0, "as_stored": 0}
    started = time.time()

    def flush():
        nonlocal pending, pending_bytes, out_n, total_out
        if not pending:
            return
        out_n += 1
        name = f"model-{out_n:05d}.safetensors"
        mx.save_safetensors(str(dst / name), pending, metadata={"format": "mlx"})
        for key in pending:
            out_index[key] = name
        size = (dst / name).stat().st_size; total_out += size
        print(f"  -> {name}  {len(pending)} tensors  {size/1e9:.2f} GB  (total {total_out/1e9:.1f} GB, {time.time()-started:.0f}s)", flush=True)
        pending, pending_bytes = {}, 0

    for gi, g in enumerate(order, 1):
        raw = fetch(groups[g])
        sane = model.sanitize(raw)
        print(f"[{gi}/{len(order)}] {g}: {len(raw)} -> {len(sane)} tensors", flush=True)
        for key, value in sane.items():
            module = key.rsplit(".", 1)[0]
            if key.endswith(".weight") and module in qpaths:
                p = qpaths[module]
                w, scales, biases = quantize(value, p["group_size"], p["bits"])
                emit = {module + ".weight": w, module + ".scales": scales, module + ".biases": biases}
                counts["quantized"] += 1
            else:
                emit = {key: materialise(value)}
                counts["as_stored"] += 1
            for k, v in emit.items():
                pending[k] = v; pending_bytes += v.nbytes
            if pending_bytes >= target:
                flush()
        del raw, sane
        opened.clear()
        mx.clear_cache()
    flush()

    quant = {"group_size": args.group_size, "bits": args.bits}
    for path, p in qpaths.items():
        if p != {"group_size": args.group_size, "bits": args.bits}:
            quant[path] = p
    cfg_out = dict(raw_cfg)
    cfg_out.pop("quantization_config", None)
    cfg_out["quantization"] = quant; cfg_out["quantization_config"] = quant
    json.dump(cfg_out, open(dst / "config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": total_out}, "weight_map": out_index}, open(dst / "model.safetensors.index.json", "w"), indent=2)
    for name in AUX_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    print(f"\n{out_n} shards, {total_out/1e9:.1f} GB, {(time.time()-started)/60:.1f} min; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
