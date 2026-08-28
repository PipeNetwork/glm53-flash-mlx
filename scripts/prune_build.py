"""REAP-prune an already-built (quantized) MLX checkpoint, streaming, at several ratios at once.

Pruning the quantized build is equivalent to pruning bf16 and requantizing: expert subsetting runs
along the expert axis (0) while affine quant groups run along the input dim, so the two are
independent — one fast pass instead of a reconvert per ratio. Dense layers (0-2) are untouched.

    python scripts/prune_build.py <SRC_BUILD> <saliency.npz> <ratio,ratio,...>
"""
import json, os, re, shutil, sys
import numpy as np
import mlx.core as mx

SRC = sys.argv[1]; SAL = sys.argv[2]
RATIOS = [int(r) for r in (sys.argv[3] if len(sys.argv) > 3 else "25,37,50").split(",")]
_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_PER_EXPERT = ("mlp.switch_mlp.gate_proj.", "mlp.switch_mlp.up_proj.", "mlp.switch_mlp.down_proj.")
_SHARD_CAP = 10_000_000_000


def keep_indices(usage, ratio):
    sal = usage["saliency"].astype(np.float64); counts = usage["counts"].astype(np.float64)
    E = sal.shape[1]; K = int(round(E * (1 - ratio / 100)))
    S = np.where(counts == 0, -1.0, sal)  # never-used experts prune first
    order = np.argsort(-S, axis=1); keep = np.sort(order[:, :K], axis=1)
    moe = [int(i) for i in usage["moe_layers"]]
    retained = np.array([np.take_along_axis(sal[i], keep[i], 0).sum() / max(sal[i].sum(), 1e-9) for i in moe])
    return keep, K, retained, moe


def subset(name, arr, keep, moe):
    m = _LAYER_RE.match(name)
    if m is None or int(m.group(1)) not in moe:
        return arr
    layer = int(m.group(1)); rest = name[m.end():]
    if rest.startswith(_PER_EXPERT) or rest in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias"):
        return arr[mx.array(keep[layer])]
    return arr


def main():
    usage = np.load(SAL); outputs = {}
    for r in RATIOS:
        keep, K, retained, moe = keep_indices(usage, r)
        dst = re.sub(r"(GLM-5\.3-Flash)-MLX", rf"\1-REAP{r}-MLX", SRC.rstrip("/"))
        outputs[r] = dict(keep=keep, K=K, dst=dst, moe=set(moe), idx={}, buf={}, bytes=0, sid=0, retained=retained)
        print(f"[prune] REAP{r}: keep {K}/{usage['saliency'].shape[1]} in {len(moe)} MoE layers | saliency retained mean {100*retained.mean():.1f}% worst {100*retained.min():.1f}% -> {dst}", flush=True)
        os.makedirs(dst, exist_ok=True)

    def flush(st):
        if not st["buf"]:
            return
        st["sid"] += 1; fn = f"model-{st['sid']:05d}.safetensors"
        mx.save_safetensors(os.path.join(st["dst"], fn), st["buf"], metadata={"format": "mlx"})
        for k in st["buf"]:
            st["idx"][k] = fn
        st["buf"] = {}; st["bytes"] = 0

    index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    for shard in sorted(set(index["weight_map"].values())):
        tensors = mx.load(os.path.join(SRC, shard))
        for name, arr in tensors.items():
            for r, st in outputs.items():
                out = subset(name, arr, st["keep"], st["moe"]); mx.eval(out)
                st["buf"][name] = out; st["bytes"] += out.nbytes
                if st["bytes"] >= _SHARD_CAP:
                    flush(st)
        del tensors; mx.clear_cache(); print(f"[prune] {shard}", flush=True)

    src_cfg = json.load(open(os.path.join(SRC, "config.json")))
    for r, st in outputs.items():
        flush(st)
        total = sum(os.path.getsize(os.path.join(st["dst"], f)) for f in set(st["idx"].values()))
        json.dump({"metadata": {"total_size": total}, "weight_map": st["idx"]}, open(os.path.join(st["dst"], "model.safetensors.index.json"), "w"), indent=2)
        cfg = dict(src_cfg); cfg["text_config"] = dict(cfg["text_config"]); cfg["text_config"]["n_routed_experts"] = st["K"]
        cfg["reap"] = {"kept_experts": st["K"], "original_experts": int(usage["saliency"].shape[1]), "ratio_pct": r,
                       "moe_layers": len(st["moe"]), "saliency_retained_mean": float(st["retained"].mean()),
                       "calibration_tokens": int(usage["tokens"]), "saliency": os.path.basename(SAL)}
        json.dump(cfg, open(os.path.join(st["dst"], "config.json"), "w"), indent=2)
        for f in os.listdir(SRC):
            if f.endswith((".json", ".jinja", ".txt", ".py")) or f == "LICENSE":
                if f not in ("config.json", "model.safetensors.index.json"):
                    shutil.copy2(os.path.join(SRC, f), os.path.join(st["dst"], f))
        print(f"[prune] wrote {st['dst']} ({st['sid']} shards, {total/1e9:.1f} GB)", flush=True)
    print("[prune] ALL DONE")


if __name__ == "__main__":
    main()
