"""Publish a built GLM-5.3-Flash MLX quant to the Hub, with a card rendered from the measurements.

    .venv/bin/python scripts/upload.py --dir <build dir> --repo pipenetwork/<name> [--yes]

Nothing uploads without --yes. The quality table comes from ppl_results.json through the paired
bootstrap in ppl_table.py, so the card carries the numbers that were measured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_table import markdown, rows

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "zai-org/GLM-5.3-Flash"
CODE_REPO = "https://github.com/PipeNetwork/glm53-flash-mlx"
ANCHOR = "GLM-5.3-Flash-MLX-8bit"
ORDER = ["GLM-5.3-Flash-MLX-8bit", "GLM-5.3-Flash-MLX-6bit", "GLM-5.3-Flash-MLX-mixed-4_8bit", "GLM-5.3-Flash-MLX-4bit"]
LABELS = {n: f"[{n.split('MLX-')[1]}](https://huggingface.co/pipenetwork/{n})" for n in ORDER}

CARD = """---
license: mit
base_model: {upstream}
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm5_next
- mixture-of-experts
- {bits_tag}
pipeline_tag: image-text-to-text
library_name: mlx
---

# {repo_name}

MLX (Apple Silicon) build of [**GLM-5.3-Flash**](https://huggingface.co/{upstream}) — 320B-A18B
hybrid of 34 Kimi-Delta linear-attention layers and 11 DeepSeek-sparse-attention (NoPE MLA +
lightning indexer) layers with manifold-constrained hyper-connections — quantized to **{recipe}**.

**These files are modified**: converted from the upstream bfloat16 release
([GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)) to MLX and quantized;
the architecture is unchanged. The multi-token-prediction layer (layer 45) is not included. The
vision tower is carried in bfloat16.

## Runtime

`glm5_next` landed in [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) `main` on 2026-08-26 (no
release carries it yet). Validating that port against `transformers` 5.16 at tiny scale found two
numerical bugs and two epsilon mismatches, which the runtime in
[{code_repo}]({code_repo}) fixes; parity is **1e-6** end to end, exact on cached decode.

| what | reference | mlx-vlm `main` | effect |
|---|---|---|---|
| `swiglu_limit` | gate clamped at 10, up at ±10, in every text MLP | no clamp anywhere in the text stack | formula mismatch on all 45 FFN blocks |
| mHC `base`/`scale` dtype | float32 | converter casts to bf16; the Metal kernel then reads `base` as float4 | `comb` mixing matrix off by ~0.5 on every layer of a converted checkpoint |
| MLA low-rank norm eps | `rms_norm_eps` = 1e-5 | 1e-6 | small |
| indexer LayerNorm eps | 1e-6 | 1e-5 | small |

This checkpoint keeps the mHC arrays and KDA decay parameters in float32 as stored, so it is
safe in either runtime; the clamp is a compute-path fix and needs the patched runtime:

```bash
git clone {code_repo} && cd glm53-flash-mlx && pip install -r requirements.txt
python scripts/smoke_generate.py /path/to/{repo_name}
```
```python
from glm53_flash_mlx.load import load
model, processor = load("/path/to/{repo_name}")
```

## Size and what is quantized

**{gb:.1f} GB** on disk (bfloat16 upstream: 642.7 GB).

| group | share of parameters | this build |
|---|---:|---|
| routed experts (`switch_mlp`, 42 layers × 288) | 304B (97%) | {expert_bits}-bit, group 64 |
| KDA and MLA projections, shared experts, dense MLPs, embeddings, `lm_head` | ~9B | {other_bits}-bit, group 64 |
| lightning-indexer projections | 0.06B | 8-bit, group 64 |
| MoE router + correction bias, mHC arrays (fp32), KDA `A_log`/`dt_bias` (fp32), convolutions, norms | — | as stored |
| vision tower | 0.56B | bfloat16 |

## Quality

Perplexity on wikitext-2 (test), {tokens:,} tokens in {windows} windows of {seq}, every build scored
on **identical** windows through this runtime. The 643 GB bfloat16 model does not fit a 512 GB
machine, so the 8-bit build is the anchor (on every model we have measured, 8-bit has been
statistically indistinguishable from bfloat16). Per-window NLL differences against 8-bit,
bootstrapped over one shared index set (20,000 resamples):

{table}

Read the interval, not the point estimate; "windows worse" counts how many of the {windows}
windows the build lost outright.

Greedy generation (a collapse detector, not a ranking) is coherent on every published build.

## License

MIT, as the upstream model. Port code: [{code_repo}]({code_repo}).
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--results", default=str(ROOT / "ppl_results.json"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    d = Path(args.dir)
    cfg = json.load(open(d / "config.json")); q = cfg["quantization"]
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    expert_bits = next((v["bits"] for k, v in overrides.items() if ".switch_mlp." in k), q["bits"])
    other_bits = next((v["bits"] for k, v in overrides.items() if ".switch_mlp." not in k and ".indexer." not in k), q["bits"])
    recipe = f"{expert_bits}-bit" if expert_bits == other_bits else f"{expert_bits}-bit experts / {other_bits}-bit everything else"
    gb = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e9
    sizes = {}
    for n in ORDER:
        p = Path("/Users/david/llm/glm53-flash-out") / n
        if p.exists():
            sizes[n] = sum(f.stat().st_size for f in p.iterdir() if f.is_file()) / 1e9
    res = json.load(open(args.results)); a = res[ANCHOR]
    table = markdown(rows(args.results, ANCHOR, ORDER, sizes, LABELS), "8-bit")
    card = CARD.format(upstream=UPSTREAM, code_repo=CODE_REPO, repo_name=args.repo.split("/")[-1],
                       bits_tag=f"{expert_bits}-bit", recipe=recipe, gb=gb, expert_bits=expert_bits,
                       other_bits=other_bits, tokens=a["tokens"], windows=a["windows"], seq=a["seq_len"], table=table)
    (d / "README.md").write_text(card)
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    print(f"repo   {args.repo}\ndir    {d}\nfiles  {len(files)}, {gb:.1f} GB\n"); print(table)
    if not args.yes:
        print("\ndry run — pass --yes to upload"); return 0
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(args.repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(d), repo_id=args.repo, repo_type="model")
    print(f"\nuploaded https://huggingface.co/{args.repo}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
