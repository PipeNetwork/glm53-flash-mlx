# glm53-flash-mlx

MLX (Apple Silicon) runtime and quantization tooling for
[**zai-org/GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash) — 320B-A18B: 34
Kimi-Delta linear-attention layers interleaved with 11 DeepSeek-sparse-attention layers (NoPE MLA
behind a lightning indexer), 288-expert MoE, manifold-constrained hyper-connections
(`model_type: glm5_next`).

Published builds: **[pipenetwork/GLM-5.3-Flash MLX](https://huggingface.co/collections/pipenetwork)**
(8-bit, 6-bit, mixed 4/8-bit, 4-bit; see [Measurements](#measurements)).

## Why this exists

[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) merged `glm5_next` on 2026-08-26 (PR #2030); no
release carries it. Reviewing that port line by line against `transformers` 5.16, and then
checking it numerically at tiny scale, found the hyper-connection, KDA, MLA and indexer math all
correct — and two things wrong that fluent output would never reveal:

| | reference | mlx-vlm `main` | consequence |
|---|---|---|---|
| `swiglu_limit` | every text MLP clamps `gate ≤ 10`, `up ∈ [−10, 10]` before `silu(gate)·up` (routed experts, shared expert, dense layers 0–2) | inherits DeepSeek's unclamped `SwitchGLU` / `DeepseekMLP`; the config value is parsed and never read | formula mismatch on all 45 FFN blocks |
| mHC `base`/`scale` dtype | float32 (`_keep_in_fp32_modules_strict`) | `mlx_vlm.convert` casts them to bf16; the Metal kernel then reads `base` through a `float4` pointer | the Sinkhorn `comb` matrix is off by ~0.5 on every layer of a converted checkpoint (verified at real dims: kernel vs ops agree to 1e-7 in fp32, differ by 0.53 with bf16 `base`) |

Plus two epsilon mismatches (MLA low-rank norms use `rms_norm_eps` = 1e-5, not 1e-6; the indexer
LayerNorm uses 1e-6, not 1e-5) and bf16 router logits where the reference uses float32.

`glm53_flash_mlx/glm5_next/` is the mlx-vlm package with those fixed: a `ClampedSwiGLU` for the
routed experts, shared expert and dense layers; a float32 router; the epsilons; a `cast_predicate`
that protects the mHC arrays and KDA `A_log`/`dt_bias`, and a sanitize that heals checkpoints
where they were already cast.

## Validation

```bash
./scripts/run_tests.sh
```

A random tiny model with `swiglu_limit` low enough to actually clip, randomised mHC arrays and KDA
decay parameters, a sequence longer than `index_topk` so the indexer selects, and more experts
than top-k, checked against `transformers`:

```
[1] full forward (KDA x3 + DSA with live indexer, mHC, clamped MoE)
  logits                                        max|delta| 9.239e-07  (scale 5.863e-01)  OK
[2] short sequence (T <= index_topk: dense-MLA bypass)
  logits                                        max|delta| 8.643e-07  OK
[3] sanitize is idempotent; bf16-cast hc/KDA scalars come back as float32   OK
[4] control: unclamped SwiGLU -> logits move by  5.865e-01  OK
[5] token-by-token decode == single forward
  incremental vs single-shot                    max|delta| 2.086e-07  OK
  chunked prefill 5+7 vs single-shot            max|delta| 0.000e+00  OK
```

`tests/debug_modules.py` is the teacher-forced, module-by-module comparison that located the
clamp: every other block matched to ~1e-7 before the fix.

## Building the quants

```bash
python scripts/quantize_stream.py --src GLM-5.3-Flash-BF16-src --dst out/GLM-5.3-Flash-MLX-4bit --bits 4
python scripts/quantize_stream.py --src ... --dst out/...-mixed-4_8bit --bits 4 --other-bits 8
```

One decoder layer at a time: the release stores each layer's 288 experts as separate tensors and
the runtime stacks them, so the natural streaming unit is the layer (14.5 GB of experts in bf16),
gathered lazily from whichever shards hold it. The 643 GB source never needs to fit. The FP8
release would also work (the port dequantizes `weight_scale_inv` blocks); the bf16 release was
used so nothing is lost before quantization.

Recipe: routed experts (304B, 97%) at `--bits`; everything else quantizable (~9B) at
`--other-bits` — uniformly, because the runtime fuses the six KDA input projections into one
matmul at load and they must share a width; the lightning-indexer projections always at 8-bit;
the router, mHC arrays (fp32), KDA `A_log`/`dt_bias` (fp32), convolutions and norms as stored;
the vision tower in bf16; the MTP layer dropped.

## Measurements

`scripts/ppl_corpus.py` tokenizes wikitext-2 (test) once; `scripts/ppl_large.py` teacher-forces
every build on the same 141 windows of 2048 tokens; `scripts/ppl_compare.py` does the paired
bootstrap. The bf16 model does not fit a 512 GB machine, so 8-bit is the anchor.

<!-- measurements -->
| build | size | perplexity | ΔNLL/token vs 8-bit [95% CI] | windows worse |
|---|---:|---:|---|---:|
| [8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-8bit) | 334.1 GB | 3.4607 | — | — |
| [6bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-6bit) | 255.9 GB | 3.4646 | +0.0011 [−0.0017, +0.0038] | 89/141 |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit) | 181.9 GB | 3.5705 | +0.0312 [+0.0271, +0.0355] | 131/141 |
| [4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-4bit) | 177.6 GB | 3.7549 | +0.0816 [+0.0755, +0.0879] | 140/141 |
<!-- /measurements -->

## Layout

| path | what |
|---|---|
| `glm53_flash_mlx/glm5_next/` | the runtime (mlx-vlm `glm5_next` + fixes) |
| `glm53_flash_mlx/load.py` | loader that replays per-module quantization from `config.json` |
| `scripts/quantize_stream.py` | per-layer streaming quantizer |
| `scripts/ppl_corpus.py`, `ppl_large.py`, `ppl_compare.py`, `ppl_table.py` | evaluation |
| `scripts/smoke_generate.py` | greedy generation, collapse detector |
| `scripts/upload.py`, `make_collection.py` | publishing, card rendered from the measurements |
| `tests/test_parity.py`, `tests/debug_modules.py` | validation |
| `docs/upstream-notes.md` | findings written up for mlx-vlm |
