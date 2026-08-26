# Notes for Blaizzy/mlx-vlm (glm5_next, merged in #2030)

Found by line-by-line review against `transformers` 5.16 and tiny-config parity
(`tests/test_parity.py`, `tests/debug_modules.py`).

1. **`swiglu_limit` is never applied.** `Glm5NextTextMLP` and `Glm5NextTextExperts._apply_gate`
   clamp `gate` at `+limit` and `up` at `±limit` before `silu(gate) * up`; `language.py` uses
   `DeepseekV32MoE` (plain `SwitchGLU`) and `DeepseekMLP` for dense/shared. Fix: pass
   `activation=ClampedSwiGLU(config.swiglu_limit)` to `SwitchGLU` (note it is called as
   `activation(x_up, x_gate)`), and a clamped MLP for the dense layers and the shared expert.
2. **mHC `base`/`scale` must stay float32.** `convert.py` `set_dtype` casts every floating param
   except what `cast_predicate` excludes (only `e_score_correction_bias`). `_hc_kernel` reads
   `base` as `float4`, so bf16 `base` yields a wrong `comb` (0.53 max diff at hidden 4096 vs 9e-8
   in fp32). Fix: exclude `.attn_hc.`, `.ffn_hc.`, `A_log`, `dt_bias` in `cast_predicate` (as
   `deepseek_v4` does), cast them to float32 in `sanitize`, and/or `astype(float32)` in the kernel.
3. **Epsilons.** `q_a_layernorm`/`kv_a_layernorm` should use `config.rms_norm_eps` (1e-5); the
   indexer `k_norm` LayerNorm uses eps 1e-6 in the reference.
4. **Router logits** are float32 in the reference (`moe_router_dtype`); `MoEGate` matmuls in bf16.
5. README claims self-speculative decoding from the MTP head; the head is dropped by sanitize.
