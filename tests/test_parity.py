"""Numerical parity of this package's `glm5_next` runtime against transformers, at tiny scale.

A random tiny model is what lets every fragile path be exercised in seconds and be *broken on
purpose*: `swiglu_limit` is set low enough that the clamp actually bites, the mHC arrays and KDA
decay parameters are randomised so Sinkhorn mixing and the gates are data-dependent, the sequence
exceeds `index_topk` so the lightning indexer selects, and there are more experts than top-k.

    .venv/bin/python tests/test_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx

from glm53_flash_mlx.glm5_next import Model
from glm53_flash_mlx.load import make_config

TEXT = dict(
    model_type="glm5_next_text", vocab_size=128, hidden_size=64, intermediate_size=96,
    moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
    n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=16,
    q_lora_rank=32, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16, qk_head_dim=16,
    n_group=1, topk_group=1, num_experts_per_tok=2, norm_topk_prob=True, hidden_act="silu",
    max_position_embeddings=4096, rms_norm_eps=1e-5, first_k_dense_replace=1,
    index_topk=4, index_head_dim=8, index_n_heads=2, head_dim=0, index_kpool=2,
    index_kpool_compress=True, index_kpool_always_select_tail=True, indexer_rope_interleave=True,
    index_share_for_mtp_iteration=True,
    layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
    indexer_types=["full"] * 4, mlp_layer_types=["dense"] + ["sparse"] * 3,
    linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0, "kda_layers": [0, 1, 2], "full_attn_layers": [3]},
    swiglu_limit=0.5,  # low enough to clip real activations, so the clamp is load-bearing
    hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20, mhc=True, mla_use_nope=True,
    moe_router_dtype="float32", num_nextn_predict_layers=0, scoring_func="sigmoid",
    topk_method="noaux_tc", attention_bias=False, tie_word_embeddings=False,
    pad_token_id=0, eos_token_id=[1],
)
VISION = dict(
    model_type="glm5_next_vision", depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
    patch_size=14, out_hidden_size=64, projection_intermediate_size=64, image_size=28,
    in_channels=3, spatial_merge_size=2, temporal_patch_size=2, rms_norm_eps=1e-5,
    attention_bias=True, hidden_act="silu", swiglu_limit=10.0,
)
CFG = dict(model_type="glm5_next", text_config=TEXT, vision_config=VISION, image_token_id=100,
           video_token_id=101, image_start_token_id=102, image_end_token_id=103,
           video_start_token_id=104, video_end_token_id=105, tie_word_embeddings=False,
           pad_token_id=0, eos_token_id=[1])
B, T = 2, 12


def inputs():
    torch.manual_seed(1)
    return torch.randint(2, TEXT["vocab_size"], (B, T))


def build_hf(seed=0):
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
    torch.manual_seed(seed)
    model = Glm5NextForConditionalGeneration(Glm5NextConfig(**CFG)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("hc_attn_fn", "hc_ffn_fn")):
                p.normal_(0, 0.5)
            elif name.endswith(("hc_attn_base", "hc_ffn_base")):
                p.normal_(0, 1.0)
            elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
                p.uniform_(0.5, 1.5)
            elif name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
                p.normal_(0, 1.0)
            elif "norm" in name and p.ndim == 1:
                p.add_(0.3 * torch.randn_like(p))
            elif name.endswith(("index_kpool_compress_ape", "index_kpool_compress_gate")):
                p.normal_(0, 0.5)
            elif "proj" in name and p.ndim >= 2:
                p.mul_(3.0)  # push activations past swiglu_limit so the clamp is exercised
    return model


def hf_weights(model):
    """The tiny model's tensors laid out as the release ships them: transformers fuses the routed
    experts into `experts.gate_up_proj` [E, 2I, H] / `experts.down_proj` [E, H, I] on load, but the
    checkpoint holds per-expert `experts.{e}.{gate,up,down}_proj.weight` — which is what the
    runtime's sanitize stacks."""
    out = {}
    for k, v in model.state_dict().items():
        v = v.detach().numpy() if not v.is_floating_point() else v.detach().float().numpy()
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]; inter = v.shape[1] // 2
            for e in range(v.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = mx.array(v[e, :inter])
                out[f"{base}{e}.up_proj.weight"] = mx.array(v[e, inter:])
        elif k.endswith("mlp.experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = mx.array(v[e])
        else:
            out[k] = mx.array(v)
    return out


def build_mlx(hf, weights=None):
    model = Model(make_config(CFG))
    weights = weights if weights is not None else model.sanitize(hf_weights(hf))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def logits_hf(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float().numpy()


def logits_mlx(model, ids, **kw):
    out = model.language_model(mx.array(ids.numpy()), **kw)
    return np.array(out.logits.astype(mx.float32))


def report(label, out, ref, tol=1e-4):
    scale = float(np.abs(ref).max()); delta = float(np.abs(out - ref).max())
    ok = delta < tol * max(scale, 1.0)
    print(f"  {label:58s} max|delta| {delta:.3e}  (scale {scale:.3e})  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ids = inputs()
    all_ok = True
    hf = build_hf()
    ref = logits_hf(hf, ids)
    model = build_mlx(hf)
    print("[1] full forward (KDA x3 + DSA with live indexer, mHC, clamped MoE)")
    all_ok &= report("logits", logits_mlx(model, ids), ref)

    print("[2] short sequence (T <= index_topk: dense-MLA bypass)")
    short = ids[:, :4]
    all_ok &= report("logits", logits_mlx(model, short), logits_hf(hf, short))

    print("[3] sanitize is idempotent; fp32 scalars healed")
    raw = hf_weights(hf); once = model.sanitize(raw); twice = model.sanitize(dict(once))
    same = len(once) == len(twice) and all(k in twice and bool(mx.array_equal(once[k], twice[k])) for k in once)
    print(f"  {'second sanitize pass changes nothing':58s} {'OK' if same else 'FAIL'}"); all_ok &= same
    bf = {k: (v.astype(mx.bfloat16) if k.endswith(model.language_model.FP32_SUFFIXES) else v) for k, v in raw.items()}
    healed = model.sanitize(bf)
    f32 = all(healed[k].dtype == mx.float32 for k in healed if k.endswith(model.language_model.FP32_SUFFIXES))
    print(f"  {'bf16-cast hc/KDA scalars come back as float32':58s} {'OK' if f32 else 'FAIL'}"); all_ok &= f32

    print("[4] controls: each fix is load-bearing")
    import glm53_flash_mlx.glm5_next.language as L
    lim = L.ClampedSwiGLU.__call__
    L.ClampedSwiGLU.__call__ = lambda self, u, g: mx.array(torch.nn.functional.silu(torch.tensor(np.array(g))).numpy()) * u
    d = float(np.abs(logits_mlx(model, ids) - ref).max()); L.ClampedSwiGLU.__call__ = lim
    print(f"  {'control: unclamped SwiGLU -> logits move by':58s} {d:.3e}  {'OK' if d > 1e-3 else 'FAIL'}"); all_ok &= d > 1e-3
    for layer in model.language_model.model.layers:
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc._base_f32 = hc.base; hc.base = hc.base.astype(mx.bfloat16)
    d = float(np.abs(logits_mlx(model, ids) - ref).max())
    for layer in model.language_model.model.layers:
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc.base = hc._base_f32
    print(f"  {'control: bf16 mHC base through the Metal kernel -> moves by':58s} {d:.3e}  (documents the upstream convert bug)")

    print("[5] token-by-token decode == single forward (cached KDA state, indexer pools, MLA)")
    cache = model.language_model.make_cache()
    steps = [np.array(model.language_model(mx.array(ids[:, t:t+1].numpy()), cache=cache).logits.astype(mx.float32)) for t in range(T)]
    all_ok &= report("incremental vs single-shot", np.concatenate(steps, 1), logits_mlx(model, ids))
    cache = model.language_model.make_cache()
    a = np.array(model.language_model(mx.array(ids[:, :5].numpy()), cache=cache).logits.astype(mx.float32))
    b_ = np.array(model.language_model(mx.array(ids[:, 5:].numpy()), cache=cache).logits.astype(mx.float32))
    all_ok &= report("chunked prefill 5+7 vs single-shot", np.concatenate([a, b_], 1), logits_mlx(model, ids))

    print("\nALL OK" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
