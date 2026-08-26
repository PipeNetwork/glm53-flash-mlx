"""Teacher-forced, module-by-module comparison against transformers: where does the port first diverge?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, mlx.core as mx
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
import test_parity as tp

ids = tp.inputs(); B, T = ids.shape
hf = tp.build_hf(); rec = {}
def hook(name):
    def f(mod, inp, out):
        ins = tuple(i.detach().float().numpy() for i in inp if torch.is_tensor(i))
        outs = tuple(o.detach().float().numpy() for o in (out if isinstance(out, tuple) else (out,)) if torch.is_tensor(o))
        rec[name] = (ins, outs)
    return f
for name, mod in hf.named_modules():
    if name: mod.register_forward_hook(hook(name))
with torch.no_grad(): ref_logits = hf(input_ids=ids).logits.float().numpy()
model = tp.build_mlx(hf); lm = model.language_model.model
def d(a, b, label):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if a.shape != b.shape: print(f"  {label:50s} SHAPE {a.shape} vs {b.shape}"); return
    e = np.abs(a-b).max(); print(f"  {label:50s} max|d| {e:.3e}   scale {np.abs(b).max():.3e} {'<--' if e > 1e-4*max(np.abs(b).max(),1) else ''}")
P = "model.language_model."
x = mx.array(ids.numpy()); h = lm.embed_tokens(x); d(h, rec[P+"embed_tokens"][1][0], "embed")
fa_mask = create_attention_mask(h, None, return_array=True); ssm_mask = create_ssm_mask(h, None)
for i, layer in enumerate(lm.layers):
    p = f"{P}layers.{i}"
    hin = rec[p][0][0]; hh = mx.array(hin)
    xc, post, comb = layer.attn_hc(hh); r = rec[p+".attn_hc"][1]  # HF: (post, comb, collapsed)
    d(post, r[0], f"L{i} attn_hc.post"); d(comb, r[1], f"L{i} attn_hc.comb"); d(xc, r[2], f"L{i} attn_hc.collapsed")
    n = layer.input_layernorm(mx.array(r[2])); d(n, rec[p+".input_layernorm"][1][0], f"L{i} input_layernorm")
    a_in = mx.array(rec[p+".input_layernorm"][1][0]); aref = rec[p+".self_attn"][1][0]
    a = layer.self_attn(a_in, ssm_mask if layer.is_linear else fa_mask, None); d(a, aref, f"L{i} self_attn ({'KDA' if layer.is_linear else 'DSA'})")
    h2 = hc_expand(mx.array(aref), hh, mx.array(r[0]), mx.array(r[1])); h2_ref = rec[p+".ffn_hc"][0][0]; d(h2, h2_ref, f"L{i} hc_expand(attn)")
    xc2, post2, comb2 = layer.ffn_hc(mx.array(h2_ref)); r2 = rec[p+".ffn_hc"][1]
    d(xc2, r2[2], f"L{i} ffn_hc.collapsed"); d(comb2, r2[1], f"L{i} ffn_hc.comb")
    m_in = mx.array(rec[p+".post_attention_layernorm"][1][0]); mo = layer.mlp(m_in); d(mo, rec[p+".mlp"][1][0], f"L{i} mlp")
    d(layer(hh, mask=(ssm_mask if layer.is_linear else fa_mask), cache=None), rec[p][1][0], f"L{i} FULL (teacher-forced input)")
fin = lm.norm(mx.array(rec[P+"layers.3"][1][0]).mean(axis=2)); d(fin, rec[P+"norm"][1][0], "final norm(mean)")
d(model.language_model.lm_head(mx.array(rec[P+"norm"][1][0])), ref_logits, "lm_head")
