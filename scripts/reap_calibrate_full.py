"""Per-expert REAP saliency for GLM-5.3-Flash, from the resident 8-bit build.

The 8-bit build (334 GB) fits in memory and is statistically indistinguishable from bf16, so
saliency (`router_weight x ||expert_output||`, accumulated in two disjoint halves of the calibration
set) is collected by running the whole model with the MoE forward instrumented — no per-layer
streaming needed.

    python scripts/reap_calibrate_full.py <8bit build> <calib_corpus.npy> <saliency.npz> [samples] [seq_len]
"""
import sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from glm53_flash_mlx.load import load_model
import glm53_flash_mlx.glm5_next.language as L

try:
    mx.set_wired_limit(int(440e9))
except Exception as e:
    print("[warn]", e)
build, ids_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
samples = int(sys.argv[4]) if len(sys.argv) > 4 else 32; seq_len = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
model, cfg = load_model(build, lazy=True)
layers = model.language_model.model.layers
E = cfg["text_config"]["n_routed_experts"]; nL = len(layers)
moe_ids = [i for i, l in enumerate(layers) if isinstance(l.mlp, L.Glm5NextMoE)]
for i in moe_ids:
    layers[i].mlp._reap_idx = i
sal = np.zeros((2, nL, E)); cnt = np.zeros((2, nL, E)); half = [0]

def moe_forward(self, x):
    inds, scores = self.gate(x)
    y = self.switch_mlp(x, inds)
    contrib = scores.astype(mx.float32) * mx.sqrt((y.astype(mx.float32) ** 2).sum(-1))
    fi = inds.reshape(-1)
    s = mx.zeros((E,), dtype=mx.float32).at[fi].add(contrib.reshape(-1)); c = mx.zeros((E,), dtype=mx.float32).at[fi].add(mx.ones((fi.size,), dtype=mx.float32))
    mx.eval(s, c); sal[half[0], self._reap_idx] += np.array(s, dtype=np.float64); cnt[half[0], self._reap_idx] += np.array(c, dtype=np.float64)
    out = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if self.config.n_shared_experts is not None:
        out = out + self.shared_experts(x)
    return out
L.Glm5NextMoE.__call__ = moe_forward
for l in layers:
    l.compile_ffn = False  # instrumented path must not be compiled

ids = np.load(ids_path)[: samples * seq_len].reshape(samples, seq_len)
print(f"calibration: {samples} x {seq_len} = {ids.size} tokens over {len(moe_ids)} MoE layers", flush=True)
t0 = time.time()
for s in range(samples):
    half[0] = 0 if s < samples // 2 else 1
    model.language_model(mx.array(ids[s:s + 1])); mx.clear_cache()
    if (s + 1) % 4 == 0:
        print(f"  {s+1}/{samples} sequences  ({time.time()-t0:.0f}s, peak {mx.get_peak_memory()/1e9:.0f} GB)", flush=True)
total, count = sal.sum(0), cnt.sum(0)
mean = np.where(count > 0, total / np.maximum(count, 1), 0.0); halves = np.where(cnt > 0, sal / np.maximum(cnt, 1), 0.0)
np.savez(out, saliency=mean, total=total, counts=count, saliency_halves=halves, counts_halves=cnt, tokens=np.array(ids.size),
         samples=np.array(samples), seq_len=np.array(seq_len), moe_layers=np.array(moe_ids))
print(f"wrote {out} ({(time.time()-t0)/60:.1f} min)\n\nsplit-half agreement:\n{'keep':>6} {'overlap':>9} {'spearman':>9}")
for keep in (0.75, 0.63, 0.5):
    k = max(1, int(round(E * keep))); ov, rh = [], []
    for i in moe_ids:
        a, b = halves[0, i], halves[1, i]; ov.append(len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k)
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); rh.append(np.corrcoef(ra, rb)[0, 1])
    print(f"{keep:>6.0%} {np.mean(ov):>8.1%} {np.mean(rh):>9.3f}")
