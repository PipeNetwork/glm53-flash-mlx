"""Paired-bootstrap summary table shared by the model cards and the README.

Every build scores the same windows, so the comparison that means anything is per-window: NLL
difference vs the anchor, bootstrapped over one shared index set, plus the per-window win rate.
"""
from __future__ import annotations

import json

import numpy as np


def paired(res: dict, anchor: str, name: str, n_boot: int = 20000, seed: int = 0):
    a, b = res[anchor], res[name]
    na, nb, tok = np.array(a["window_nll"]), np.array(b["window_nll"]), np.array(a["window_tok"])
    assert (tok == np.array(b["window_tok"])).all(), "builds scored different windows"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tok), size=(n_boot, len(tok)))
    t = tok[idx].sum(1)
    d = (nb[idx].sum(1) - na[idx].sum(1)) / t          # mean NLL difference per token
    lo, hi = np.percentile(d, [2.5, 97.5])
    point = (nb.sum() - na.sum()) / tok.sum()
    worse = int(((nb / tok) > (na / tok)).sum())
    return point, lo, hi, worse, len(tok)


def rows(results_path: str, anchor: str, order: list[str], sizes_gb: dict[str, float], labels: dict[str, str]):
    res = json.load(open(results_path))
    out = []
    for name in order:
        if name not in res:
            continue
        r = res[name]
        if name == anchor:
            out.append((labels.get(name, name), sizes_gb.get(name), r["perplexity"], "—", "—"))
            continue
        point, lo, hi, worse, n = paired(res, anchor, name)
        sign = "+" if point >= 0 else "−"
        out.append((labels.get(name, name), sizes_gb.get(name), r["perplexity"],
                    f"{sign}{abs(point):.4f} [{'+' if lo >= 0 else '−'}{abs(lo):.4f}, {'+' if hi >= 0 else '−'}{abs(hi):.4f}]",
                    f"{worse}/{n}"))
    return out


def markdown(rows_, anchor_label: str) -> str:
    lines = ["| build | size | perplexity | ΔNLL/token vs " + anchor_label + " [95% CI] | windows worse |",
             "|---|---:|---:|---|---:|"]
    for label, size, ppl, delta, worse in rows_:
        size_s = f"{size:.1f} GB" if size else "—"
        lines.append(f"| {label} | {size_s} | {ppl:.4f} | {delta} | {worse} |")
    return "\n".join(lines)
