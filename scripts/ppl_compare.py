"""Paired comparison of builds scored on identical windows.

Independent confidence intervals are the wrong test here. Every build scores the
same 195 windows, so most of each interval's width is *window difficulty* — some
passages are simply harder — not any difference between builds. That shared
variance swamps the effect.

Pairing removes it: resample windows, and compute both builds' perplexity on the
**same** resample, then look at the distribution of the difference. Also reports a
per-window win rate, which needs no distributional assumption at all.

    python scripts/ppl_compare.py [RESULTS.json]
"""

import itertools
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppl_results.json")
N_BOOT = 20000


def paired_bootstrap(nll_a, nll_b, tok, n=N_BOOT, seed=0):
    """Bootstrap the perplexity *ratio* B/A over shared window resamples."""
    rng = np.random.default_rng(seed)
    k = len(tok)
    idx = rng.integers(0, k, size=(n, k))
    ta = tok[idx].sum(1)
    pa = np.exp(nll_a[idx].sum(1) / ta)
    pb = np.exp(nll_b[idx].sum(1) / ta)
    ratio = pb / pa
    return ratio


def main():
    res = json.load(open(RES))
    all_names = sorted(res, key=lambda n: int("".join(c for c in n if c.isdigit()) or 0))

    # A build that has collapsed (degenerate output) is not on the same scale as the
    # others; pairing it produces ratios of ~0 that crowd out the real comparisons.
    # Report it separately rather than ranking it.
    BROKEN = 100.0
    names = [n for n in all_names if res[n]["perplexity"] < BROKEN]
    broken = [n for n in all_names if res[n]["perplexity"] >= BROKEN]
    if broken:
        print("EXCLUDED — collapsed, not comparable:")
        for n in broken:
            print(f"  {n:32s} ppl {res[n]['perplexity']:,.1f}")
        print()
    if len(names) < 2:
        raise SystemExit(f"need >=2 usable builds in {RES}, have {names}")

    print(f"corpus: {res[names[0]]['tokens']:,} tokens in "
          f"{res[names[0]]['windows']} windows of {res[names[0]]['seq_len']}\n")
    print(f"{'build':32s} {'ppl':>8s}   95% CI (independent)")
    for n in names:
        r = res[n]
        print(f"  {n:30s} {r['perplexity']:8.4f}   [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")

    print(f"\npaired comparison ({N_BOOT:,} resamples of the same windows):")
    print(f"{'A vs B':40s} {'ppl ratio B/A':>14s}  {'95% CI':>20s}  {'B better':>9s}")
    for a, b in itertools.combinations(names, 2):
        ra, rb = res[a], res[b]
        na = np.array(ra["window_nll"]); nb = np.array(rb["window_nll"])
        tk = np.array(ra["window_tok"])
        assert (np.array(ra["window_tok"]) == np.array(rb["window_tok"])).all(), \
            "builds scored different windows — not comparable"

        ratio = paired_bootstrap(na, nb, tk)
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        point = np.exp(nb.sum() / tk.sum()) / np.exp(na.sum() / tk.sum())
        # per-window win rate: how often B assigns lower NLL per token
        win = float(np.mean((nb / tk) < (na / tk)))
        sig = "yes" if (lo > 1.0 or hi < 1.0) else "no"
        short_a = a.replace("GLM-5.3-Flash-", "").replace("MLX-", "")
        short_b = b.replace("GLM-5.3-Flash-", "").replace("MLX-", "")
        print(f"  {short_a:>6s} vs {short_b:<28s} {point:14.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  {win*100:8.1f}%   significant: {sig}")

    print("\n  ratio < 1 means B has lower (better) perplexity.")
    print("  'significant' = the paired 95% CI excludes 1.0.")


if __name__ == "__main__":
    main()
