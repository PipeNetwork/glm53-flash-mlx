"""Build the REAP calibration corpus — deliberately disjoint from the eval set (wikitext-2 *test*).

Only wikitext-2 *train* English prose is used, mixed with ~10 languages of Wikipedia and real code
(routing specialises by language and domain; a monolingual profiler prunes the wrong experts), and
the result is checked for 32-gram overlap against the eval tokens before saving.

    python scripts/calib_corpus.py <MODEL_DIR_WITH_TOKENIZER> [OUT.npy] [N_TOKENS]
"""
import os, sys
import numpy as np
from transformers import AutoTokenizer

MODEL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib_corpus.npy")
N_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000
EVAL_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppl_corpus.npy")
LANGS = ["zh", "es", "fr", "de", "ru", "ja", "ar", "hi", "pt", "ko"]


def add(chunks, text, cap=3000):
    text = (text or "").strip()
    if len(text) >= 200:
        chunks.append(text[:cap])


def main():
    from datasets import load_dataset
    parts = {}
    ch = []
    for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]:
        add(ch, t)
        if len(ch) >= 400:
            break
    parts["wikitext-2/train"] = ch
    ch = []
    for lang in LANGS:
        try:
            n = 0
            for ex in load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True):
                add(ch, ex.get("text", "")); n += 1
                if n >= 12:
                    break
        except Exception as e:
            print(f"[calib] {lang} skipped: {str(e)[:60]}", flush=True)
    parts["wikipedia x10 langs"] = ch
    ch = []
    try:
        for ex in load_dataset("bigcode/the-stack-smol", split="train", streaming=True):
            add(ch, ex.get("content", ""))
            if len(ch) >= 150:
                break
    except Exception as e:
        print(f"[calib] the-stack-smol skipped: {str(e)[:60]}", flush=True)
    parts["code"] = ch
    for k, v in parts.items():
        print(f"[calib] {k:22s} {len(v):4d} chunks, {sum(map(len, v)):>9,} chars", flush=True)
    text = "\n\n".join(c for v in parts.values() for c in v)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = np.array(tok(text, add_special_tokens=False)["input_ids"][:N_TOKENS], dtype=np.int32)
    print(f"[calib] tokenized -> {len(ids):,} tokens", flush=True)
    ev = np.load(EVAL_CORPUS); N = 32
    ev_grams = {ev[i:i + N].tobytes() for i in range(0, len(ev) - N, N)}
    hits = sum(ids[i:i + N].tobytes() in ev_grams for i in range(0, len(ids) - N, N)); total = len(range(0, len(ids) - N, N))
    print(f"[calib] eval-set 32-gram overlap: {hits}/{total} ({100*hits/max(total,1):.3f}%)")
    if hits / max(total, 1) > 0.01:
        raise SystemExit("calibration overlaps the eval set — pruning would be fit to it")
    np.save(OUT, ids); print(f"[calib] saved -> {OUT}")


if __name__ == "__main__":
    main()
