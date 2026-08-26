"""Tokenize wikitext-2 (test split) once, so every build is scored on identical tokens.

    python scripts/ppl_corpus.py <MODEL_DIR_WITH_TOKENIZER> <OUT.npy> [N_TOKENS]

Reads the local parquet copy (no network) and writes an int32 array. If each build
re-tokenized its own text, build differences would be confounded with input differences.
"""
import sys
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PARQUET = "/Volumes/models/eval-corpus/wikitext2-test.parquet"
model_dir, out = sys.argv[1], sys.argv[2]
n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 0

rows = pq.read_table(PARQUET).column("text").to_pylist()
text = "".join(rows)  # wikitext rows already carry their newlines
tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"[corpus] {len(rows)} rows, {len(text):,} chars -> {len(ids):,} tokens ({tok.__class__.__name__})")
if n_tokens:
    ids = ids[:n_tokens]
np.save(out, np.array(ids, dtype=np.int32))
with open(out.replace(".npy", ".meta.txt"), "w") as fh:
    fh.write(f"source: wikitext-2-raw-v1/test (local parquet)\ntokenizer: {model_dir}\ntokens: {len(ids)}\n")
print(f"[corpus] saved {len(ids):,} tokens -> {out}")
