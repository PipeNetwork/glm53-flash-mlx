#!/bin/sh
# publish.sh <out_root> <results.json> <anchor_name> <build>...  — upload each build (guarded), then refresh every card.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT=$1; RES=$2; ANCHOR=$3; shift 3
PY="$ROOT/.venv/bin/python"
for b in "$@"; do
  ok=$($PY -c "import json,sys; r=json.load(open('$RES')); a=r['$ANCHOR']['perplexity']; p=r.get('$b',{}).get('perplexity'); print('yes' if p is not None and p < 1.5*a else 'no')")
  if [ "$ok" != "yes" ]; then echo "SKIP $b: perplexity missing or > 1.5x anchor (collapse guard)"; continue; fi
  echo "=== upload $b $(date)"; $PY "$ROOT/scripts/upload.py" --dir "$OUT/$b" --repo "pipenetwork/$b" --results "$RES" --yes || echo "UPLOAD FAILED: $b"
done
for b in "$@"; do $PY "$ROOT/scripts/upload.py" --dir "$OUT/$b" --repo "pipenetwork/$b" --results "$RES" --yes --card-only 2>&1 | tail -1; done
echo "=== done $(date)"
