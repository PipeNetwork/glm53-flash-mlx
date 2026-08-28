#!/bin/sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT=/Users/david/llm/glm53-flash-out; PY="$ROOT/.venv/bin/python"
for b in GLM-5.3-Flash-REAP25-MLX-mixed-4_8bit GLM-5.3-Flash-REAP37-MLX-mixed-4_8bit GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit; do
  echo "=== upload $b $(date)"
  $PY "$ROOT/scripts/upload.py" --dir "$OUT/$b" --repo "pipenetwork/$b" --yes || $PY "$ROOT/scripts/upload.py" --dir "$OUT/$b" --repo "pipenetwork/$b" --yes || echo "UPLOAD FAILED: $b"
done
rm -rf $OUT/GLM-5.3-Flash-REAP25-MLX-4bit $OUT/GLM-5.3-Flash-REAP37-MLX-4bit $OUT/GLM-5.3-Flash-REAP50-MLX-4bit && echo "deleted the three unpublished 4-bit REAP prunes"
for b in GLM-5.3-Flash-MLX-8bit GLM-5.3-Flash-MLX-6bit GLM-5.3-Flash-MLX-mixed-4_8bit GLM-5.3-Flash-MLX-4bit GLM-5.3-Flash-REAP25-MLX-mixed-4_8bit GLM-5.3-Flash-REAP37-MLX-mixed-4_8bit GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit; do
  $PY "$ROOT/scripts/upload.py" --dir "$OUT/$b" --repo "pipenetwork/$b" --yes --card-only 2>&1 | tail -1
done
$PY "$ROOT/scripts/make_collection.py" --items "$ROOT/collection_items.json" --yes 2>&1 | tail -2
echo "FLASH PUBLISH DONE"
