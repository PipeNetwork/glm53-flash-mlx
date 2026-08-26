"""Create the Hugging Face collection for the GLM-5.3-Flash MLX builds (dry run without --yes)."""
from __future__ import annotations
import argparse
NAMESPACE, TITLE = "pipenetwork", "GLM-5.3-Flash MLX"
DESCRIPTION = ("Apple Silicon (MLX) builds of GLM-5.3-Flash with a validated, fixed glm5_next runtime: "
               "github.com/PipeNetwork/glm53-flash-mlx")
ITEMS = []  # filled in by the caller once measurements exist: (repo_id, note)

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--yes", action="store_true"); ap.add_argument("--items", help="json file of [[repo, note], ...]")
    args = ap.parse_args()
    items = ITEMS
    if args.items:
        import json; items = json.load(open(args.items))
    print(f"collection: {NAMESPACE}/{TITLE}\n{DESCRIPTION} ({len(DESCRIPTION)} chars)\n")
    for item, note in items: print(f"  {item}\n     {note}")
    if not args.yes:
        print("\ndry run — pass --yes to create"); return 0
    from huggingface_hub import HfApi
    api = HfApi()
    col = api.create_collection(title=TITLE, namespace=NAMESPACE, description=DESCRIPTION, exists_ok=True)
    for item, note in items:
        api.add_collection_item(col.slug, item_id=item, item_type="model", note=note, exists_ok=True)
    print(f"\nhttps://huggingface.co/collections/{col.slug}"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
