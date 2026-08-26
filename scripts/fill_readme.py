"""Replace the MEASUREMENTS_TABLE marker (or a previously rendered table) in README.md with the paired table."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_table import markdown, rows
import upload as U

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
res_path = os.path.join(root, "ppl_results.json")
sizes = {}
for n in U.ORDER:
    p = os.path.join(os.path.dirname(U.ROOT), U.ROOT.name.replace("-mlx", "-out"), n)
    if os.path.isdir(p):
        sizes[n] = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))) / 1e9
if hasattr(U, "ANCHOR") and U.ANCHOR.endswith("-src"):
    sizes[U.ANCHOR] = 360.0
table = markdown(rows(res_path, U.ANCHOR, U.ORDER, sizes, U.LABELS), "bf16" if U.ANCHOR.endswith("-src") else "8-bit")
block = "<!-- measurements -->\n" + table + "\n<!-- /measurements -->"
readme = os.path.join(root, "README.md"); s = open(readme).read()
if "MEASUREMENTS_TABLE" in s:
    s = s.replace("MEASUREMENTS_TABLE", block)
else:
    s = re.sub(r"<!-- measurements -->.*?<!-- /measurements -->", block, s, flags=re.S)
open(readme, "w").write(s); print(table)
