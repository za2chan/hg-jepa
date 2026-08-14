#!/usr/bin/env bash
# Build the Overleaf upload from paper_ICLR/. Ships only what a compile needs:
# the .tex sources, the ICLR class files, and the figures actually referenced.
# Leaves out iclr2025.zip, main.pdf and any figure no \includegraphics names.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=hglp_overleaf.zip
rm -f "$OUT"

python3 - <<'PY'
import pathlib, re, zipfile
root = pathlib.Path("paper_ICLR")
blob = "".join(p.read_text() for p in root.glob("*.tex"))
figs = {f for f in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', blob)
        if (root / f).exists()}
# iclr2025_conference.tex is the template's OWN document, with its own
# \documentclass. Shipping it makes Overleaf offer two main files and pick the
# wrong one, so it stays out; the class file it goes with does not.
keep = sorted([p for p in root.glob("*.tex") if p.name != "iclr2025_conference.tex"]
              + [p for p in root.glob("*.sty")]
              + [p for p in root.glob("*.bst")]
              + [p for p in root.glob("*.bib")]
              + [root / f for f in figs])
missing = [f for f in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', blob)
           if not (root / f).exists() and "myfile" not in f]
with zipfile.ZipFile("hglp_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in keep:
        z.write(p, p.relative_to(root))
print(f"{len(keep)} files")
for m in sorted(set(missing)):
    print(f"  WARNING referenced but absent: {m}")
PY
ls -lh "$OUT"
