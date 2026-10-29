"""Page size (MediaBox, pt) and embedded font names of matplotlib PDFs (pdfinfo / pdffonts are not on Jean-Zay)."""
import re
import sys

for f in sys.argv[1:]:
    b = open(f, "rb").read()
    box = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", b)
    w, h = (float(box.group(3)) - float(box.group(1)), float(box.group(4)) - float(box.group(2))) if box else (None, None)
    fonts = sorted({m.decode() for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-_]+)", b)})
    embedded = bool(re.search(rb"/FontFile[23]?", b))
    bold = [x for x in fonts if "bold" in x.lower()]
    ok = w is not None and abs(w - 238.8) <= 3 and abs(h - 224.4) <= 3
    print(f"{f.split('/')[-1]:62s} {w:7.1f} x {h:6.1f} pt {'OK ' if ok else 'BAD'} | fonts {fonts} embedded={embedded} bold={bold}")
