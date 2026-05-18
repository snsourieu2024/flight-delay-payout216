"""Render reports/final_report.md -> reports/final_report.pdf, correctly.

The gstack make-pdf tool renders the markdown via a temporary HTML directory,
so relative image paths (figures/*.png) do not resolve and the charts come
out broken. This script fixes that reproducibly:

  * every figure is inlined as a base64 data URI inside a sized <img>
    (path-independent, and capped width so diagrams never overflow the page),
  * a small <style> block enlarges the body text and centres figures,
  * make-pdf is invoked with reduced page margins.

final_report.md itself keeps portable relative image paths (works on GitHub);
only the transient render copy is rewritten.
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD = REPORTS / "final_report.md"
TMP = REPORTS / "_final_report_render.md"
PDF = REPORTS / "final_report.pdf"

# Bigger body text, smaller default margins handled via --margins, and a hard
# cap on figure width so diagrams render inside the page, not off it.
STYLE = """<style>
body { font-size: 12.6pt !important; line-height: 1.42 !important; }
h1 { font-size: 21pt !important; }
h2 { font-size: 15.5pt !important; }
table { font-size: 11pt !important; }
img { max-width: 70% !important; height: auto !important;
      display: block !important; margin: 14px auto 4px auto !important; }
figcaption { text-align: center; font-size: 9.5pt; color: #5a6b73;
             margin-bottom: 14px; }
</style>

"""


def _datauri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _img_html(m: re.Match) -> str:
    alt, rel = m.group(1), m.group(2)
    src = _datauri(REPORTS / rel)
    return (f'<figure><img src="{src}" alt="{alt}">'
            f"<figcaption>{alt}</figcaption></figure>")


def _find_binary() -> str:
    for c in (ROOT / ".claude/skills/gstack/make-pdf/dist/pdf",
              Path.home() / ".claude/skills/gstack/make-pdf/dist/pdf"):
        if c.is_file():
            return str(c)
    sys.exit("make-pdf binary not found (run gstack ./setup).")


def main() -> None:
    md = MD.read_text()
    md = re.sub(r"!\[([^\]]*)\]\((figures/[^)]+)\)", _img_html, md)
    # Keep the document starting with the H1 (a leading raw <style> block
    # makes make-pdf emit a blank first page); inject styles just before the
    # Abstract instead -- CSS applies globally regardless of position.
    if "## Abstract" in md:
        md = md.replace("## Abstract", STYLE + "## Abstract", 1)
    else:
        md = md + "\n" + STYLE
    TMP.write_text(md)
    binary = _find_binary()
    try:
        subprocess.run([binary, "generate", "--margins", "0.55in",
                        TMP.name, PDF.name], cwd=REPORTS, check=True)
    finally:
        TMP.unlink(missing_ok=True)
    data = PDF.read_bytes()
    imgs = data.count(b"/Subtype /Image") + data.count(b"/Subtype/Image")
    pages = len(re.findall(rb"/Type\s*/Page(?![s])", data))
    print(f"[report-pdf] {PDF} | {len(data)//1024} KB | "
          f"{imgs} images | {pages} pages")


if __name__ == "__main__":
    main()
