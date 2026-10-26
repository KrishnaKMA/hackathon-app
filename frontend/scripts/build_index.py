from pathlib import Path

partials = [
    "head.html",
    "home.html",
    "detector.html",
    "learn.html",
    "citations.html",
    "footer.html",
]

base = Path(__file__).resolve().parent.parent / "partials"
out_path = Path(__file__).resolve().parent.parent / "index.html"

content = "".join((base / name).read_text(encoding="utf-8") for name in partials)
out_path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8", newline="\n")
print(f"Built {out_path}")
