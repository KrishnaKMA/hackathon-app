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

content = "".join((base / name).read_text() for name in partials)
out_path.write_text(content)
print(f"Built {out_path}")
