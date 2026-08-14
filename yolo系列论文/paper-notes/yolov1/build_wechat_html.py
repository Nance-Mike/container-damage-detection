#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a WeChat-ready, self-contained HTML from wechat-article.md.

Pipeline:
  1. pandoc  -> HTML fragment with KaTeX math spans
  2. Render each unique formula to a tight PNG via headless Chrome + KaTeX
  3. Replace math spans with embedded PNG <img>
  4. Embed local figures as base64
  5. Inline WeChat-safe styles onto block elements
Output: wechat-article.html (same directory)
"""
import os, re, base64, subprocess, shutil, sys, json, hashlib, html as ihtml
from tempfile import mkdtemp

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "wechat-article.md")
TMP  = os.path.join(BASE, "_raw.html")
OUT  = os.path.join(BASE, "wechat-article.html")
NODE = r"C:\Users\administrator\.workbuddy\binaries\node\versions\22.22.2\node.exe"
MATH_RENDERER = os.path.join(BASE, "render_math.cjs")

PANDOC = shutil.which("pandoc") or "pandoc"

def title_from_frontmatter():
    m = re.search(r'^---\s*\n.*?title:\s*"([^"]+)".*?---', open(SRC, encoding="utf-8").read(), re.S|re.M)
    return m.group(1) if m else "YOLOv1 阅读笔记"

# ----------------------------------------------------------------------------
# 1. pandoc -> fragment (no standalone, so no CDN refs / default CSS)
# ----------------------------------------------------------------------------
print("[1/5] pandoc: markdown -> html fragment")
cmd = [PANDOC, SRC, "-f", "markdown", "-t", "html5",
       "--katex", "--highlight-style=tango", "-o", TMP]
r = subprocess.run(cmd, capture_output=True)
if r.returncode != 0:
    print("pandoc failed:\n", (r.stderr or b"").decode("utf-8", "replace")); sys.exit(1)

html = open(TMP, encoding="utf-8").read()
print("      raw bytes:", len(html), "math spans:", html.count('class="math'), "imgs:", html.count("<img"))

# ----------------------------------------------------------------------------
# 2. collect + render unique formulas to PNG
# ----------------------------------------------------------------------------
MATH_RE = re.compile(r'<span class="math (inline|display)">(.*?)</span>', re.S)

def make_key(kind, tex):
    return f"{kind}_{hashlib.md5(tex.encode('utf-8')).hexdigest()[:12]}"

formulas = []
seen = set()
for kind, raw in MATH_RE.findall(html):
    tex = ihtml.unescape(raw)
    key = make_key(kind, tex)
    if key in seen: continue
    seen.add(key)
    formulas.append({"key": key, "tex": tex, "display": kind == "display"})

print("[2/5] unique formulas to render:", len(formulas))
math_map = {}
if formulas:
    tmpdir = mkdtemp(prefix="wx_math_")
    formulas_path = os.path.join(tmpdir, "formulas.json")
    map_path = os.path.join(tmpdir, "map.json")
    png_dir = os.path.join(tmpdir, "png")
    open(formulas_path, "w", encoding="utf-8").write(json.dumps(formulas, ensure_ascii=False))
    rc = subprocess.run([NODE, MATH_RENDERER, formulas_path, png_dir, map_path]).returncode
    if rc == 0 and os.path.exists(map_path):
        math_map = json.load(open(map_path, encoding="utf-8"))
        print("      rendered:", len(math_map), "of", len(formulas))
    else:
        print("      WARN: formula rendering failed or partial; some math will stay as raw text")
    try: shutil.rmtree(tmpdir)
    except Exception: pass

# ----------------------------------------------------------------------------
# 3. replace math spans with embedded PNG <img>
# ----------------------------------------------------------------------------
print("[3/5] replacing math spans with PNG images")
def math_repl(m):
    kind, raw = m.group(1), m.group(2)
    tex = ihtml.unescape(raw)
    key = make_key(kind, tex)
    src = math_map.get(key)
    if not src:
        return m.group(0)  # fallback to original span if render failed
    if kind == "inline":
        style = "display:inline-block;vertical-align:-0.15em;height:1.08em;width:auto;margin:0 1px;"
    else:
        style = "display:block;margin:0.8em auto;max-width:100%;height:auto;width:auto;"
    return f'<img src="{src}" style="{style}" class="wx-math-{kind}" alt="{ihtml.escape(tex)}">'

html = MATH_RE.sub(math_repl, html)
print("      math images:", html.count('class="wx-math'))

# ----------------------------------------------------------------------------
# 4. embed local figures as base64
# ----------------------------------------------------------------------------
print("[4/5] embedding figures as base64")
def _img(m):
    attrs, src = m.group(1), m.group(2)
    if src.startswith(("data:", "http://", "https://", "//")):
        return m.group(0)
    path = src if os.path.isabs(src) else os.path.normpath(os.path.join(BASE, src))
    if not os.path.exists(path):
        print("      WARN missing image:", path); return m.group(0)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg",
            "gif":"image/gif","webp":"image/webp","svg":"image/svg+xml"}.get(ext,"image/png")
    data = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f'<img{attrs} src="data:{mime};base64,{data}" style="max-width:100%;height:auto;display:block;margin:1.1em auto;border-radius:6px;"'

html = re.sub(r'<img\b([^>]*?)\bsrc="([^"]+)"', _img, html)
print("      base64 figures:", html.count("data:image/") - html.count('class="wx-math'))

# ----------------------------------------------------------------------------
# 5. inline WeChat-safe styles onto block elements
# ----------------------------------------------------------------------------
print("[5/5] inlining WeChat-safe styles")
STYLES = {
    "h1":  "font-size:22px;font-weight:700;color:#2f2f2f;text-align:center;border-bottom:1px solid #e6e6e6;padding-bottom:.4em;margin:1.6em 0 .6em;line-height:1.4;",
    "h2":  "font-size:19px;font-weight:700;color:#2f2f2f;border-left:4px solid #576b95;padding-left:.5em;margin:1.6em 0 .6em;line-height:1.4;",
    "h3":  "font-size:17px;font-weight:700;color:#2f2f2f;margin:1.6em 0 .6em;line-height:1.4;",
    "p":   "margin:.9em 0;line-height:1.85;color:#3f3f3f;word-break:break-word;",
    "a":   "color:#576b95;word-break:break-all;",
    "strong": "color:#1f1f1f;",
    "blockquote": "margin:1em 0;padding:.6em 1em;background:#fafafa;border-left:4px solid #d9d9d9;color:#8a8a8a;border-radius:0 4px 4px 0;line-height:1.85;",
    "pre": "background:#f6f8fa;border:1px solid #eaecef;border-radius:8px;padding:14px 16px;overflow-x:auto;margin:1.1em 0;font-size:13.5px;line-height:1.7;font-family:SFMono-Regular,Consolas,Liberation Mono,Menlo,monospace;",
    "table": "border-collapse:collapse;width:100%;margin:1.2em 0;font-size:14.5px;",
    "thead": "",
    "tbody": "",
    "tr":  "",
    "th":  "border:1px solid #e6e6e6;padding:8px 10px;text-align:left;background:#f2f6fc;font-weight:700;",
    "td":  "border:1px solid #e6e6e6;padding:8px 10px;text-align:left;",
    "hr":  "border:none;border-top:1px solid #e6e6e6;margin:2em 0;",
    "ul":  "padding-left:1.6em;margin:.6em 0;",
    "ol":  "padding-left:1.6em;margin:.6em 0;",
    "li":  "margin:.4em 0;line-height:1.85;",
}

def merge_style(tag_open, add):
    if not add: return tag_open
    existing = re.search(r'\bstyle="([^"]*)"', tag_open)
    if existing:
        old = existing.group(1).rstrip().rstrip(';')
        new_style = (old + ";" + add).strip(';')
        return tag_open[:existing.start()] + f'style="{new_style}"' + tag_open[existing.end():]
    return tag_open.rstrip(">") + f' style="{add}">'

for tag, style in STYLES.items():
    pat = re.compile(rf'<({tag})\b([^>]*)>', re.I)
    html = pat.sub(lambda m: merge_style(f'<{m.group(1)}{m.group(2)}>', style), html)

# blockquote paragraphs get a tighter margin
html = re.sub(r'<blockquote\b[^>]*>\s*<p\b[^>]*>', lambda m: m.group(0).replace('<p', '<p style="margin:.3em 0;"', 1), html, flags=re.I)

# ----------------------------------------------------------------------------
# assemble final document
# ----------------------------------------------------------------------------
title = title_from_frontmatter()
doc = (
    "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    f"<title>{ihtml.escape(title)}</title>\n"
    "</head>\n"
    "<body style=\"max-width:740px;margin:0 auto;padding:24px 18px 60px;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;font-size:16px;line-height:1.85;color:#3f3f3f;word-break:break-word;\">\n"
    f"{html}\n"
    "</body>\n</html>\n"
)

open(OUT, "w", encoding="utf-8").write(doc)
print("DONE ->", OUT, "(", len(doc), "bytes )")
