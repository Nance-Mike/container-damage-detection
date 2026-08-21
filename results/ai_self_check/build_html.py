import re, subprocess, pathlib

BASE = pathlib.Path(r"C:\Users\administrator\Desktop\2026数模国赛\选题D\results\ai_self_check")

proc = subprocess.run(
    ["pandoc", "-f", "markdown", "-t", "html", str(BASE / "AI自查报告.md")],
    capture_output=True, text=True, encoding="utf-8",
)
body = proc.stdout

# 去掉与页面 Hero 重复的第一个 h1
body = re.sub(r"<h1.*?</h1>", "", body, count=1, flags=re.S)

def badge_for(t):
    t = t.strip()
    if t.startswith("合格"):
        return '<span class="badge ok">合格</span>' + t[len("合格"):]
    if t.startswith("存在问题"):
        return '<span class="badge warn">存在问题</span>' + t[len("存在问题"):]
    if t.startswith("待核实"):
        return '<span class="badge info">待核实</span>' + t[len("待核实"):]
    if t.startswith("口径注意") or t.startswith("口径混用"):
        return '<span class="badge warn">' + t + "</span>"
    return None

def cell_repl(m):
    inner = m.group(1)
    b = badge_for(inner)
    if b is not None:
        return "<td>" + b + "</td>"
    s = inner.strip()
    if s in ("高", "中", "低"):
        cls = {"高": "high", "中": "mid", "低": "low"}[s]
        return '<td><span class="chip ' + cls + '">' + s + "</span></td>"
    return m.group(0)

body = re.sub(r"<td>(.*?)</td>", cell_repl, body, flags=re.S)
body = re.sub(r"(<table.*?</table>)", r'<div class="card">\1</div>', body, flags=re.S)

template = (BASE / "report_template.html").read_text(encoding="utf-8")
html = template.replace("<!--BODY-->", body)
(BASE / "AI自查报告.html").write_text(html, encoding="utf-8")
print("HTML written:", len(html), "chars")
