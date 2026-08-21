import re, os

BASE = r"C:\Users\administrator\Desktop\2026数模国赛\选题D"
SRC = os.path.join(BASE, "results", "ai_self_check", "main_pages.txt")

lines = open(SRC, encoding="utf-8").read().splitlines()

# ---- page map ----
def page_of(line_no):
    cur = None
    for ln in lines[:line_no]:
        m = re.match(r"@@PAGE (\d+) START@@", ln)
        if m:
            cur = int(m.group(1))
    return cur

pages = {}
for i, ln in enumerate(lines):
    for p in ["摘要", "关键词", "一、问题重述", "二、问题分析", "三、模型假设",
              "四、符号说明", "五、数据处理", "六、问题一", "七、问题二",
              "八、问题三", "九、模型评价", "参考文献", "附录A", "附录B", "附录C"]:
        if p not in pages and ln.startswith(p):
            pages[p] = page_of(i)
print("章节页码:", pages)

# ---- abstract: only page 1 block ----
def page_block(n):
    out, on = [], False
    for ln in lines:
        if ln.startswith(f"@@PAGE {n} START@@"):
            on = True
            continue
        if ln.startswith(f"@@PAGE {n} END@@"):
            break
        if on:
            out.append(ln)
    return "\n".join(out)

p1 = page_block(1)
m = re.search(r"摘要[：:]\s*(.*?)\s*关键词[：:]\s*(.*)", p1, re.S)
if m:
    ab_raw = m.group(1)
    ab_nospace = re.sub(r"\s", "", ab_raw)
    ab_cjk = re.sub(r"[^\u4e00-\u9fff0-9A-Za-z.%+\-]", "", ab_nospace)
    kw_raw = m.group(2)
    kw_nospace = re.sub(r"\s", "", kw_raw)
    kws = [x for x in re.split(r"[；;]", kw_nospace) if x.strip()]
    print(f"\n摘要正文: 含空白{len(ab_raw)}字符 / 去空白{len(ab_nospace)}字符 / 仅中英数{len(ab_cjk)}字符")
    print(f"关键词条数: {len(kws)} -> {kws}")

# ---- assumptions ----
si = next(i for i, ln in enumerate(lines) if ln.startswith("三、模型假设"))
ei = next(i for i, ln in enumerate(lines) if ln.startswith("四、符号说明"))
items = [ln for ln in lines[si:ei] if re.match(r"^\s*\d+[.、．]", ln.strip())]
print(f"\n模型假设条目数: {len(items)}")
for it in items:
    print("  ", re.sub(r"\s", "", it.strip())[:90])

# ---- figures / tables captions ----
figs, tabs = [], []
for ln in lines:
    s = re.sub(r"\s", "", ln.strip())
    if re.match(r"^图\d+-\d+", s) and len(s) > 8 and not s.startswith("图注"):
        figs.append(s[:60])
    if re.match(r"^表\d+-\d+", s) and len(s) > 8:
        tabs.append(s[:60])
print(f"\n图注数量(正文, 按行): {len(figs)}")
for f in figs:
    print("  ", f)
print(f"表注数量(正文, 按行): {len(tabs)}")
for t in tabs:
    print("  ", t)

# ---- references ----
ri = next(i for i, ln in enumerate(lines) if ln.startswith("参考文献"))
ai = next(i for i, ln in enumerate(lines) if ln.startswith("附录A"))
refs = [re.sub(r"\s", "", ln.strip()) for ln in lines[ri:ai] if re.search(r"\[\d+\]", ln)]
print(f"\n参考文献条目数: {len(refs)}")
for r in refs:
    print("  ", r[:110])

# ---- keyword probes ----
full = "\n".join(lines)
for probe in ["严重程度", "最多取", "AI使用", "人工智能", "声明", "ChatGPT", "GPT", "大模型", "大语言模型",
              "自动化工具", "生成式"]:
    cnt = full.count(probe)
    if cnt:
        print(f"\n命中 {probe!r}: {cnt} 次")

# ---- anonymous check (refined) ----
print("\n--- 匿名信息细化检索（正文 2-29 页） ---")
body = "\n".join(page_block(n) for n in range(2, 30))
for probe in ["大学", "学院", "姓名", "队伍", "队长", "队员", "学号", "指导教师", "老师", "手机", "电话", "邮箱", "@", "团队"]:
    hits = [ln for ln in body.splitlines() if probe in ln and "全国大学生" not in ln]
    if hits:
        print(f"{probe}: {len(hits)} 行 -> {[re.sub(chr(92)+'s','',h.strip())[:50] for h in hits[:3]]}")
