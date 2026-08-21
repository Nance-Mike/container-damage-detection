import re

src = r"C:\Users\administrator\Desktop\2026数模国赛\选题D\论文\sections\abstract.tex"
s = open(src, encoding="utf-8").read()
start = s.index("针对")
end = s.index("关键词：")
t = s[start:end]
t = re.sub(r"\\vspace\{[^}]*\}", "", t)
t = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", t)   # commands with/without args
t = t.replace("\\%", "%")
t = re.sub(r"[${}]", "", t)
t = re.sub(r"\s", "", t)               # all whitespace
print("摘要去空白字符数:", len(t))
print(t[:150])
