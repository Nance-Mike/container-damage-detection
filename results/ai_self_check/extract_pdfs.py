import fitz
import os

BASE = r"C:\Users\administrator\Desktop\2026数模国赛\选题D"
OUT = os.path.join(BASE, "results", "ai_self_check")
os.makedirs(OUT, exist_ok=True)

FILES = {
    "main": os.path.join(BASE, "论文", "main.pdf"),
    "problem_D": os.path.join(BASE, "选题D.pdf"),
    "data_desc": os.path.join(BASE, "集装箱数据描述 - 初赛.pdf"),
    "dataset_desc": os.path.join(BASE, "数据集说明.pdf"),
}

for key, path in FILES.items():
    doc = fitz.open(path)
    lines = [f"@@TOTAL {doc.page_count}@@"]
    total_chars = 0
    for i, page in enumerate(doc):
        text = page.get_text("text")
        total_chars += len(text.strip())
        lines.append(f"@@PAGE {i+1} START@@")
        lines.append(text)
        lines.append(f"@@PAGE {i+1} END@@")
    out_path = os.path.join(OUT, f"{key}_pages.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{key}: pages={doc.page_count}, chars={total_chars}, avg={total_chars/max(doc.page_count,1):.1f}/page")
    doc.close()
