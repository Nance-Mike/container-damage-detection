# -*- coding: utf-8 -*-
"""Verify the compiled paper PDF: page numbers, headings, ?? refs, fonts."""
import os
import re

import fitz

PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main.pdf')
doc = fitz.open(PDF)
print('pages:', len(doc))

full = []
for i, page in enumerate(doc):
    t = page.get_text()
    full.append((i + 1, t))
    if i < 2 or i >= len(doc) - 3:
        pass

all_text = '\n'.join(t for _, t in full)
print('question marks "??":', all_text.count('??'))

heads = ['一、问题重述', '二、问题分析', '三、模型假设', '四、符号说明',
         '五、数据处理与探索性分析', '六、问题一：基于极值理论',
         '七、问题二：基于 YOLOv8', '八、问题三：模型多维度评估',
         '九、模型评价与推广', '参考文献', '附录A', '附录B', '附录C']
for h in heads:
    pages = [p for p, t in full if h.replace(' ', '') in t.replace(' ', '')]
    print('%-24s -> page %s' % (h, pages[:5]))

fonts = set()
for page in doc:
    for f in page.get_fonts():
        fonts.add((f[3], f[4]))
print('fonts:')
for name, typ in sorted(fonts):
    print('  ', name, typ)

# abstract page fits one page: page 1 should contain 摘要 and 关键词
p1 = full[0][1]
print('page1 has 摘要:', '摘' in p1, '| 关键词:', '关键词' in p1)
print('page2 starts with:', p2 := full[1][1][:80].replace('\n', ' '))

# page size
print('page size (pt):', doc[0].rect)

# caption numbering sample
caption_lines = []
for page in doc:
    for line in page.get_text().splitlines():
        s = line.strip()
        if re.match(r'^(图|表)\s*\d+-\d+\s', s) or s.startswith('附录'):
            if s not in caption_lines:
                caption_lines.append(s)
print('sample captions:')
for c in caption_lines[:26]:
    print('  ', c)

# appendix code rendering
t_app = '\n'.join(doc[i].get_text() for i in range(26, min(30, len(doc))))
print('appendix has code:', 'def train_baseline' in t_app,
      '| Chinese comment renders:', '项目根目录' in t_app)
print('---- page 27/28 first 40 lines ----')
for i in (26, 27):
    lines = doc[i].get_text().splitlines()
    print('page', i + 1, 'lines:', len(lines))
    for line in lines[:20]:
        print('   ', repr(line))

# check all images embedded (xref count of images)
img_count = sum(len(page.get_images(full=True)) for page in doc)
print('embedded images:', img_count)
