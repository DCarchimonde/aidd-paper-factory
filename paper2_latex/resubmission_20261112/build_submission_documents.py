"""Create editable highlights and interest declaration for the submission package."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'submission_assets'
TITLE = 'Separating coverage gains from error targeting in molecular prediction sets'

def document(title):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(1)
    sec.left_margin = sec.right_margin = Inches(1)
    for style_name in ['Normal', 'Title', 'List Bullet']:
        style = doc.styles[style_name]
        style.font.name = 'Times New Roman'
        style.font.size = Pt(11)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(10)
    doc.styles['Title'].font.size = Pt(16)
    doc.styles['Title'].font.bold = True
    doc.styles['Normal'].paragraph_format.line_spacing = 1.15
    doc.add_paragraph(title, style='Title')
    doc.core_properties.author = 'Siyuan Tong and Yuechen Wang'
    doc.core_properties.title = title
    return doc

def main():
    OUT.mkdir(exist_ok=True)
    doc = document('Highlights')
    doc.add_paragraph(TITLE)
    lines = (ROOT / 'highlights.txt').read_text().splitlines()
    assert len(lines) == 5 and all(len(s) <= 85 for s in lines)
    for line in lines:
        doc.add_paragraph(line, style='List Bullet')
    doc.save(OUT / 'Highlights.docx')
    doc = document('Declaration of interest')
    doc.add_paragraph(TITLE)
    doc.add_paragraph('Siyuan Tong and Yuechen Wang')
    doc.add_paragraph('The authors declare no competing interests.')
    doc.save(OUT / 'Declaration-of-Interest.docx')
    print('Created Highlights.docx and Declaration-of-Interest.docx')

if __name__ == '__main__':
    main()
