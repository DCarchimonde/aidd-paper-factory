"""Build Chinese handoff PDF using Pandoc, XeLaTeX and installed CJK fonts.

Default fonts are available in the delivery environment. On another system,
pass --mainfont and --monofont with installed names. No experiment is run.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
HEADER=r'''
\usepackage{listings}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true,frame=single,rulecolor=\color{gray},xleftmargin=4pt,xrightmargin=4pt}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small CHEMOLAB 拒稿重投 / 二作执行交接}
\fancyhead[R]{\small 方案版本 2026-11-04}
\fancyfoot[C]{\small\thepage}
\setlength{\headheight}{15pt}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
'''
FILTER=r'''
function Code(el)
  local mapping = {['\\']='\\textbackslash{}', ['_']='\\_\\allowbreak{}', ['%']='\\%', ['&']='\\&', ['#']='\\#', ['$']='\\$', ['{']='\\{', ['}']='\\}', ['~']='\\textasciitilde{}', ['^']='\\textasciicircum{}', ['/']='/\\allowbreak{}', ['.']='.\\allowbreak{}', ['-']='-\\allowbreak{}'}
  local s = el.text:gsub('.', function(c) return mapping[c] or c end)
  return pandoc.RawInline('latex', '\\texttt{' .. s .. '}')
end
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mainfont',default='Noto Sans CJK SC')
    p.add_argument('--monofont',default='DejaVu Sans Mono')
    a=p.parse_args()
    output=HERE/'CHEMOLAB_Second_Author_Plan_20261104.pdf'
    with tempfile.TemporaryDirectory() as temp:
        header=Path(temp)/'header.tex';header.write_text(HEADER,encoding='utf-8')
        filt=Path(temp)/'wrap.lua';filt.write_text(FILTER,encoding='utf-8')
        subprocess.run(['pandoc',str(HERE/'START_HERE_CN.md'),'--pdf-engine=xelatex','--listings',
            '--lua-filter',str(filt),'-V','mainfont='+a.mainfont,'-V','monofont='+a.monofont,
            '-V','fontsize=10pt','-V','papersize=a4','-V','geometry:margin=22mm',
            '-V','colorlinks=true','-V','linkcolor=blue','-V','urlcolor=blue',
            '-H',str(header),'-o',str(output)],check=True)
    print(output)

if __name__=='__main__':main()
