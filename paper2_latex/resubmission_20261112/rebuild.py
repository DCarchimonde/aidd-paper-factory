"""Rebuild figures, tables, algebra checks, and four PDF documents."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent
def run(args):
 p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout[-4000:]+p.stderr[-1000:])
def main():
 run([sys.executable,'build_results.py']);run([sys.executable,'check_activation_identity.py'])
 warnings={}
 for stem in ['manuscript','supplementary','response','cover_letter']:
  for _ in range(2):run(['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'])
  log=(ROOT/(stem+'.log')).read_text()
  warnings[stem]=[s for s in log.splitlines() if any(k in s for k in ['Overfull','undefined','LaTeX Warning'])]
 report={'documents':list(warnings),'warnings':warnings,'scientific_training_performed':False}
 (ROOT/'BUILD_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
