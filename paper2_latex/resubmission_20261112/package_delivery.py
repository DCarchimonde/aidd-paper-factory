"""Build the complete TAME delivery, including lossless returned numerical tables."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def zip_paths(dest,paths):
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(paths):z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True,type=Path);ap.add_argument('--commit',required=True);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 assert (ROOT/'evidence/all_cell_metrics.csv.gz').is_file()
 paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(s in {'qa','__pycache__'} for s in p.relative_to(ROOT).parts) and p.suffix not in {'.log','.aux','.out','.toc','.pyc'}]
 zip_paths(out/'TAME_LaTeX_Source.zip',[p for p in paths if p.suffix=='.tex' or (p.parent.name=='figures' and p.suffix=='.pdf')])
 zip_paths(out/'TAME_Numerical_Supplement.zip',[p for p in paths if p.name!='.gitignore' and not (p.parent==ROOT and p.suffix=='.pdf')])
 names=['manuscript.pdf','supplementary.pdf','response.pdf','cover_letter.pdf','README_CN.md','highlights.txt','collect_runtime_sources.py']
 for name in names:(out/name).write_bytes((ROOT/name).read_bytes())
 names+=['TAME_LaTeX_Source.zip','TAME_Numerical_Supplement.zip']
 receipt={'date':'2026-11-12','title':'Separating coverage gains from error targeting in molecular prediction sets','repository':'DCarchimonde/aidd-paper-factory','branch':'paper2-tame-resubmission-20261112','commit':a.commit,'status':'MANUSCRIPT_COMPLETE_RUNTIME_SOURCE_RECONCILIATION_PENDING','remaining':['Retrieve and inspect the executed CoDrug and KMM-CP sources.'],'scientific_training_performed_this_revision':False,'submitted_to_journal':False,'files':{n:{'sha256':sha(out/n),'bytes':(out/n).stat().st_size} for n in names},'verification':{n:json.loads((ROOT/n).read_text()) for n in ['NUMERICAL_VERIFICATION.json','ACTIVATION_IDENTITY_CHECK.json','BUILD_VERIFICATION.json','COLLECTOR_VERIFICATION.json','VISUAL_QA.json']}}
 (out/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 final=out/'TAME_Revision_20261112.zip'
 with zipfile.ZipFile(final,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names+['DELIVERY_RECEIPT.json']:z.write(out/n,n)
 with zipfile.ZipFile(final) as z:assert z.testzip() is None
 print(json.dumps({'file':str(final),'bytes':final.stat().st_size,'sha256':sha(final)}))
if __name__=='__main__':main()
