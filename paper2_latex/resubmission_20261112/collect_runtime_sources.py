"""Collect actual executed baseline sources without training or changing files."""
from pathlib import Path
import argparse,hashlib,json,subprocess,zipfile
from datetime import datetime,timezone
BASE='ae9c3927f0e34f86f58005c4ccea9d50ff5a7376'
PREFIX='paper2_admet_benchmark/revision_2026/full_baselines/'
EXPECTED={'codrug.py':'dd0e16ff8c9d76ac797afc7637d23e2320e1d424229f430e3976bf5e49a76f44','kmm_cp.py':'e57042e1121be80b49d3d26d63714af2b12cb6a720d69feb0d46c058f1fb7637'}
PRE_CODRUG='4f902c30f23548021f7c3ded5ba8c802bbf1164f6b68c26f9b1944d23eca7760'
def digest(b):return hashlib.sha256(b).hexdigest()
def git(root,*args):
 p=subprocess.run(['git','-C',str(root),*args],capture_output=True)
 return p.returncode,p.stdout,p.stderr
def identify(data,expected):
 lf=data.replace(b'\r\n',b'\n').replace(b'\r',b'\n')
 hashes={'raw':digest(data),'lf':digest(lf),'crlf':digest(lf.replace(b'\n',b'\r\n'))}
 matches=[k for k,v in hashes.items() if v==expected]
 return {'sha256':hashes['raw'],'normalized_lf_sha256':hashes['lf'],'recorded_sha256':expected,'matching_encoding':matches,'matches_recorded_source':bool(matches)}
def collect(repo,output,result_dir=None,pre_fix=None):
 repo=repo.resolve();output=output.resolve()
 if not (repo/PREFIX).is_dir():raise ValueError('Not the formal run repository: '+str(repo))
 if output.exists():raise FileExistsError('Output exists; choose a new --output path: '+str(output))
 entries={};report={'created_utc':datetime.now(timezone.utc).isoformat(),'reference_commit':BASE,'sources':{},'missing':[]}
 for name,expected in EXPECTED.items():
  p=repo/PREFIX/name
  if not p.is_file():report['missing'].append(name);continue
  data=p.read_bytes();entries['actual_sources/'+name]=data;report['sources'][name]=identify(data,expected)
  code,old,err=git(repo,'show',BASE+':'+PREFIX+name)
  if code==0:entries['reference_sources/'+name]=old
  code,diff,err=git(repo,'diff',BASE,'--',PREFIX+name)
  if code==0:entries['diff/'+name+'.patch']=diff
  else:report.setdefault('git_errors',{})[name]=err.decode(errors='replace')[:1500]
 code,head,_=git(repo,'rev-parse','HEAD');report['current_git_head']=head.decode().strip() if code==0 else None
 if pre_fix:
  data=pre_fix.read_bytes();entries['pre_scaffold_fix/codrug.py']=data;report['pre_scaffold_fix']=identify(data,PRE_CODRUG)
 if result_dir:
  for name in ['provenance.json','provenance.pre_scaffold_fix.json','auxiliary_source.json','experiment_complete.json']:
   p=result_dir/name
   if p.is_file():entries['run_records/'+name]=p.read_bytes()
 report['status']='MATCHED_RECORDED_SOURCES' if len(report['sources'])==2 and all(x['matches_recorded_source'] for x in report['sources'].values()) else 'COLLECTED_REQUIRES_SOURCE_RECONCILIATION'
 report['scope']='Baseline source, scoped diffs and optional provenance only; no training or raw molecular rows.'
 report['files']={name:digest(data) for name,data in entries.items()}
 entries['COLLECTION_REPORT.json']=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
 output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for name,data in entries.items():z.writestr(name,data)
 return report
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,default=Path.cwd());ap.add_argument('--output',type=Path,default=Path('TAME_Runtime_Sources_20261112.zip'));ap.add_argument('--result-dir',type=Path);ap.add_argument('--pre-fix-codrug',type=Path);a=ap.parse_args()
 r=collect(a.repo,a.output,a.result_dir,a.pre_fix_codrug);print(r['status']);print(a.output.resolve())
 if r['status']!='MATCHED_RECORDED_SOURCES':print('Return this ZIP for reconciliation. Do not retrain or overwrite the run sources.')
if __name__=='__main__':main()
