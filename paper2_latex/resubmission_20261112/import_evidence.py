"""Import the formal review bundle into fixed manuscript inputs."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parent
CORE=['mondrian_lac_empty_to_full','tame_label0_historical','tame_label1_active','matched_label0_historical_omitted_label_probability','matched_label1_active_omitted_label_probability']
METRICS=['class_0_coverage','class_1_coverage','minimum_class_coverage','macro_csy','active_false_clearance','inactive_false_discard','operational_defer_rate']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('review_dir',type=Path);a=ap.parse_args();src=a.review_dir
 inp=ROOT/'inputs';ev=ROOT/'evidence';inp.mkdir(exist_ok=True);ev.mkdir(exist_ok=True);(inp/'mechanism').mkdir(exist_ok=True)
 manifest=[]
 def copy(relative,dest,compressed=False):
  b=(src/relative).read_bytes();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(b,mtime=0) if compressed else b)
  manifest.append({'source':relative,'output':str(dest.relative_to(ROOT)),'source_sha256':hashlib.sha256(b).hexdigest(),'compression':'gzip; lossless original bytes' if compressed else 'none'})
 for stem in ['all_cell_metrics','all_panel_metrics','all_paired_contrasts','gate_stratified_own_budget','synthetic_mechanism_means']:
  copy('report/'+stem+'.csv',ev/(stem+'.csv.gz'),True)
 p=pd.read_csv(src/'report/all_panel_metrics.csv');cols=['phase','model','alpha','method','population','panel','cells','endpoints','seeds']+METRICS
 p[cols].to_csv(inp/'panel_metrics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 c=pd.read_csv(src/'report/all_cell_metrics.csv')
 c[(c.method.isin(CORE))&(c.population=='full_eligible_labeled')].to_csv(inp/'core_cell_metrics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 for n in ['all_paired_contrasts','gate_stratified_own_budget','synthetic_mechanism_means']:copy('report/'+n+'.csv',inp/(n+'.csv.gz'),True)
 for n in ['BUNDLE_SCOPE.json','protocol_snapshot.json','provenance.json','provenance.pre_scaffold_fix.json']:copy(n,inp/n)
 copy('experiment_complete.json',inp/'execution_summary.json')
 copy('synthetic_mechanisms/replicate_metrics.csv',inp/'synthetic_replicate_metrics.csv.gz',True)
 copy('synthetic_mechanisms/report.json',inp/'synthetic_report.json')
 for phase in ['development','retrospective_epa','independent_random','independent_scaffold']:
  copy('report/mechanism/'+phase+'/cell_counts.csv',inp/'mechanism'/(phase+'.csv'))
 rows=[]
 for f in sorted((src/'shared').rglob('audits.json')):
  phase,endpoint,seed,model=f.parent.name.split('__')
  for view,d in json.loads(f.read_text()).items():rows.append(dict(phase=phase,endpoint=endpoint,seed=int(seed),model=model,**d))
 pd.DataFrame(rows).to_csv(inp/'gate_audits.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 (inp/'SOURCE_MANIFEST.json').write_text(json.dumps({'files':manifest,'compact_panel_columns':cols,'core_methods':CORE,'raw_archive_sha256':'b4b30da53a8ea1a89d681d737571e1f971d7a51dfbd301075febbe70897d6083'},indent=2)+'\n')
 print('Imported',len(c),'cell rows,',len(p),'panel rows;',len(rows),'gate records')
if __name__=='__main__':main()
