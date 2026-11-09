"""Post-hoc, no-refit review of one fixed molecular split per setting."""
import argparse,csv,json,hashlib,gzip
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from sklearn.metrics import average_precision_score,balanced_accuracy_score
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--return-root',type=Path,required=True,
                    help='Directory containing the extracted prepare/ and screen/ return packages')
parser.add_argument('--output-dir',type=Path,required=True,
                    help='Directory to create for metrics_review.json')
args=parser.parse_args()
ROOT=args.return_root.expanduser().resolve()
SOURCE=ROOT/'screen'
if not SOURCE.is_dir() or not (ROOT/'prepare').is_dir():
 parser.error('--return-root must contain prepare/ and screen/ directories')
OUT=args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True,exist_ok=True)
N_BOOT=2000
SEED=20261109
rows=list(csv.DictReader((SOURCE/'test_predictions.csv').open()))
metrics=json.loads((SOURCE/'metrics.json').read_text())
NUM=('q_lower','q_upper','q_midpoint','test_molecule_weight','resolved_molecule_weight','status_molecule_weight')
for row in rows:
 for k in NUM: row[k]=float(row[k])
 row['masked_nll']=float(row['masked_nll']) if row['masked_nll'] else None
 row['replicate']=int(row['replicate'])
blocks=defaultdict(list)
for row in rows: blocks[(row['split'],row['method'])].append(row)

def weights(block):
 c=Counter(r['group_id'] for r in block)
 return np.array([1/c[r['group_id']] for r in block])
def mean(v,w):return float(np.average(v,weights=w))
def assert_close(a,b):
 if a is None or b is None: assert a is None and b is None;return
 assert np.isclose(a,b,rtol=1e-10,atol=1e-12),(a,b)
def stats(block):
 w=weights(block);status=np.array([r['window_reference'] for r in block]);y=(status=='definite_yes').astype(int)
 lo=np.array([r['q_lower'] for r in block]);hi=np.array([r['q_upper'] for r in block]);q=(lo+hi)/2
 assert (0<=lo).all() and (lo<=hi).all() and (hi<=1+1e-12).all()
 resolved=status!='unresolved';rb=[r for r in block if r['window_reference']!='unresolved'];rw=weights(rb);yr=y[resolved]
 fixed=(q-y)**2
 lower=np.where(resolved,fixed,np.minimum(q*q,(1-q)**2));upper=np.where(resolved,fixed,np.maximum(q*q,(1-q)**2))
 best_q=np.clip(yr,lo[resolved],hi[resolved]);worst=np.maximum((lo[resolved]-yr)**2,(hi[resolved]-yr)**2)
 return {'rows':len(block),'groups':len(set(r['group_id'] for r in block)),
  'resolved_rows':int(resolved.sum()),'positive_rows':int(yr.sum()),'resolved_groups':len(set(r['group_id'] for r in rb)),
  'resolved_weighted_prevalence':mean(yr,rw),
  'brier':mean((q[resolved]-yr)**2,rw),
  'average_precision':float(average_precision_score(yr,q[resolved],sample_weight=rw)),
  'balanced_accuracy_at_0_5':float(balanced_accuracy_score(yr,q[resolved]>=.5,sample_weight=rw)),
  'all_reference_brier_lower':mean(lower,w),'all_reference_brier_upper':mean(upper,w),
  'resolution_diagnostic':{
   'AP_using_q_lower':float(average_precision_score(yr,lo[resolved],sample_weight=rw)),
   'AP_using_q_upper':float(average_precision_score(yr,hi[resolved],sample_weight=rw)),
   'optimistic_label_aware_Brier_within_q_interval':mean((best_q-yr)**2,rw),
   'pessimistic_Brier_within_q_interval':mean(worst,rw),
   'mean_width_all_rows':mean(hi-lo,w)},
  'positive_probability_mean':mean(q[resolved][yr==1],rw[yr==1]),
  'negative_probability_mean':mean(q[resolved][yr==0],rw[yr==0])}

def groupmeans(block,values):
 s=defaultdict(list)
 for r,v in zip(block,values):s[r['group_id']].append(float(v))
 return np.array([np.mean(s[g]) for g in sorted(s)])
def boot(values,seed):
 rng=np.random.default_rng(seed);means=np.empty(N_BOOT)
 for i in range(N_BOOT):means[i]=values[rng.integers(0,len(values),len(values))].mean()
 return {'mean':float(np.mean(values)),'percentile_95_ci':np.quantile(means,[.025,.975]).tolist(),'groups':len(values),'bootstrap_repetitions':N_BOOT,'seed':seed}

report={'interpretation':'Post-hoc diagnostic on supplied fixed test partitions. No retraining. Bootstrap intervals condition on these fitted models and splits; they are not independent external validation or across-split training uncertainty. q_lower/q_upper AP values are sensitivity scores, not AP confidence bounds. Optimistic Brier uses the outcome to choose a point inside each q interval and is not an available predictor.',
 'source_predictions_sha256':hashlib.sha256((SOURCE/'test_predictions.csv').read_bytes()).hexdigest(),'row_count':len(rows),'recomputed':[],'main_comparison':{}}
for m in metrics:
 key=(m['split'],m['method']);b=blocks[key];st=stats(b);rep=stats([r for r in b if r['replicate']==3]);
 for name,computed in [('all_repeats',st),('repeat3_reference',rep)]:
  for k in ('brier','average_precision','balanced_accuracy_at_0_5','all_reference_brier_lower','all_reference_brier_upper'):assert_close(computed[k],m[name][k])
 # Compare all exported weights with fresh per-subset counts.
 allw=weights(b);resolved_block=[r for r in b if r['window_reference']!='unresolved'];rc=Counter(r['group_id'] for r in resolved_block)
 sc=Counter((r['group_id'],r['window_reference']) for r in b)
 for r,w in zip(b,allw):
  assert_close(r['test_molecule_weight'],w)
  assert_close(r['resolved_molecule_weight'],0 if r['window_reference']=='unresolved' else 1/rc[r['group_id']])
  assert_close(r['status_molecule_weight'],1/sc[(r['group_id'],r['window_reference'])])
 nll=None
 if m['method']!='binary_window':
  nll=mean([r['masked_nll'] for r in b],allw);assert_close(nll,m['masked_nll_molecule_weighted'])
  for s in ['definite_yes','definite_no','unresolved']:
   sb=[r for r in b if r['window_reference']==s]
   assert_close(mean([r['masked_nll'] for r in sb],weights(sb)),m['by_reference_status'][s]['masked_nll_molecule_weighted'])
 # Every CSV row matches corresponding returned run prediction without tolerance for string IDs.
 run=json.loads((SOURCE/'runs'/f"{m['job_id']}.json").read_text())
 assert len(run['predictions'])==len(b)
 for csvrow,jsonrow in zip(b,run['predictions']):
  for k,v in jsonrow.items():
   a=csvrow[k]
   if isinstance(v,bool): assert a==str(v)
   elif isinstance(v,(float,int)):assert_close(float(a),v)
   else:assert a==v,(k,a,v)
 report['recomputed'].append({'split':m['split'],'method':m['method'],'masked_nll':nll,'all_repeats':st,'repeat3_reference':rep})
for si,split in enumerate(['random','scaffold']):
 joint=blocks[(split,'joint_censored')];ind=blocks[(split,'independent_censored')]
 identity=lambda r:tuple(r[k] for k in ('group_id','sample_id','protocol','replicate','target_curve_id','viability_curve_id'))
 assert [identity(r) for r in joint]==[identity(r) for r in ind]
 delta=np.array([j['masked_nll']-i['masked_nll'] for j,i in zip(joint,ind)])
 w=weights(joint);totalw=w.sum();by_status={}
 for k,s in enumerate(['definite_yes','definite_no','unresolved']):
  take=np.array([r['window_reference']==s for r in joint]);sb=[r for r in joint if r['window_reference']==s]
  by_status[s]={'rows':int(take.sum()),'within_status_equal_molecule_delta':boot(groupmeans(sb,delta[take]),SEED+10*si+k+1),'additive_contribution_to_overall_delta':float((w[take]*delta[take]).sum()/totalw),'overall_weight_fraction':float(w[take].sum()/totalw)}
 overall=boot(groupmeans(joint,delta),SEED+10*si)
 assert_close(overall['mean'],sum(x['additive_contribution_to_overall_delta'] for x in by_status.values()))
 report['main_comparison'][split]={'difference_direction':'joint minus independent; negative is better for joint','overall':overall,'relative_NLL_reduction_percent':-overall['mean']/mean([r['masked_nll'] for r in ind],weights(ind))*100,'by_reference_status':by_status}
# Cluster diagnostic preserves the molecule-weighted estimand, resampling whole
# represented scaffold clusters; counts are reweighted through the ratio denominator.
derived=json.loads(gzip.decompress((SOURCE/'derived_records.json.gz').read_bytes()))
group_scaffold={}
for r in derived:
 if r['group_id'] in group_scaffold: assert group_scaffold[r['group_id']]==r['scaffold_id']
 group_scaffold[r['group_id']]=r['scaffold_id']
def scaffold_boot(block,values,seed):
 by_group=defaultdict(list)
 for r,v in zip(block,values):by_group[r['group_id']].append(float(v))
 by_scaffold=defaultdict(list)
 for g,vals in by_group.items():by_scaffold[group_scaffold[g]].append(float(np.mean(vals)))
 names=sorted(by_scaffold);sums=np.array([sum(by_scaffold[k]) for k in names]);counts=np.array([len(by_scaffold[k]) for k in names])
 rng=np.random.default_rng(seed);means=np.empty(N_BOOT)
 for b in range(N_BOOT):
  take=rng.integers(0,len(names),len(names));means[b]=sums[take].sum()/counts[take].sum()
 return {'mean':float(sums.sum()/counts.sum()),'percentile_95_ci':np.quantile(means,[.025,.975]).tolist(),'scaffold_clusters':len(names),'molecular_groups':int(counts.sum()),'largest_scaffold_molecular_groups':int(counts.max()),'largest_scaffold_group_fraction':float(counts.max()/counts.sum()),'bootstrap_repetitions':N_BOOT,'seed':seed,'largest_five_clusters':[{'scaffold_id':names[i],'molecular_groups':int(counts[i]),'mean_delta':float(sums[i]/counts[i])} for i in np.argsort(-counts)[:5]],'resampling_unit':'whole represented test scaffold cluster; ratio-of-sums retains equal molecule weight; fixed fitted model diagnostic'}
joint=blocks[('scaffold','joint_censored')];ind=blocks[('scaffold','independent_censored')]
delta=np.array([j['masked_nll']-i['masked_nll'] for j,i in zip(joint,ind)])
report['scaffold_cluster_diagnostic']={'all':scaffold_boot(joint,delta,SEED+100)}
for k,status in enumerate(['definite_yes','definite_no','unresolved']):
 take=np.array([r['window_reference']==status for r in joint]);sb=[r for r in joint if r['window_reference']==status]
 report['scaffold_cluster_diagnostic'][status]=scaffold_boot(sb,delta[take],SEED+101+k)

report['validation']={'all_exported_weights_recomputed':True,'CSV_matches_all_eight_run_prediction_arrays':True,'all_reported_Brier_AP_BA_all_reference_Brier_bounds_recomputed':True,'all_reported_overall_and_status_NLL_recomputed':True,'fit_calls':0}
(OUT/'metrics_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'main_comparison':report['main_comparison'],'validation':report['validation']},indent=2))
