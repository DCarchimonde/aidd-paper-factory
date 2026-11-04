"""Paired inference conditional on a fixed endpoint panel and fitted predictors.

Resamples molecular structure clusters jointly across endpoints, methods and
seeds. Seeds are repeated fitted models, never independent test observations.
No endpoint or method is chosen by the observed outcome. Intervals are pointwise
and exploratory on the already-exposed EPA cohort.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY = ['Tox21_NR_AhR','Tox21_NR_ER','Tox21_SR_ARE','Tox21_SR_ATAD5','Tox21_SR_MMP','Tox21_SR_p53']
METRICS = ['class_0_coverage','class_1_coverage','minimum_class_coverage',
           'macro_csy','active_false_clearance','inactive_false_discard',
           'active_fraction_among_cleared','clear_yield','defer_rate','empty_rate']


def _finite_json(value):
    if isinstance(value, dict): return {k:_finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_finite_json(v) for v in value]
    if isinstance(value, np.generic): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if 'model' in frame and 'predictor' not in frame:
        frame=frame.rename(columns={'model':'predictor'})
    required={'phase','endpoint','seed','predictor','method','alpha','sample_id',
              'structure_id','y','include_0','include_1','eligible'}
    if not required.issubset(frame): raise ValueError('missing columns: '+str(sorted(required-set(frame))))
    out=frame.copy()
    for c in ('include_0','include_1','eligible'):
        out[c]=pd.to_numeric(out[c],errors='raise')
        if not out[c].isin([0,1]).all(): raise ValueError(c+' must be binary')
    out['y']=pd.to_numeric(out.y,errors='raise')
    if not (out.y.isna() | out.y.isin([-1,0,1])).all(): raise ValueError('outcomes must be 0/1 or explicit missing (-1/NaN)')
    out=out[(out.eligible==1)&out.y.isin([0,1])].copy()
    if out.empty: raise ValueError('no eligible labeled predictions')
    for c in ('sample_id','structure_id'):
        if out[c].isna().any() or out[c].astype(str).eq('').any(): raise ValueError('empty '+c)
        out[c]=out[c].astype(str)
    key=['phase','endpoint','seed','predictor','method','alpha','sample_id']
    if out.duplicated(key).any(): raise ValueError('duplicate prediction identity')
    return out


def paired_cells(frame,method,reference,predictor,reference_predictor,alpha,endpoints,phase):
    f=validate_frame(frame)
    f=f[(f.phase==phase)&(f.alpha.astype(float)==alpha)&f.endpoint.isin(endpoints)]
    a=f[(f.method==method)&(f.predictor==predictor)].copy()
    b=f[(f.method==reference)&(f.predictor==reference_predictor)].copy()
    key=['endpoint','seed','sample_id']
    a=a.sort_values(key).reset_index(drop=True); b=b.sort_values(key).reset_index(drop=True)
    if a.empty or b.empty or not a[key].equals(b[key]): raise ValueError('paired methods must have identical eligible endpoint/seed/sample identities')
    if set(a.endpoint)!=set(endpoints): raise ValueError('incomplete requested endpoint panel')
    if not np.array_equal(a[['structure_id','y']].values,b[['structure_id','y']].values):
        raise ValueError('paired molecular identities or outcome labels differ')
    seed_sets=[set(g.seed) for _,g in a.groupby('endpoint')]
    if any(s!=seed_sets[0] for s in seed_sets): raise ValueError('endpoint seed panels differ')
    # Same test cohort across seeds; fit randomness cannot change the denominator.
    for endpoint,g in a.groupby('endpoint'):
        identity=[sub.sort_values('sample_id')[['sample_id','structure_id','y']].reset_index(drop=True)
                  for _,sub in g.groupby('seed')]
        if any(not x.equals(identity[0]) for x in identity[1:]): raise ValueError('test cohort changes across seeds: '+endpoint)
    structures=sorted(set(a.structure_id)); index={s:i for i,s in enumerate(structures)}
    cells=[]
    for (endpoint,seed),g in a.groupby(['endpoint','seed'],sort=True):
        h=b[(b.endpoint==endpoint)&(b.seed==seed)].sort_values('sample_id')
        g=g.sort_values('sample_id')
        if set(g.y)!={0,1}: raise ValueError('a requested primary cell lacks a class; report descriptively instead')
        cells.append((_sufficient(g,index),_sufficient(h,index)))
    return cells,structures,sorted(seed_sets[0])


def _sufficient(g,index):
    y=g.y.to_numpy(int); s=g[['include_0','include_1']].to_numpy(bool)
    size=s.sum(1); zero=y==0; one=~zero
    clear=(size==1)&s[:,0]; flag=(size==1)&s[:,1]
    terms=np.column_stack([np.ones(len(g)),zero,one,zero&s[:,0],one&s[:,1],
             zero&clear,one&flag,one&clear,zero&flag,clear,size!=1,size==0])
    out=np.zeros((len(index),12),float)
    np.add.at(out,np.array([index[x] for x in g.structure_id]),terms)
    return out


def _metrics(t):
    # Supports an arbitrary leading batch dimension. No nanmean over missing classes.
    n,n0,n1,c0,c1,cs0,cs1,fn,fp,clear,defer,empty=np.moveaxis(t,-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        cov0=c0/n0; cov1=c1/n1
        return np.stack([cov0,cov1,np.minimum(cov0,cov1),(cs0/n0+cs1/n1)/2,
                fn/n1,fp/n0,fn/clear,clear/n,defer/n,empty/n],axis=-1)


def bootstrap_comparison(cells,n_structures,replicates=2000,seed=44021):
    if replicates<20: raise ValueError('use at least 20 bootstrap replicates (2000 for the formal analysis)')
    if n_structures<2: raise ValueError('at least two molecular clusters are required')
    ca=np.stack([a for a,_ in cells]); cb=np.stack([b for _,b in cells])
    pa=_metrics(ca.sum(axis=1)).mean(axis=0); pb=_metrics(cb.sum(axis=1)).mean(axis=0)
    rng=np.random.default_rng(seed); draws=[]
    for start in range(0,replicates,100):
        w=rng.multinomial(n_structures,np.full(n_structures,1/n_structures),size=min(100,replicates-start))
        # Same molecule multiplicities in every endpoint, seed and both methods.
        ma=_metrics(np.einsum('bn,cnk->bck',w,ca)).mean(axis=1)
        mb=_metrics(np.einsum('bn,cnk->bck',w,cb)).mean(axis=1)
        draws.append(ma-mb)
    draws=np.concatenate(draws)
    results=[]
    for j,name in enumerate(METRICS):
        valid=np.isfinite(draws[:,j]); count=int(valid.sum())
        ci=np.quantile(draws[valid,j],[.025,.975]) if count>=.95*replicates else [np.nan,np.nan]
        results.append({'metric':name,'method_value':pa[j],'reference_value':pb[j],
            'delta':pa[j]-pb[j],'ci_95_low':ci[0],'ci_95_high':ci[1],
            'valid_bootstrap_replicates':count,'total_bootstrap_replicates':replicates,
            'interval_status':'pointwise_exploratory' if count>=.95*replicates else 'undefined_sparse_denominator'})
    return _finite_json(results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True,help='CSV or run directory recursively containing predictions.csv')
    p.add_argument('--additional-predictions',type=Path,nargs='+',default=[],help='Additional scored external CSV files or scored-only directories; never label-free import directories')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',required=True); p.add_argument('--reference',required=True)
    p.add_argument('--predictor',default='stack'); p.add_argument('--reference-predictor')
    p.add_argument('--phase',default='retrospective_epa'); p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--endpoints',nargs='+',default=PRIMARY)
    p.add_argument('--replicates',type=int,default=2000); p.add_argument('--seed',type=int,default=44021)
    a=p.parse_args()
    paths=[]
    for source in [a.predictions]+a.additional_predictions:
        paths.extend([source] if source.is_file() else sorted(source.rglob('predictions.csv')))
    if len({p.resolve() for p in paths})!=len(paths): raise ValueError('input prediction files overlap; supply each file once')
    if not paths: raise ValueError('no prediction files found')
    frames=[pd.read_csv(path,dtype={'sample_id':str,'structure_id':str}) for path in paths]
    frame=pd.concat(frames,ignore_index=True)
    ref_predictor=a.reference_predictor or a.predictor
    cells,structures,seeds=paired_cells(frame,a.method,a.reference,a.predictor,ref_predictor,a.alpha,a.endpoints,a.phase)
    report={'status':'retrospective_exploratory_comparison','method':a.method,'reference':a.reference,
       'predictor':a.predictor,'reference_predictor':ref_predictor,'alpha':a.alpha,'phase':a.phase,
       'endpoint_panel':a.endpoints,'seeds':seeds,'molecular_clusters':len(structures),
       'inference':'Fixed endpoint panel; shared molecular-cluster bootstrap; conditional on fitted models; seeds averaged, not independently resampled; no multiplicity-adjusted superiority claim.',
       'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
       'results':bootstrap_comparison(cells,len(structures),a.replicates,a.seed)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise FileExistsError('choose a new analysis output; never overwrite an earlier result')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.output),'status':report['status'],'cells':len(cells),'clusters':len(structures)}))

if __name__=='__main__': main()
