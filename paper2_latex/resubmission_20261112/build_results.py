"""Build all numerical tables and figures from fixed aggregate inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;I=ROOT/'inputs';T=ROOT/'tables';F=ROOT/'figures'
MET=['class_0_coverage','class_1_coverage','minimum_class_coverage','macro_csy','active_false_clearance','inactive_false_discard','operational_defer_rate']
PRIMARY=['Tox21_NR_AhR','Tox21_NR_ER','Tox21_SR_ARE','Tox21_SR_ATAD5','Tox21_SR_MMP','Tox21_SR_p53']
BASE='mondrian_lac_empty_to_full';HIST='tame_label0_historical';MATCH='matched_label0_historical_omitted_label_probability'
def latex(name,headers,rows,align=None):
 align=align or ('l'+'r'*(len(headers)-1))
 txt='\\begin{tabular}{'+align+'}\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
 txt+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\n'
 (T/name).write_text(txt)
def main():
 T.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
 p=pd.read_csv(I/'panel_metrics.csv.gz');c=pd.read_csv(I/'core_cell_metrics.csv.gz');z=pd.read_csv(I/'all_paired_contrasts.csv.gz')
 e=p[(p.phase=='retrospective_epa')&(p.alpha==.1)&(p.population=='full_eligible_labeled')]
 def row(method,model='stack',panel='primary_six'):
  d=e[(e.method==method)&(e.model==model)&(e.panel==panel)];assert len(d)==1,(method,model,panel);return d.iloc[0]
 checks=[]
 for panel in ['primary_six','all_assays']:
  for model in ['logistic','stack','equal_logit']:
   for method in [BASE,HIST,'tame_label1_active',MATCH,'matched_label1_active_omitted_label_probability']:
    s=c[(c.phase=='retrospective_epa')&(c.alpha==.1)&(c.model==model)&(c.method==method)]
    if panel=='primary_six':s=s[s.endpoint.isin(PRIMARY)]
    target=row(method,model,panel);err=max(abs(s[m].mean()-target[m]) for m in MET)
    assert err<1e-12;checks.append(dict(panel=panel,model=model,method=method,cells=len(s),max_abs_error=err))
 choices=[('Mondrian LAC baseline',BASE),('TAME, label 0',HIST),('Omitted-label probability',MATCH),('Probability uncertainty','matched_label0_historical_uncertainty'),('Chemical similarity','matched_label0_historical_ad'),('Seeded random selection','matched_label0_historical_random'),('TAME, label 1','tame_label1_active')]
 latex('historical.tex',['Rule',r'$C_{\min}$',r'$E_0$',r'MacroCSY','Deferral'],[[name]+[f'{100*row(m)[k]:.2f}' for k in ['minimum_class_coverage','inactive_false_discard','macro_csy','operational_defer_rate']] for name,m in choices])
 latex('predictors.tex',['Predictor',r'$C_0$',r'$C_1$',r'$C_{\min}$','MacroCSY','Deferral'],[[label]+[f'{100*row(BASE,m)[k]:.2f}' for k in ['class_0_coverage','class_1_coverage','minimum_class_coverage','macro_csy','operational_defer_rate']] for m,label in [('logistic','Logistic'),('stack','Stack'),('equal_logit','Equal-logit')]])
 refs=e[(e.panel=='primary_six')&(e.model.isin(['codrug','codrug_margin_swap','kmm_cp']))]
 refrows=[]
 for _,r in refs.iterrows():
  method=r['method']
  if r['model'].startswith('codrug'):
   label='CoDrug, '+('swapped' if r['model']=='codrug_margin_swap' else 'printed')+' / '+('global' if method.endswith('global') else 'candidate')
  else:
   label=('Selective KMM' if method.startswith('skmm') else 'KMM')+' / '+('global' if method.endswith('global') else 'predicted' if 'predicted' in method else 'candidate')
  refrows.append([label]+[f'{100*r[k]:.2f}' for k in ['active_false_clearance','inactive_false_discard','minimum_class_coverage','macro_csy','operational_defer_rate']])
 assert len(refrows)==10
 latex('full_references.tex',['Reference configuration',r'$E_1$',r'$E_0$',r'$C_{\min}$','MacroCSY','Deferral'],refrows)
 gates=pd.read_csv(I/'gate_audits.csv.gz')
 gates=gates[gates['view'].isin(['physchem_descriptors','component_and_stack_logits'])]
 gatecells=gates.groupby(['phase','endpoint','seed','model']).active.all().reset_index()
 mechanisms={phase:pd.read_csv(I/'mechanism'/(phase+'.csv')) for phase in ['retrospective_epa','independent_random','independent_scaffold']}
 records=[]
 configs=[('EPA, historical six','retrospective_epa',PRIMARY),('EPA, all twelve','retrospective_epa',None)]
 for phase,split in [('independent_random','random'),('independent_scaffold','scaffold')]:
  for endpoint in ['AMES','CYP2D6_Veith','DILI','hERG']:configs.append((endpoint.replace('_Veith','')+' / '+split,phase,[endpoint]))
 for label,phase,endpoints in configs:
  a=mechanisms[phase];a=a[(a.predictor=='stack')&(a.alpha==.1)&(a.baseline_class==0)]
  g=gatecells[(gatecells.phase==phase)&(gatecells.model=='stack')]
  if endpoints is not None:a=a[a.endpoint.isin(endpoints)];g=g[g.endpoint.isin(endpoints)]
  r={'setting':label,'cells':len(a),'gate_pass':int(g.active.sum())}
  for out,col in [('added','defer_budget_method'),('method_only','method_only_deferred'),('reference_only','reference_only_deferred'),('rescued','method_error_rescues'),('reference_rescued','reference_error_rescues')]:r[out]=int(a[col].sum())
  records.append(r)
 assert records[0]['added']==0 and records[0]['gate_pass']==30
 assert records[1]['added']==21 and records[1]['rescued']==0
 latex('activation.tex',['Setting','Pass / cells','Added','A-only / P-only','Rescued'],[[r['setting'],f"{r['gate_pass']} / {r['cells']}",r['added'],f"{r['method_only']} / {r['reference_only']}",f"{r['rescued']} / {r['reference_rescued']}"] for r in records])
 pd.DataFrame(records).to_csv(T/'activation.csv',index=False)
 # Denominators come from the baseline, joined to the recorded gate calibration counts.
 b=c[(c.method==BASE)&(c.alpha==.1)&(c.model=='stack')].copy()
 gc=gates[(gates.model=='stack')&(gates['view']=='physchem_descriptors')][['phase','endpoint','seed','source_n']]
 b=b.merge(gc,on=['phase','endpoint','seed'],validate='one_to_one')
 countrows=[]
 def span(s):
  lo,hi=int(s.min()),int(s.max());return str(lo) if lo==hi else f'{lo}--{hi}'
 for (phase,endpoint),a in b.groupby(['phase','endpoint']):
  countrows.append([phase.replace('independent_','').replace('retrospective_',''),endpoint.replace('Tox21_','').replace('_',r'\_')]+[span(a[k]) for k in ['n_target','n_eligible','class_0_n','class_1_n','source_n']])
 latex('counts.tex',['Phase','Assay','Raw','Eligible',r'$N_0$',r'$N_1$','Calib.'],countrows,align='llrrrrr')
 plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axs=plt.subplots(1,2,figsize=(10.6,3.6))
 for kind,label,color in [('uncertainty','Probability uncertainty','#d95f02'),('ad','Chemical similarity','#1b7789'),('random','Seeded random','#777777')]:
  d=e[(e.model=='stack')&(e.panel=='primary_six')&e.method.str.startswith('mondrian_defer_'+kind+'_')].sort_values('operational_defer_rate')
  for ax,m in zip(axs,['minimum_class_coverage','macro_csy']):ax.plot(100*d.operational_defer_rate,100*d[m],marker='o',ms=3,lw=1.3,label=label,color=color)
 for ax,m,label in zip(axs,['minimum_class_coverage','macro_csy'],['Mean minimum-class coverage (%)','Mean macro correct-singleton yield (%)']):
  r=row(HIST);ax.scatter(100*r.operational_defer_rate,100*r[m],s=115,c='#49346f',marker='*',zorder=5,label='TAME label 0 = matched probability');ax.set(xlabel='Operational deferral (%)',ylabel=label);ax.grid(alpha=.15)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.06),fontsize=9);fig.tight_layout(rect=[0,.08,1,1])
 for ext in ['pdf','png']:fig.savefig(F/('workload_tradeoff.'+ext),bbox_inches='tight',dpi=180)
 plt.close(fig)
 d=c[(c.phase=='retrospective_epa')&(c.model=='stack')&(c.alpha==.1)]
 endpoints=sorted(d.endpoint.unique());fig,ax=plt.subplots(figsize=(7.6,5.6))
 for m,off,color,label,mark in [(BASE,-.17,'#969696','Mondrian baseline','o'),(HIST,0,'#252c67','TAME label 0','s'),(MATCH,.17,'#d95f02','Matched probability','+')]:
  a=d[d.method==m].groupby('endpoint').minimum_class_coverage.mean().reindex(endpoints)
  ax.scatter(100*a,np.arange(len(endpoints))+off,color=color,label=label,marker=mark,s=22)
 ax.set_yticks(np.arange(len(endpoints)),[s.replace('Tox21_','').replace('_','-') for s in endpoints]);ax.invert_yaxis();ax.set_xlabel('Mean minimum-class coverage across five fits (%)');ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',frameon=True,fontsize=9);fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(F/('assay_coverage.'+ext),bbox_inches='tight',dpi=180)
 plt.close(fig)
 historical={'baseline_minimum_coverage':float(row(BASE).minimum_class_coverage),'tame_minimum_coverage':float(row(HIST).minimum_class_coverage),'coverage_gain_pp':100*float(row(HIST).minimum_class_coverage-row(BASE).minimum_class_coverage),'extra_deferral_pp':100*float(row(HIST).operational_defer_rate-row(BASE).operational_defer_rate),'macro_csy_change_pp':100*float(row(HIST).macro_csy-row(BASE).macro_csy)}
 x=z[(z.phase=='retrospective_epa')&(z.panel=='primary_six')&(z.model=='stack')&(z.method==HIST)&(z.reference==MATCH)&(z.metric=='minimum_class_coverage')];assert len(x)==1
 historical.update(matched_difference_pp=float(x.iloc[0].delta*100),matched_ci_pp=[float(x.iloc[0].ci_95_low*100),float(x.iloc[0].ci_95_high*100)])
 report={'historical':historical,'activation':records,'cell_to_panel_checks':checks,'bootstrap':'Archived paired intervals retained; molecule-level bootstrap was not rerun from aggregates.'}
 (ROOT/'NUMERICAL_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Built tables and figures; 30 cell-to-panel checks passed.')
if __name__=='__main__':main()
