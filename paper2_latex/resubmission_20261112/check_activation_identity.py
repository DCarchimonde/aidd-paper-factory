"""Check the strict activation condition against the archived executable quantile."""
from pathlib import Path
import json
import numpy as np
from inputs.weighted_threshold_reference import weighted_test_thresholds
ROOT=Path(__file__).resolve().parent
def main():
 rng=np.random.default_rng(20261112);checks=0
 for _ in range(100):
  n=int(rng.integers(0,100));cal=rng.integers(0,21,(n,2))/20;y=rng.integers(0,2,n);w=rng.integers(1,20,n).astype(float)
  scores=rng.integers(0,21,(100,2))/20;query=rng.integers(1,20,100).astype(float)
  for alpha in [.05,.1,.2,.5]:
   q=weighted_test_thresholds(cal,y,w,query,alpha)
   for label in [0,1]:
    mask=y==label;B=np.array([w[mask][cal[mask,label]<s].sum() for s in scores[:,label]])
    membership=B<(1-alpha)*(w[mask].sum()+query)
    assert np.array_equal(scores[:,label]<=q[:,label],membership)
    checks+=len(query)
 cal=np.array([[.2,.2],[.5,.5]]);y=np.ones(2,dtype=int);w=np.array([2.,1.])
 q=weighted_test_thresholds(cal,y,w,np.array([1.]),.5)
 assert q[0,1]==.2 and not (.5<=q[0,1])
 for _ in range(100):
  p=rng.integers(0,101,200)/100;pool=rng.random(200)>.2
  cal=rng.integers(0,101,(80,2))/100;y=rng.integers(0,2,80);weights=rng.integers(1,20,80).astype(float)
  q1=weighted_test_thresholds(cal,y,weights,np.ones(200)*3,.1)[:,1]
  q2=weighted_test_thresholds(cal,y,weights[::-1],np.ones(200)*7,.1)[:,1]
  selected=pool & (1-p<=q1) & (1-p<=q2)
  ids=np.flatnonzero(pool);ranked=ids[np.argsort(-p[ids],kind='stable')]
  chosen=np.zeros(200,dtype=bool);chosen[ranked[:int(selected.sum())]]=True
  assert np.array_equal(chosen,selected)
 report={'weighted_membership_checks':checks,'strict_boundary_check':True,'constant_query_rank_checks':100,'status':'PASS'}
 (ROOT/'ACTIVATION_IDENTITY_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
