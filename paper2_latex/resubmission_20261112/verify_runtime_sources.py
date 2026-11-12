"""Verify the returned execution sources and their three changed behaviors.

Run from the full repository. No model training or manuscript result regeneration
is performed. OSQP 1.0.4 and RDKit 2026.03.4 are the recorded execution versions.
"""
from pathlib import Path
import argparse,ast,hashlib,io,json,sys,unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--repo',type=Path,default=ROOT.parents[1])
 ap.add_argument('--dependency-path',type=Path)
 args=ap.parse_args()
 sys.path.insert(0,str(args.repo))
 if args.dependency_path:sys.path.insert(0,str(args.dependency_path))
 runtime=ROOT/'runtime_sources'
 provenance=json.loads((ROOT/'inputs/provenance.json').read_text())
 pre_provenance=json.loads((ROOT/'inputs/provenance.pre_scaffold_fix.json').read_text())
 collection=json.loads((runtime/'COLLECTION_REPORT.json').read_text())
 hashes={}
 for rel,h in collection['files'].items():
  assert digest((runtime/rel).read_bytes())==h,rel
 allowed={'codrug.py':{'_scaffold_string','_train'},'kmm_cp.py':{'_solve_qp'}}
 for name,changed in allowed.items():
  path='paper2_admet_benchmark/revision_2026/full_baselines/'+name
  raw=(runtime/'actual_sources'/name).read_bytes()
  expected=provenance['identity']['code'][path.replace('/','\\')]
  assert digest(raw)==expected
  assert (args.repo/path).read_bytes()==raw
  ref=ast.parse((runtime/'reference_sources'/name).read_text())
  final=ast.parse(raw.decode())
  a={getattr(n,'name',str(i)):ast.dump(n,include_attributes=False) for i,n in enumerate(ref.body)}
  b={getattr(n,'name',str(i)):ast.dump(n,include_attributes=False) for i,n in enumerate(final.body)}
  actual={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
  assert actual==changed,(name,actual)
  hashes[name]={'sha256':expected,'matches_formal_provenance':True,'active_repository_copy_identical':True,'changed_functions':sorted(actual)}
 pre=(runtime/'pre_scaffold_fix/codrug.py').read_bytes()
 assert digest(pre)==pre_provenance['identity']['code']['paper2_admet_benchmark\\revision_2026\\full_baselines\\codrug.py']
 import numpy as np
 import osqp,rdkit
 from paper2_admet_benchmark.revision_2026.full_baselines import codrug,kmm_cp
 from paper2_admet_benchmark.revision_2026.full_baselines.test_kmm_cp import KMMCPTests
 from paper2_admet_benchmark.revision_2026.full_baselines.test_codrug import WeightedSetTests,GraphTrainingTests
 class ChemistryOnlyTests(GraphTrainingTests):
  # These three selected methods use only RDKit; the inherited class-level
  # skip also requires PyTorch for unrelated graph-training tests.
  __unittest_skip__=False
  __unittest_skip_why__=''
 class RuntimeFixTests(unittest.TestCase):
  def test_qp_rescaling_preserves_known_optimum_and_objective_units(self):
   P=np.array([[4.,1.],[1.,2.]]);q=np.array([-1.,-1.])
   x,d=kmm_cp._solve_qp(P,q,np.eye(2),np.zeros(2),np.full(2,10.),np.zeros(2),kmm_cp._config(None))
   np.testing.assert_allclose(x,np.linalg.solve(P,-q),atol=3e-6)
   self.assertAlmostEqual(d['objective'],float(.5*x@P@x+q@x),places=7)
   self.assertAlmostEqual(d['objective_scale'],float(np.linalg.eigvalsh(P)[-1]),places=12)
   self.assertEqual(d['objective_ridge'],0.)
  def test_scaffold_retry_keeps_nonisomeric_scaffold(self):
   Chem,Murcko=codrug._rdkit();original=Murcko.MurckoScaffoldSmiles
   canonical=codrug._canonical('C[C@H]1CCCCO1');expected=original(mol=Chem.MolFromSmiles(canonical))
   calls=[]
   def once(*a,**k):
    calls.append(1)
    if len(calls)==1:raise RuntimeError('synthetic stereo precondition')
    return original(*a,**k)
   codrug._scaffold_string.cache_clear()
   with patch.object(Murcko,'MurckoScaffoldSmiles',side_effect=once),patch.object(Chem,'RemoveStereochemistry',wraps=Chem.RemoveStereochemistry) as strip:
    self.assertEqual(codrug._scaffold_string(canonical),expected)
    self.assertEqual(strip.call_count,1);self.assertEqual(len(calls),2)
  def test_cuda_resume_expression_moves_each_rng_state_to_cpu(self):
   tree=ast.parse((runtime/'actual_sources/codrug.py').read_text())
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_train')
   node=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='set_rng_state_all')
   received=[];calls=[]
   class State:
    def __init__(self,i):self.i=i
    def cpu(self):calls.append(self.i);return ('cpu',self.i)
   expression=ast.fix_missing_locations(ast.Expression(body=node))
   eval(compile(expression,'<executed CUDA RNG expression>','eval'),{'torch':SimpleNamespace(cuda=SimpleNamespace(set_rng_state_all=received.append)),'saved':{'cuda_rng':[State(0),State(1)]}})
   self.assertEqual(calls,[0,1]);self.assertEqual(received,[[('cpu',0),('cpu',1)]])
 suite=unittest.TestSuite()
 loader=unittest.defaultTestLoader
 for cls in [KMMCPTests,WeightedSetTests,RuntimeFixTests]:suite.addTests(loader.loadTestsFromTestCase(cls))
 for name in ['test_auxiliary_filter_does_not_relax_empty_pool','test_auxiliary_scaffold_error_is_excluded_and_counted','test_auxiliary_string_cache_preserves_pool_counts_and_margin_reuse']:suite.addTest(ChemistryOnlyTests(name))
 log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 (runtime/'verification_tests.txt').write_text(log.getvalue())
 report={'status':'PASS' if result.wasSuccessful() else 'FAIL','sources':hashes,
  'pre_scaffold_codrug':{'sha256':digest(pre),'matches_recorded_pre_fix_provenance':True,'origin':'Reconstructed from supplied reference source plus the observed CUDA RNG restoration change; converted to the recorded CRLF byte encoding.'},
  'reference_commit':collection['reference_commit'],
  'tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)},
  'validation_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'osqp':osqp.__version__,'rdkit':rdkit.__version__},
  'cuda_check_scope':'Executed the saved RNG transfer expression with CPU-token mocks; no GPU resume or training run.',
  'scientific_training_performed':False,'frozen_predictions_recomputed':False}
 (ROOT/'RUNTIME_SOURCE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(log.getvalue());print(json.dumps(report,indent=2))
 if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
