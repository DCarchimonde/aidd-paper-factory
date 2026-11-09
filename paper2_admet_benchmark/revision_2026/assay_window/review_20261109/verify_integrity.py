"""Verify returned assay-window artifacts without fitting models or changing source."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--return-root', type=Path, required=True, help='Directory containing extracted prepare/ and screen/ outputs')
parser.add_argument('--repo', type=Path, required=True, help='Local aidd-paper-factory checkout')
parser.add_argument('--raw-archive', type=Path, required=True, help='Original NCATS protocol ZIP')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--reference-commit', default='e223c20cbad7f6238b22feab3cddd6e48339de62')
args = parser.parse_args()
BASE = args.return_root.resolve()
REPO = args.repo.resolve()
PKG = REPO/'paper2_admet_benchmark/revision_2026/assay_window'
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
REFERENCE = subprocess.check_output(['git', 'rev-parse', args.reference_commit+'^{commit}'], cwd=REPO).decode().strip()

def reference_bytes(name):
 return subprocess.check_output(['git','show',f'{REFERENCE}:paper2_admet_benchmark/revision_2026/assay_window/{name}'],cwd=REPO)

def read(p): return json.loads(p.read_text())
def gzread(p): return json.loads(gzip.decompress(p.read_bytes()))
def sha(b): return hashlib.sha256(b).hexdigest()
prep,screen=BASE/'prepare',BASE/'screen'
records=gzread(screen/'paired_records.json.gz')
derived=gzread(screen/'derived_records.json.gz')
config=read(screen/'configuration.json')
report={'real_model_fits_performed':False,'reference_commit':REFERENCE}
report['source_code_hashes']={}
for name,expected in config['code'].items():
 b=reference_bytes(name)
 local_bytes=(PKG/name).read_bytes()
 assert sha(local_bytes)==sha(b), f'Local source differs from reference commit: {name}'
 variants={'git_bytes':sha(b),'LF':sha(b.replace(b'\r\n',b'\n')),'CRLF':sha(b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))}
 matching=[k for k,v in variants.items() if v==expected]
 assert matching,(name,expected,variants)
 report['source_code_hashes'][name]={'matches':matching,'returned_sha256':expected,'local_file_sha256':sha(local_bytes),'reference_file_sha256':sha(b),'local_matches_reference':True}
# Only import the verified local source, never another installed package.
sys.path.insert(0,str(REPO))
from paper2_admet_benchmark.revision_2026.assay_window import execute, data, model, targets
for module in (execute, data, model, targets):
 imported_path=Path(module.__file__).resolve()
 assert imported_path==PKG/(module.__name__.rsplit('.',1)[-1]+'.py'), imported_path
 assert sha(imported_path.read_bytes())==sha(reference_bytes(imported_path.name))
report['imported_module_sources']='execute, data, model, targets paths and byte hashes match the reference checkout'
report['prepare_screen_shared_artifacts']={}
for name in ['paired_records.json.gz','derived_records.json.gz','data_profile.json','source_diagnostics.json']:
 a,b=(prep/name).read_bytes(),(screen/name).read_bytes()
 report['prepare_screen_shared_artifacts'][name]={'byte_identical':a==b,'sha256':sha(a)}
 assert a==b
for mode in ['prepare','screen']:
 d=BASE/mode;c=read(d/'configuration.json');comp=read(d/'experiment_complete.json')
 assert comp['identity']['configuration_sha256']==execute.digest(c)
 assert comp['identity']['paired_records_sha256']==execute.digest(records)
report['configuration_and_record_identity']='matched for both modes'
rederived,masks=execute.derive_records(records)
assert rederived==derived
assert execute.describe_records(derived)==read(screen/'data_profile.json')
report['rederived_target_and_profile']={'all_records_match':True,'rows':len(records),'all_uninformative_masks':int(masks.all(axis=(1,2)).sum())}
# Verify supplied identities from supplied standardized SMILES; no local RDKit runtime.
for r in records:
 assert r['group_id']==sha(f"{data.STRUCTURE_POLICY}|{r['canonical_smiles']}".encode())
 assert r['compound_id']==r['group_id']
 assert r['structure_id']==sha(f"canonical_full_isomeric|{r['canonical_full_smiles']}".encode())
report['supplied_structure_id_hashes']='all consistent; RDKit transformations not independently rerun locally'
source=read(screen/'source_diagnostics.json')
protocol=config['protocols'][0];s=source[protocol]
archive=args.raw_archive.resolve()
assert sha(archive.read_bytes())==s['raw_archive_sha256']==s['download']['sha256']==data._spec(protocol)['sha256']
# Replay every raw curve and pairing decision using returned structure metadata.
structure_keys=['smiles','standardized_smiles','canonical_smiles','canonical_full_smiles','structure_id','compound_id','group_id','scaffold_id','structure_policy','rdkit_version']
structures={r['raw_smiles']:{k:r[k] for k in structure_keys} for r in records}
def supplied_structure(raw):
 if raw not in structures: raise ValueError('not_retained_by_returned_RDKit_preparation')
 return structures[raw]
reparsed,diagnostics=data.parse_protocol_zip(archive,protocol,standardizer=supplied_structure)
assert reparsed==records
for key,v in diagnostics.items(): assert v==s[key],key
report['raw_source_replay']={'archive_sha256':s['raw_archive_sha256'],'all_paired_records_match_raw_curves':True,'all_parser_diagnostics_match':True,'chemistry_validation':'Uses supplied standardized structure metadata, not independent RDKit re-standardization.'}
old=json.loads(reference_bytes('evidence/real_protocol_parser_counts.json'))
invalid=[x for x in s['exclusions'] if 'invalid_structure:ValueError' in x['reasons']]
assert len(invalid)==old['retained_pairs']-len(records)==545
report['difference_from_pre_RDKit_read']={'before_pairs':old['retained_pairs'],'after_pairs':len(records),'removed_pairs':len(invalid),'reason':'invalid_structure:ValueError','affected_samples':len({x['sample_id'] for x in invalid}),'before_raw_SMILES_groups':old['retained_compound_groups'],'after_retained_raw_SMILES_groups':len({r['raw_smiles'] for r in records}),'after_parent_structure_groups':len({r['group_id'] for r in records}),'before_samples':old['retained_samples'],'after_samples':s['retained_samples']}
report['splits']={}
for split in ['random','scaffold']:
 train,test=execute.split_indices(derived,split);manifest=read(screen/f'split_{split}.json')
 train_groups={records[int(i)]['group_id'] for i in train};test_groups={records[int(i)]['group_id'] for i in test}
 assert train_groups==set(manifest['train_group_ids']) and test_groups==set(manifest['test_group_ids'])
 assert len(train)==manifest['train_rows'] and len(test)==manifest['test_rows']
 assert not train_groups & test_groups
 train_scaffolds={records[int(i)]['scaffold_id'] for i in train};test_scaffolds={records[int(i)]['scaffold_id'] for i in test}
 train_samples={records[int(i)]['sample_id'] for i in train};test_samples={records[int(i)]['sample_id'] for i in test}
 train_curves={records[int(i)][key] for i in train for key in ['target_curve_id','viability_curve_id']};test_curves={records[int(i)][key] for i in test for key in ['target_curve_id','viability_curve_id']}
 assert not train_samples & test_samples and not train_curves & test_curves
 if split=='scaffold': assert not train_scaffolds & test_scaffolds
 entry={'replayed_split_exact_match':True,'train_groups':len(train_groups),'test_groups':len(test_groups),'test_group_fraction':len(test_groups)/(len(train_groups)+len(test_groups)),'train_rows':len(train),'test_rows':len(test),'test_row_fraction':len(test)/len(records),'parent_overlap':0,'sample_overlap':0,'curve_overlap':0,'scaffold_overlap':len(train_scaffolds & test_scaffolds),'unique_train_scaffolds':len(train_scaffolds),'unique_test_scaffolds':len(test_scaffolds)}
 if split=='scaffold':
  buckets=defaultdict(set)
  for r in records: buckets[r['scaffold_id']].add(r['group_id'])
  ordering=sorted(buckets,key=lambda x:execute.digest(['window_split',9021,'scaffold',x]))
  wanted=math.ceil(.25*len(set(r['group_id'] for r in records)));selected=set();chosen=[]
  for key in ordering[:-1]:
   if len(selected)>=wanted:break
   before=len(selected);selected.update(buckets[key]);chosen.append({'scaffold':key,'groups':len(buckets[key]),'cumulative_before':before,'cumulative_after':len(selected)})
  assert selected==test_groups
  entry['fraction_overshoot']={'target_groups':wanted,'selected_scaffold_buckets':len(chosen),'last_bucket':chosen[-1],'largest_buckets':sorted([{'scaffold':k,'groups':len(v),'in_test':bool(v & test_groups)} for k,v in buckets.items()],key=lambda x:-x['groups'])[:10],'explanation':'Hash-order whole scaffold buckets are appended until at least 25% of groups is reached; the crossing bucket is kept intact.'}
 report['splits'][split]=entry
# Each run must use exactly the corresponding test rows and the shared identity.
key=lambda r:tuple(r[k] for k in ['group_id','sample_id','protocol','replicate','target_curve_id','viability_curve_id'])
all_jobs=read(screen/'required_jobs.json');complete=read(screen/'experiment_complete.json')
assert all_jobs['identity']==complete['identity']
for job in all_jobs['jobs']:
 result=read(screen/'runs'/f"{job['job_id']}.json")
 assert result['identity']==complete['identity'] and result['job']==job
 _,test=execute.split_indices(derived,job['split'])
 expected=[key(records[int(i)]) for i in test];got=[key(r) for r in result['predictions']]
 assert expected==got
 assert result['metrics']['convergence']['converged']
report['run_integrity']={'runs':len(all_jobs['jobs']),'all_converged':True,'all_test_record_keys_exactly_matched':True}
report['material_findings']=['The scaffold holdout is 43.57% of parent groups, not a 25% holdout. The deterministic whole-bucket rule is followed exactly; no scaffold leakage. Report actual split sizes rather than 75/25.','545 fewer pairs than the initial parser-only count are all marked invalid_structure:ValueError. This audit replays curve content and pairing exactly, but cannot independently validate RDKit transformations without that runtime.','No source revision, data-pairing, derived-target, or split mismatch found.']
out=OUTPUT/'integrity_review.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k in ['difference_from_pre_RDKit_read','splits','run_integrity','material_findings']},ensure_ascii=False,indent=2))
