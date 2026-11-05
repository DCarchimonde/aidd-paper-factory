"""Mechanism, invariance, role-isolation and runnable CoDrug training tests."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper2_admet_benchmark.revision_2026.full_baselines import codrug


HAS_GRAPH_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("rdkit") is not None


class WeightedSetTests(unittest.TestCase):
    def test_test_atom_and_uniform_ratio(self):
        member,q = codrug.weighted_sets_from_log_ratios([.1,.8,.3,.7],[0,1,0,1],[.1,.9],[0]*4,[0]*2,.2)
        np.testing.assert_array_equal(member,[[True,False],[False,True]])
        np.testing.assert_allclose(q,.3)
        full,q = codrug.weighted_sets_from_log_ratios([.1,.8],[0,1],[.1,.9],[0,0],[1000,1000],.1)
        self.assertTrue(full.all());self.assertTrue(np.isinf(q).all())

    def test_scale_invariance_and_candidate_class(self):
        args=([.1,.9,.2,.8],[0,1,0,1],[.1,.4,.9])
        a,_=codrug.weighted_sets_from_log_ratios(*args,[0,1,2,3],[0,1,2],.4,True)
        b,_=codrug.weighted_sets_from_log_ratios(*args,[900,901,902,903],[900,901,902],.4,True)
        np.testing.assert_array_equal(a,b)
        absent,q=codrug.weighted_sets_from_log_ratios([.1,.2],[0,0],[.99],[0,0],[0],.5,True)
        self.assertTrue(absent[0,1]);self.assertTrue(np.isinf(q[0,1]))

    def test_kde_domains_and_no_outcome_parameter(self):
        cal=np.array([[0,0],[1,0],[0,1],[1,1]],float);target=cal+.5
        a,b,info=codrug.fit_kde_ratios(cal,target,{"kde_folds":2,"kde_bandwidths":[.2,1.0]},seed=3)
        self.assertEqual(a.shape,(4,));self.assertEqual(b.shape,(4,));self.assertTrue(np.isfinite(a).all())
        self.assertEqual([x["domain"] for x in info["kde"]],["calibration","target"])
        threaded_a,threaded_b,threaded_info=codrug.fit_kde_ratios(cal,target,{"kde_folds":2,"kde_bandwidths":[.2,1.0]},seed=3,jobs=2)
        np.testing.assert_array_equal(a,threaded_a);np.testing.assert_array_equal(b,threaded_b)
        self.assertEqual(info,threaded_info)

    def test_pinned_auxiliary_rejects_bad_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"aux.csv";p.write_text("smiles,ignored_label\nCCO,1\n")
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                codrug.acquire_auxiliary_smiles(d,{"auxiliary_csv":str(p),"auxiliary_sha256":"0"*64})


@unittest.skipUnless(HAS_GRAPH_DEPS,"PyTorch and RDKit required")
class GraphTrainingTests(unittest.TestCase):
    def setUp(self):
        codrug._canonical.cache_clear()
        codrug._scaffold_string.cache_clear()
        codrug._filtered_auxiliary_pool.cache_clear()
        self.cfg={"graph_size":16,"hidden_sizes":[12,8],"gnn_layers":2,"readout_steps":2,
                  "batch_size":3,"max_epochs":2,"patience":2,"kde_folds":2,
                  "kde_bandwidths":[.2,1.0],"aux_max_molecules":10}

    def test_graph_permutation_batch_and_isolated_atom(self):
        torch=codrug._torch();torch.manual_seed(1)
        net=codrug.build_attentivefp(self.cfg);net.eval()
        graphs=[codrug.molecular_graph("CCO"),codrug.molecular_graph("c1ccccc1"),codrug.molecular_graph("[Na+]")]
        with torch.no_grad():
            joint=net(codrug._batch(graphs,"cpu"))[0]
            single=torch.cat([net(codrug._batch([g],"cpu"))[0] for g in graphs])
            np.testing.assert_allclose(joint.numpy(),single.numpy(),atol=1e-6)
            g=graphs[0];perm=np.array([2,0,1]);inverse=np.argsort(perm)
            changed=codrug.MolecularGraph(g.atoms[perm],g.bonds,inverse[g.source],inverse[g.destination])
            np.testing.assert_allclose(net(codrug._batch([changed],"cpu"))[0].numpy(),single[:1].numpy(),atol=1e-6)

    def test_energy_formula_and_graph_gradients(self):
        torch=codrug._torch()
        a=torch.tensor([[0.,0.]],requires_grad=True);b=torch.tensor([[100.,100.]],requires_grad=True)
        loss=codrug.energy_penalty(a,b)
        expected=(5-np.log(2))**2+(65+np.log(2))**2
        self.assertAlmostEqual(float(loss.detach()),expected,places=2)
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.isfinite(b.grad).all())
        net=codrug.build_attentivefp(self.cfg)
        logits,_,_=net(codrug._batch([codrug.molecular_graph("CCO"),codrug.molecular_graph("c1ccccc1")],"cpu"))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])).backward()
        self.assertTrue(torch.isfinite(net.atom_projection.weight.grad).all())
        self.assertGreater(float(net.atom_projection.weight.grad.abs().sum()),0)

    def test_auxiliary_filter_does_not_relax_empty_pool(self):
        kept,info=codrug.filter_auxiliary_smiles(["CCO","c1ccccc1","C1CCCCC1"],["CCN"],self.cfg)
        self.assertEqual(len(kept),2);self.assertEqual(info["source_scaffold_or_similarity"],1)
        with self.assertRaisesRegex(ValueError,"No auxiliary"):
            codrug.filter_auxiliary_smiles(["CCO"],["CCN"],self.cfg)

    def test_auxiliary_scaffold_error_is_excluded_and_counted(self):
        Chem,Murcko=codrug._rdkit()
        original=Murcko.MurckoScaffoldSmiles
        def scaffold(*args,**kwargs):
            if Chem.MolToSmiles(kwargs["mol"])=="C1CCCCC1":
                raise RuntimeError("Pre-condition Violation: bad bond stereo")
            return original(*args,**kwargs)
        with patch.object(Murcko,"MurckoScaffoldSmiles",side_effect=scaffold):
            kept,info=codrug.filter_auxiliary_smiles(["C1CCCCC1","c1ccccc1"],["CCN"],self.cfg)
            self.assertEqual(kept,["c1ccccc1"])
            self.assertEqual(info["scaffold_failure"],1)
            with self.assertRaises(RuntimeError):
                codrug.filter_auxiliary_smiles(["c1ccccc1"],["C1CCCCC1"],self.cfg)

    def test_auxiliary_string_cache_preserves_pool_counts_and_margin_reuse(self):
        auxiliary=["CCN","NCC","CCO","c1ccccc1","C1CCCCC1","c1ccncc1","c1ccccc1"]
        source=["CCN"]
        config={**self.cfg,"aux_max_molecules":2}
        # Explicitly bypass every cache for the reference evaluation.
        with patch.object(codrug,"_canonical",codrug._canonical.__wrapped__), \
             patch.object(codrug,"_scaffold_string",codrug._scaffold_string.__wrapped__), \
             patch.object(codrug,"_filtered_auxiliary_pool",codrug._filtered_auxiliary_pool.__wrapped__):
            reference=codrug.filter_auxiliary_smiles(auxiliary,source,config,seed=3)
        Chem,_=codrug._rdkit()
        with patch.object(Chem,"MolFromSmiles",wraps=Chem.MolFromSmiles) as parser:
            cold=codrug.filter_auxiliary_smiles(auxiliary,source,config,seed=3)
            parsed=parser.call_count
            warm=codrug.filter_auxiliary_smiles(auxiliary,source,config,seed=3)
            swapped=codrug.filter_auxiliary_smiles(auxiliary,source,
                {**config,"margin_in":-35.,"margin_out":-5.},seed=3)
            self.assertGreater(parsed,0)
            self.assertEqual(parser.call_count,parsed)
        self.assertEqual(reference,cold);self.assertEqual(cold,warm);self.assertEqual(cold,swapped)
        self.assertEqual(codrug._filtered_auxiliary_pool.cache_info().hits,2)
        self.assertEqual(codrug._canonical.cache_info().maxsize,300000)
        self.assertEqual(codrug._scaffold_string.cache_info().maxsize,300000)
        self.assertEqual(cold[1]["duplicates"],2)
        self.assertEqual(cold[1]["source_identity"],1)
        cold[0].clear();cold[1]["retained"]=-1
        self.assertEqual(codrug.filter_auxiliary_smiles(auxiliary,source,config,seed=3),reference)

    def test_complete_training_checkpoint_and_target_label_absence(self):
        fit=["CCO","CCN","CCC","CCCC","CCCO","CCCN"]
        router=["CCCl","CCBr"]
        cal=["CCF","CC(C)O","CC(C)N","CC(C)C"]
        target=["COC","COCC","CNC","CNCC"]
        aux=["c1ccccc1","C1CCCCC1","c1ccncc1","C1CCOC1"]
        with tempfile.TemporaryDirectory() as d:
            result=codrug.fit_predict(fit,[0,1,0,1,0,1],router,[0,1],cal,[0,1,0,1],target,[.1,.2],17,
                                     config=self.cfg,model_dir=d,auxiliary_smiles=aux)
            self.assertEqual(result["probability_1"].shape,(4,))
            self.assertEqual(result["target_embedding"].shape,(4,8))
            self.assertTrue(np.isfinite(result["probability_1"]).all())
            self.assertFalse(result["diagnostics"]["target_labels_used"])
            self.assertEqual(result["diagnostics"]["training"]["epochs_completed"],2)
            self.assertTrue((Path(d)/"training_checkpoint.pt").is_file())
            repeat=codrug.fit_predict(fit,[0,1,0,1,0,1],router,[0,1],cal,[0,1,0,1],target,[.1,.2],17,
                                     config=self.cfg,model_dir=d,auxiliary_smiles=aux)
            np.testing.assert_array_equal(result["probability_1"],repeat["probability_1"])
            self.assertTrue(repeat["diagnostics"]["training"]["checkpoint_reused"])
            with self.assertRaisesRegex(ValueError,"different data"):
                codrug.fit_predict(fit,[1,1,0,1,0,1],router,[0,1],cal,[0,1,0,1],target,[.1],17,
                                   config=self.cfg,model_dir=d,auxiliary_smiles=aux)

    def test_vanilla_kmm_protocol_no_auxiliary(self):
        cfg={**self.cfg,"energy_weight":0.,"hidden_sizes":[7],"class_weight":"inverse_frequency",
             "scheduler_gamma":.8,"early_stop_on":"train_loss","compute_conformal":False}
        result=codrug.fit_predict(["CCO","CCN","CCC","CCCC"],[0,1,0,1],["CCCl","CCBr"],[0,1],
                                  ["CCF","CC(C)O"],[0,1],["COC","CNC"],[.1],19,config=cfg)
        self.assertEqual(result["sets_by_alpha"],{})
        self.assertEqual(result["target_embedding"].shape,(2,7))
        self.assertFalse(result["diagnostics"]["auxiliary"]["used"])
        self.assertEqual(result["diagnostics"]["training"]["selection_metric"],"train_loss")


if __name__=="__main__": unittest.main()
