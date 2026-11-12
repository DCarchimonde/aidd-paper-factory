"""Complete paper-based CoDrug reimplementation; no target outcomes accepted.

Implements an AttentiveFP graph classifier, the published energy-margin loss,
separate cross-validated calibration/target KDEs, and test-mass weighted sets.
The graph operations independently express the architecture documented in
DGL-LifeSci (Apache-2.0); this is not the unavailable CoDrug authors' source.
See CODRUG_IMPLEMENTATION.md for equations, reported defaults and differences.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import urllib.request

import numpy as np
from scipy.special import logsumexp, softmax
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


PAPER_URL = "https://proceedings.neurips.cc/paper_files/paper/2023/file/7691484a7a35d5e2742279c1d926b778-Paper-Conference.pdf"
ZINC_URL = "https://raw.githubusercontent.com/the-matter-lab/chemical_vae/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
ZINC_SHA256 = "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"
METHOD_NAMES = ("codrug_global", "codrug_mondrian")
DEFAULT_CONFIG = {
    "graph_size": 512, "hidden_sizes": [256, 8], "gnn_layers": 2,
    "readout_steps": 2, "dropout": 0.0, "learning_rate": 0.001,
    "batch_size": 64, "max_epochs": 100, "patience": 15,
    "energy_weight": 0.01, "margin_in": -5.0, "margin_out": -35.0,
    "temperature": 1.0, "weight_decay": 0.0, "density_representation": "logits",
    "class_weight": "none", "scheduler_gamma": None, "scheduler_step_size": 1,
    "early_stop_on": "router_cross_entropy", "compute_conformal": True,
    "kde_folds": 10, "kde_bandwidths": np.linspace(10 ** -1.3, 10.0, 25).tolist(),
    "aux_filter": "scaffold", "aux_max_molecules": None,
    "aux_fingerprint_radius": 2, "aux_fingerprint_bits": 2048,
    "graph_cache_size": 20000, "verbose": False,
}


def default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _configuration(config=None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg.get("use_energy") is False:
        cfg["energy_weight"] = 0.0
    for key in ("graph_size", "gnn_layers", "readout_steps", "batch_size", "max_epochs", "patience", "kde_folds"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["hidden_sizes"] or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in cfg["hidden_sizes"]):
        raise ValueError("hidden_sizes requires positive integers")
    for key in ("learning_rate", "temperature"):
        if not math.isfinite(float(cfg[key])) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("energy_weight", "weight_decay"):
        if not math.isfinite(float(cfg[key])) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= cfg["dropout"] < 1 or not all(math.isfinite(float(cfg[k])) for k in ("margin_in", "margin_out")):
        raise ValueError("Invalid dropout or energy margins")
    bandwidths = np.asarray(cfg["kde_bandwidths"], float)
    if not bandwidths.size or not np.isfinite(bandwidths).all() or np.any(bandwidths <= 0):
        raise ValueError("KDE bandwidths must be finite and positive")
    if cfg["density_representation"] not in {"logits", "embedding"}:
        raise ValueError("density_representation must be logits or embedding")
    if cfg["class_weight"] not in {"none", "inverse_frequency"} or cfg["early_stop_on"] not in {"router_cross_entropy", "train_loss"}:
        raise ValueError("Invalid class weighting or early-stopping metric")
    if cfg["scheduler_gamma"] is not None and (not math.isfinite(float(cfg["scheduler_gamma"])) or cfg["scheduler_gamma"]<=0):
        raise ValueError("scheduler_gamma must be positive or null")
    if not isinstance(cfg["scheduler_step_size"],int) or cfg["scheduler_step_size"]<1:
        raise ValueError("scheduler_step_size must be a positive integer")
    if not isinstance(cfg["compute_conformal"],bool):
        raise ValueError("compute_conformal must be boolean")
    if cfg["aux_filter"] not in {"scaffold", "fingerprint"}:
        raise ValueError("aux_filter must be scaffold or fingerprint")
    limit = cfg["aux_max_molecules"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("aux_max_molecules must be null or a positive integer")
    return cfg


def acquire_auxiliary_smiles(cache_dir, config=None) -> tuple[list[str], dict]:
    """Acquire the standard ZINC250k snapshot and verify its pinned SHA256.

    Reads only the SMILES column. No ZINC calculated-property columns are used.
    A custom auxiliary_csv is allowed only with an explicit auxiliary_sha256.
    """
    cfg = config or {}
    folder = Path(cache_dir); folder.mkdir(parents=True, exist_ok=True)
    custom = cfg.get("auxiliary_csv")
    path = Path(custom) if custom else folder / "zinc250k.csv"
    expected = cfg.get("auxiliary_sha256") if custom else ZINC_SHA256
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Custom auxiliary data require auxiliary_sha256")
    if not path.exists():
        if custom:
            raise FileNotFoundError(path)
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(ZINC_URL, timeout=90) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            observed = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError("Downloaded auxiliary data differ from the pinned ZINC snapshot")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError("Cached auxiliary data SHA256 mismatch")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError("Auxiliary CSV requires a smiles column")
        smiles = [r["smiles"].strip() for r in reader if r["smiles"].strip()]
    metadata = {"url": ZINC_URL if not custom else "user_supplied_pinned_file",
                "sha256": observed, "rows": len(smiles), "path": str(path),
                "columns_used": ["smiles"]}
    (folder / "auxiliary_source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return smiles, metadata


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:
        raise ImportError("CoDrug requires RDKit; install revision_2026 full requirements") from error
    return Chem, MurckoScaffold


@lru_cache(maxsize=300000)
def _canonical(smiles: str) -> str:
    Chem, _ = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecular SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=300000)
def _scaffold_string(canonical: str) -> str:
    """Cache immutable text only; no RDKit Mol is retained by this cache.

    Some otherwise-valid conjugated stereo structures trigger an RDKit
    MolToSmiles precondition failure inside the Murcko transform. Murcko
    scaffolds are not isomeric, so a stereo-stripped retry is equivalent for
    this OOD rule and avoids dropping the molecule.
    """
    Chem, Murcko = _rdkit()
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"Invalid molecular SMILES: {canonical!r}")
    try:
        return Murcko.MurckoScaffoldSmiles(mol=mol)
    except RuntimeError:
        Chem.RemoveStereochemistry(mol)
        return Murcko.MurckoScaffoldSmiles(mol=mol)


def filter_auxiliary_smiles(auxiliary_smiles, source_smiles, config=None, seed=0):
    """Return a private copy of the exact source-dependent auxiliary pool.

    Chemistry string caches are bounded at 300,000 entries. The last eight
    complete filtering results are immutable tuples/JSON, keyed only by data,
    filtering settings and seed. Energy-margin variants share these results.
    """
    cfg = _configuration(config)
    auxiliary = tuple(str(s) for s in auxiliary_smiles)
    source = tuple(sorted({_canonical(str(s)) for s in source_smiles}))
    kept, counts = _filtered_auxiliary_pool(
        auxiliary, source, cfg["aux_filter"], cfg["aux_max_molecules"],
        cfg["aux_fingerprint_radius"], cfg["aux_fingerprint_bits"], int(seed))
    return list(kept), json.loads(counts)


@lru_cache(maxsize=8)
def _filtered_auxiliary_pool(auxiliary_smiles, source, aux_filter, limit, radius, bits, seed):
    """Label-free scaffold exclusion, or the paper's strict fingerprint rule.

    Source includes fit/router/calibration structures, never their outcomes.
    Acyclic molecules share the empty Murcko scaffold and are treated alike.
    """
    Chem, _ = _rdkit()
    source = set(source)
    scaffolds = {_scaffold_string(s) for s in source}
    fp_source = None
    threshold = None
    if aux_filter == "fingerprint":
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit import DataStructs
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
        fp_source = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in source]
        if len(fp_source) < 2:
            raise ValueError("Fingerprint OOD filtering requires at least two source molecules")
        threshold = min(min(DataStructs.BulkTanimotoSimilarity(fp_source[i], fp_source[i + 1:])) for i in range(len(fp_source) - 1))
    counts = {"input": 0, "invalid": 0, "duplicates": 0, "source_identity": 0,
              "source_scaffold_or_similarity": 0, "scaffold_failure": 0}
    kept, seen = [], set()
    for text in auxiliary_smiles:
        counts["input"] += 1
        try:
            canonical = _canonical(text)
        except ValueError:
            counts["invalid"] += 1
            continue
        if canonical in seen:
            counts["duplicates"] += 1
            continue
        seen.add(canonical)
        if canonical in source:
            counts["source_identity"] += 1
            continue
        if aux_filter == "scaffold":
            try:
                excluded = _scaffold_string(canonical) in scaffolds
            except RuntimeError:
                # Some otherwise parseable ZINC stereo structures fail RDKit's
                # scaffold transform. Exclude them; never bypass the OOD rule.
                counts["scaffold_failure"] += 1
                continue
        else:
            excluded = max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles(canonical)), fp_source)) >= threshold
        if excluded:
            counts["source_scaffold_or_similarity"] += 1
        else:
            kept.append(canonical)
    kept.sort()
    counts["eligible_before_limit"] = len(kept)
    if limit is not None and len(kept) > limit:
        indices = np.random.default_rng(seed).choice(len(kept), limit, replace=False)
        kept = [kept[i] for i in sorted(indices)]
    counts.update({"retained": len(kept), "filter": aux_filter, "fingerprint_threshold": threshold,
                   "pool_sha256": _hash(kept), "subsample_limit": limit})
    if not kept:
        raise ValueError("No auxiliary molecules satisfy the stated OOD rule; do not silently relax it")
    return tuple(kept), json.dumps(counts, sort_keys=True)


@dataclass(frozen=True)
class MolecularGraph:
    atoms: np.ndarray
    bonds: np.ndarray
    source: np.ndarray
    destination: np.ndarray


def molecular_graph(smiles: str) -> MolecularGraph:
    """39 atom / 10 bond AttentiveFP features; two directed edges per bond."""
    Chem, _ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot featurize an invalid or empty molecule")
    def onehot(value, options, unknown=False):
        return [float(value == x) for x in options] + ([float(value not in options)] if unknown else [])
    hybrid = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
              Chem.rdchem.HybridizationType.SP3D2]
    atoms = []
    for atom in mol.GetAtoms():
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        atoms.append(onehot(atom.GetSymbol(), ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At'], True)
                     + onehot(atom.GetDegree(), range(6)) + [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
                     + onehot(atom.GetHybridization(), hybrid, True) + [float(atom.GetIsAromatic())]
                     + onehot(atom.GetTotalNumHs(), range(5)) + [float(atom.HasProp("_ChiralityPossible"))]
                     + onehot(cip, ["R", "S"]))
    bonds, source, destination = [], [], []
    types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    stereos = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
    for bond in mol.GetBonds():
        feat = onehot(bond.GetBondType(), types) + [float(bond.GetIsConjugated()), float(bond.IsInRing())] + onehot(bond.GetStereo(), stereos)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.extend([feat, feat]); source.extend([a,b]); destination.extend([b,a])
    return MolecularGraph(np.asarray(atoms, np.float32).reshape(-1,39), np.asarray(bonds,np.float32).reshape(-1,10),
                          np.asarray(source,np.int64), np.asarray(destination,np.int64))


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("CoDrug requires PyTorch; install revision_2026 full requirements") from error
    return torch


def _batch(graphs, device):
    torch = _torch()
    offsets = np.cumsum([0] + [len(g.atoms) for g in graphs[:-1]])
    return (torch.as_tensor(np.concatenate([g.atoms for g in graphs]), device=device),
            torch.as_tensor(np.concatenate([g.bonds for g in graphs]), device=device),
            torch.as_tensor(np.concatenate([g.source + o for g,o in zip(graphs,offsets)]),device=device),
            torch.as_tensor(np.concatenate([g.destination + o for g,o in zip(graphs,offsets)]),device=device),
            torch.as_tensor(np.concatenate([np.full(len(g.atoms),i,np.int64) for i,g in enumerate(graphs)]),device=device), len(graphs))


def build_attentivefp(config=None):
    """Pure PyTorch AttentiveFP message passing/readout, without DGL binaries."""
    torch = _torch(); nn = torch.nn; F = torch.nn.functional
    cfg = _configuration(config); size = cfg["graph_size"]; dropout = cfg["dropout"]
    def segmented_softmax(logits, groups, n):
        maximum = logits.new_full((n,1), -torch.inf)
        maximum.scatter_reduce_(0, groups[:,None], logits.detach(), reduce="amax", include_self=True)
        exp = torch.exp(logits - maximum[groups])
        total = logits.new_zeros((n,1)); total.index_add_(0,groups,exp)
        return exp / total[groups].clamp_min(torch.finfo(logits.dtype).tiny)
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_projection = nn.Linear(39,size)
            self.bond_projection = nn.Linear(49,size)
            self.context_attention = nn.Linear(2*size,1)
            self.context_message = nn.Linear(size,size)
            self.context_gru = nn.GRUCell(size,size)
            self.message_attention = nn.ModuleList([nn.Linear(2*size,1) for _ in range(cfg["gnn_layers"]-1)])
            self.message_projection = nn.ModuleList([nn.Linear(size,size) for _ in range(cfg["gnn_layers"]-1)])
            self.message_gru = nn.ModuleList([nn.GRUCell(size,size) for _ in range(cfg["gnn_layers"]-1)])
            self.pool_attention = nn.ModuleList([nn.Linear(2*size,1) for _ in range(cfg["readout_steps"])])
            self.pool_projection = nn.ModuleList([nn.Linear(size,size) for _ in range(cfg["readout_steps"])])
            self.pool_gru = nn.ModuleList([nn.GRUCell(size,size) for _ in range(cfg["readout_steps"])])
            dims = [size] + list(cfg["hidden_sizes"])
            self.hidden = nn.ModuleList([nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])])
            self.output = nn.Linear(dims[-1],2)
            self.dropout = nn.Dropout(dropout)
        def forward(self,batch):
            atoms,bonds,source,dest,graph_index,n_graph = batch
            h = F.leaky_relu(self.atom_projection(atoms), negative_slope=.01)
            edges = F.leaky_relu(self.bond_projection(torch.cat([atoms[source],bonds],dim=1)),negative_slope=.01)
            logits = F.leaky_relu(self.context_attention(self.dropout(torch.cat([h[dest],edges],dim=1))),negative_slope=.01)
            weights = segmented_softmax(logits,dest,len(h))
            context = h.new_zeros(h.shape); context.index_add_(0,dest,weights*self.context_message(self.dropout(edges)))
            h = F.relu(self.context_gru(F.elu(context),h))
            for att,project,gru in zip(self.message_attention,self.message_projection,self.message_gru):
                logits = F.leaky_relu(att(self.dropout(torch.cat([h[dest],h[source]],dim=1))),negative_slope=.01)
                weights = segmented_softmax(logits,dest,len(h))
                context = h.new_zeros(h.shape); context.index_add_(0,dest,weights*project(self.dropout(h))[source])
                h = F.relu(gru(F.elu(context),h))
            graph = h.new_zeros((n_graph,size)); graph.index_add_(0,graph_index,h)
            for att,project,gru in zip(self.pool_attention,self.pool_projection,self.pool_gru):
                logits = F.leaky_relu(att(torch.cat([F.relu(graph)[graph_index],h],dim=1)),negative_slope=.01)
                weights = segmented_softmax(logits,graph_index,n_graph)
                context = graph.new_zeros(graph.shape); context.index_add_(0,graph_index,weights*project(self.dropout(h)))
                graph = gru(F.elu(context),graph)
            embedding = graph
            for layer in self.hidden:
                embedding = F.relu(layer(embedding))
            return self.output(embedding), embedding, graph
    return Network()


def energy_penalty(logits_in, logits_out, margin_in=-5.0, margin_out=-35.0, temperature=1.0):
    """CoDrug eqs. 15/17, differentiable; no sampling-based energy objective."""
    torch = _torch()
    energy_in = -temperature * torch.logsumexp(logits_in / temperature,dim=1)
    energy_out = -temperature * torch.logsumexp(logits_out / temperature,dim=1)
    return torch.relu(energy_in-margin_in).square().mean() + torch.relu(margin_out-energy_out).square().mean()


def _labels(y,n):
    arr = np.asarray(y)
    if arr.shape != (n,) or not np.isin(arr,[0,1]).all():
        raise ValueError("Role labels must be aligned binary 0/1 values")
    return arr.astype(np.int64)


def _predict_network(model, smiles, graph_getter, device, batch_size):
    torch = _torch(); values = [[],[],[]]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(smiles),batch_size):
            output = model(_batch([graph_getter(s) for s in smiles[start:start+batch_size]],device))
            for accumulator,value in zip(values,output):
                accumulator.append(value.detach().cpu().numpy())
    return tuple(np.concatenate(v) for v in values)


def _train(train, train_y, tuning, tuning_y, auxiliary, cfg, seed,device,jobs,model_dir,signature):
    torch = _torch()
    torch.set_num_threads(max(1,int(jobs)))
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose cpu explicitly")
    generator = np.random.default_rng(seed)
    graph_getter = lru_cache(maxsize=cfg["graph_cache_size"])(molecular_graph)
    model = build_attentivefp(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer,step_size=cfg["scheduler_step_size"],gamma=cfg["scheduler_gamma"])
                 if cfg["scheduler_gamma"] is not None else None)
    class_weights = None
    if cfg["class_weight"] == "inverse_frequency":
        counts = np.bincount(train_y,minlength=2)
        if np.any(counts == 0): raise ValueError("Inverse-frequency loss requires both fit classes")
        class_weights = torch.as_tensor(len(train_y)/(2*counts),dtype=torch.float32,device=device)
    folder = Path(model_dir) if model_dir else None
    checkpoint = folder / "training_checkpoint.pt" if folder else None
    start, best_loss, best_epoch, stale, history, best_state = 0, math.inf, -1, 0, [], None
    completed = False
    if folder:
        folder.mkdir(parents=True,exist_ok=True)
        if checkpoint.exists():
            saved = torch.load(checkpoint,map_location=device,weights_only=False)
            if saved["training_signature"] != signature:
                raise ValueError("Checkpoint belongs to different data/configuration; use a new model directory")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None: scheduler.load_state_dict(saved["scheduler"])
            start=saved["next_epoch"];best_loss=saved["best_loss"];best_epoch=saved["best_epoch"]
            stale=saved["stale"];history=saved["history"];best_state=saved["best_model"];completed=saved["completed"]
            generator.bit_generator.state=saved["numpy_rng"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            if str(device).startswith("cuda") and saved.get("cuda_rng") is not None:
                # map_location=device may have moved the saved RNG states to CUDA;
                # set_rng_state_all requires CPU ByteTensors.
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    for epoch in range(start,cfg["max_epochs"]):
        if completed: break
        model.train(); permutation=generator.permutation(len(train)); totals=np.zeros(3); count=0
        for offset in range(0,len(train),cfg["batch_size"]):
            ids=permutation[offset:offset+cfg["batch_size"]]
            logits,_,_=model(_batch([graph_getter(train[i]) for i in ids],device))
            ce=torch.nn.functional.cross_entropy(logits/cfg["temperature"],torch.as_tensor(train_y[ids],device=device),weight=class_weights)
            penalty=logits.new_tensor(0.0)
            if cfg["energy_weight"]>0:
                aux_ids=generator.integers(0,len(auxiliary),size=len(ids))
                out_logits,_,_=model(_batch([graph_getter(auxiliary[i]) for i in aux_ids],device))
                penalty=energy_penalty(logits,out_logits,cfg["margin_in"],cfg["margin_out"],cfg["temperature"])
            loss=ce+cfg["energy_weight"]*penalty
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite CoDrug training loss")
            optimizer.zero_grad();loss.backward();optimizer.step()
            totals+=np.asarray([loss.item(),ce.item(),penalty.item()])*len(ids);count+=len(ids)
        validation_logits,_,_=_predict_network(model,tuning,graph_getter,device,cfg["batch_size"])
        lp=validation_logits/cfg["temperature"]
        validation_loss=float(np.mean(logsumexp(lp,axis=1)-lp[np.arange(len(tuning)),tuning_y]))
        selection_loss=validation_loss if cfg["early_stop_on"]=="router_cross_entropy" else float(totals[0]/count)
        improved=selection_loss<best_loss-1e-8
        if improved:
            best_loss=selection_loss;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        history.append({"epoch":epoch+1,"train_loss":float(totals[0]/count),"train_cross_entropy":float(totals[1]/count),
                        "train_energy_penalty":float(totals[2]/count),"router_cross_entropy":validation_loss})
        if scheduler is not None: scheduler.step()
        completed=stale>=cfg["patience"] or epoch+1>=cfg["max_epochs"]
        if checkpoint:
            saved={"training_signature":signature,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                   "best_model":best_state,"next_epoch":epoch+1,"best_loss":best_loss,"best_epoch":best_epoch,
                   "stale":stale,"history":history,"completed":completed,"numpy_rng":generator.bit_generator.state,
                   "scheduler":scheduler.state_dict() if scheduler is not None else None,
                   "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            temp=checkpoint.with_suffix(".tmp");torch.save(saved,temp);os.replace(temp,checkpoint)
            (folder/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
        if cfg["verbose"]: print(json.dumps({"stage":"codrug_training",**history[-1]}),flush=True)
        if completed: break
    if best_state is None: raise RuntimeError("No finite selected CoDrug checkpoint")
    model.load_state_dict(best_state)
    return model,graph_getter,{"epochs_completed":len(history),"best_epoch":best_epoch+1,"best_selection_loss":best_loss,
                              "best_router_cross_entropy":history[best_epoch]["router_cross_entropy"],
                              "selection_metric":cfg["early_stop_on"],"parameter_count":sum(p.numel() for p in model.parameters()),
                              "history":history,"training_signature":signature,"checkpoint_reused":bool(start)}


def fit_kde_ratios(calibration_features,target_features,config=None,seed=0,jobs=1):
    """KDE bandwidth CV uses covariates only and fits each domain separately."""
    cfg=_configuration(config)
    cal=np.asarray(calibration_features,float);target=np.asarray(target_features,float)
    if cal.ndim!=2 or target.ndim!=2 or cal.shape[1]!=target.shape[1] or min(len(cal),len(target))<2:
        raise ValueError("KDE requires aligned finite features and at least two molecules in each domain")
    if not np.isfinite(cal).all() or not np.isfinite(target).all(): raise ValueError("Nonfinite KDE features")
    estimators=[];records=[]
    for i,values in enumerate((cal,target)):
        folds=min(cfg["kde_folds"],len(values))
        if folds<2: raise ValueError("KDE cross-validation requires at least two folds")
        search=GridSearchCV(KernelDensity(kernel="gaussian"),{"bandwidth":cfg["kde_bandwidths"]},
                            cv=KFold(folds,shuffle=True,random_state=seed+i),n_jobs=jobs,error_score="raise")
        # KDE fits release the GIL; threads avoid spawning workers from a
        # process that may already hold a CUDA model and a large chemistry cache.
        with parallel_backend("threading", n_jobs=jobs):
            search.fit(values)
        estimators.append(search.best_estimator_)
        records.append({"domain":"calibration" if i==0 else "target","n":len(values),"cv_folds":folds,
                        "bandwidth":float(search.best_params_["bandwidth"]),"cv_log_likelihood":float(search.best_score_)})
    source_kde,target_kde=estimators
    log_cal=target_kde.score_samples(cal)-source_kde.score_samples(cal)
    log_target=target_kde.score_samples(target)-source_kde.score_samples(target)
    return log_cal,log_target,{"kde":records,"feature_dimension":cal.shape[1],"weight_clipping":"none; stable log arithmetic"}


def weighted_sets_from_log_ratios(calibration_probability,calibration_y,target_probability,log_calibration_ratio,log_target_ratio,alpha,mondrian=False):
    """Eqs. 22/23 with test mass at infinity, computed without weight clipping."""
    p=np.asarray(calibration_probability,float);q=np.asarray(target_probability,float)
    y=_labels(calibration_y,len(p));a=np.asarray(log_calibration_ratio,float);b=np.asarray(log_target_ratio,float)
    if p.ndim!=1 or q.ndim!=1 or a.shape!=p.shape or b.shape!=q.shape or not 0<float(alpha)<1:
        raise ValueError("Invalid probability/weight shapes or alpha")
    if not all(np.isfinite(x).all() for x in (p,q,a,b)) or np.any((p<0)|(p>1)) or np.any((q<0)|(q>1)):
        raise ValueError("Probabilities and log ratios must be finite and probabilities in [0,1]")
    true_scores=np.where(y==1,1-p,p)
    candidate_scores=np.column_stack([q,1-q]);thresholds=np.full((len(q),2),np.inf)
    for c in (0,1):
        mask=y==c if mondrian else np.ones(len(y),bool)
        if not mask.any(): continue
        scores=true_scores[mask];logs=a[mask];order=np.argsort(scores,kind="stable")
        scores=scores[order];logs=logs[order];total=logsumexp(logs)
        cdf=np.cumsum(np.exp(logs-total));cdf[-1]=1.0
        with np.errstate(over="ignore"):
            required=(1-float(alpha))*(1+np.exp(b-total))
        index=np.searchsorted(cdf,required,side="left");finite=index<len(scores)
        thresholds[finite,c]=scores[index[finite]]
    return candidate_scores<=thresholds,thresholds


def fit_predict(train_smiles,train_y,tuning_smiles,tuning_y,cal_smiles,cal_y,target_smiles,
                alphas,seed,device="cpu",jobs=1,model_dir=None,config=None,auxiliary_smiles=None):
    """Train, select on router, then calibrate/predict; target labels are absent.

    energy_weight=0 supplies an independently trained vanilla AttentiveFP for
    KMM. Other architectures/hidden dimensions must be recorded in config.
    """
    cfg=_configuration(config)
    train=[_canonical(s) for s in train_smiles];tuning=[_canonical(s) for s in tuning_smiles]
    cal=[_canonical(s) for s in cal_smiles];target=[_canonical(s) for s in target_smiles]
    if min(len(train),len(tuning),len(cal),len(target))<1: raise ValueError("All fit/router/calibration/target roles must be nonempty")
    y=_labels(train_y,len(train));vy=_labels(tuning_y,len(tuning));cy=_labels(cal_y,len(cal))
    roles=[set(train),set(tuning),set(cal)]
    if any(roles[i]&roles[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("A molecular identity occurs in multiple source label roles")
    alphas=[float(a) for a in alphas]
    if not alphas or len(set(alphas))!=len(alphas) or any(not 0<a<1 for a in alphas): raise ValueError("Invalid requested alphas")
    aux_info={"used":False};aux=[]
    if cfg["energy_weight"]>0:
        source_info={"source":"caller_supplied_unlabeled_smiles"}
        if auxiliary_smiles is None:
            cache=Path(cfg.get("auxiliary_cache_dir") or (Path(model_dir).parent/"auxiliary" if model_dir else Path.cwd()/"codrug_auxiliary"))
            auxiliary_smiles,source_info=acquire_auxiliary_smiles(cache,cfg)
        aux,filtered=filter_auxiliary_smiles(auxiliary_smiles,train+tuning+cal,cfg,seed)
        aux_info={"used":True,"source":source_info,"filter":filtered}
    training_cfg={k:v for k,v in cfg.items() if not k.startswith("kde_") and k not in {"density_representation","verbose","auxiliary_cache_dir"}}
    signature=_hash({"fit":list(zip(train,y.tolist())),"router":list(zip(tuning,vy.tolist())),"calibration_structures":cal,
                     "auxiliary_pool":aux,"configuration":training_cfg,"seed":int(seed),"implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    model,graph_getter,train_info=_train(train,y,tuning,vy,aux,cfg,int(seed),device,jobs,model_dir,signature)
    cal_logits,cal_embedding,cal_graph=_predict_network(model,cal,graph_getter,device,cfg["batch_size"])
    target_logits,target_embedding,target_graph=_predict_network(model,target,graph_getter,device,cfg["batch_size"])
    cp=softmax(cal_logits/cfg["temperature"],axis=1)[:,1];tp=softmax(target_logits/cfg["temperature"],axis=1)[:,1]
    prefix="codrug" if cfg["energy_weight"]>0 else "attentivefp_kde_noenergy"
    sets={};kde_info={"computed":False};log_cal=None
    if cfg["compute_conformal"]:
        density_cal,density_target=(cal_logits,target_logits) if cfg["density_representation"]=="logits" else (cal_embedding,target_embedding)
        log_cal,log_target,kde_info=fit_kde_ratios(density_cal,density_target,cfg,int(seed),jobs)
        for alpha in alphas:
            sets[alpha]={f"{prefix}_global":weighted_sets_from_log_ratios(cp,cy,tp,log_cal,log_target,alpha,False)[0],
                         f"{prefix}_mondrian":weighted_sets_from_log_ratios(cp,cy,tp,log_cal,log_target,alpha,True)[0]}
    info={"implementation":"independent_paper_reimplementation","paper":PAPER_URL,"configuration":cfg,
          "training":train_info,"auxiliary":aux_info,"density":kde_info,
          "source_role_n":{"fit":len(train),"router":len(tuning),"conformal":len(cal)},"target_n":len(target),
          "target_source_identity_overlap":len(set(target)&set.union(*roles)),
          "target_labels_used":False,"margin_note":"Paper appendix prints m_in=-5,m_out=-35; defaults preserve that order.",
          "guarantee":"Estimated representation-space ratios do not give an exact finite-sample coverage guarantee.",
          "versions":{"numpy":np.__version__,"torch":_torch().__version__}}
    if log_cal is not None:
        for c in (0,1):
            values=log_cal[cy==c]
            info[f"class_{c}_calibration_ess"]=float(np.exp(2*logsumexp(values)-logsumexp(2*values))) if len(values) else 0.0
    if model_dir:
        Path(model_dir,"codrug_diagnostics.json").write_text(json.dumps(info,indent=2)+"\n")
    return {"probability_1":tp,"calibration_probability":cp,"calibration_embedding":cal_embedding,
            "target_embedding":target_embedding,"calibration_graph_embedding":cal_graph,"target_graph_embedding":target_graph,
            "calibration_logits":cal_logits,"target_logits":target_logits,"sets_by_alpha":sets,"diagnostics":info}
