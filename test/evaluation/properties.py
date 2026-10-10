from __future__ import annotations


def predict_editing_properties(smiles, config):
    """Original Task3 independent RF proxies; return HOMO/LUMO/GAP in eV.

    Features match proxy_method/task3_eval.py. GAP is predicted by its own
    regressor, not replaced with the difference of the other predictions.
    """
    import joblib
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdPartialCharges

    unit = config.get("proxy_unit", "hartree").lower()
    if unit not in ("hartree", "ev"):
        raise ValueError("proxy_unit must be hartree or eV")
    if not smiles:
        return np.empty((0, 3))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f"Invalid proxy input: {value!r}")
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()
                   if a.HasProp('_GasteigerCharge')]
        descriptors = np.array([
            Crippen.MolMR(mol), Crippen.MolLogP(mol), Descriptors.MolWt(mol),
            Descriptors.TPSA(mol), Descriptors.NumValenceElectrons(mol),
            max(charges) if charges else 0., min(charges) if charges else 0.,
            sum(abs(c) for c in charges),
        ], dtype=np.float32)
        features.append(np.concatenate([
            generator.GetFingerprintAsNumPy(mol).astype(np.float32), descriptors]))
    features = np.asarray(features)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite Task3 proxy features (including Gasteiger charges)")
    predictions = []
    # Each forest is ~1 GB; load/release sequentially rather than keeping three.
    for name in ("HOMO", "LUMO", "GAP"):
        model = joblib.load(Path(config['proxy_dir']) / f'proxy_{name}.pkl')
        if hasattr(model, 'n_jobs'):
            model.n_jobs = int(config.get('proxy_jobs', 1))
        predictions.append(np.asarray(model.predict(features), dtype=float))
        del model
    values = np.column_stack(predictions)
    if values.shape != (len(smiles), 3) or not np.isfinite(values).all():
        raise ValueError("Invalid Task3 proxy predictions")
    return values * (27.211386245988 if unit == "hartree" else 1.)


def predict_properties(smiles, config):
    """Return HOMO/LUMO in eV; keep proxy outputs labelled as predictions."""
    if str(config.get("dataset", "")).lower() != "fg26":
        raise ValueError("Task 4 supports FG26 only")
    if not smiles:
        return np.empty((0, 2))
    device = config.get("device") or "cpu"
    model, mean, std, _ = load_proxy(Path(config["proxy_path"]), device)
    predictions, _ = proxy_predict(
        smiles, model, mean, std, config.get("proxy_batch_size", 256), device
    )
    return predictions


from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict, Sequence
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from tqdm import tqdm

HOMO_RANGE = (-7.5, -5.5)
LUMO_RANGE = (-2.6, -1.6)


def interval_labels(y: np.ndarray) -> np.ndarray:
    homo_ok = (y[:, 0] >= HOMO_RANGE[0]) & (y[:, 0] <= HOMO_RANGE[1])
    lumo_ok = (y[:, 1] >= LUMO_RANGE[0]) & (y[:, 1] <= LUMO_RANGE[1])
    return np.column_stack([homo_ok, lumo_ok, homo_ok & lumo_ok]).astype(np.float32)


def one_hot(value, choices: Sequence) -> List[float]:
    return [float(value == x) for x in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    vals: List[float] = []
    vals += one_hot(atom.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53])
    vals += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    vals += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    vals += one_hot(int(atom.GetChiralTag()), [0, 1, 2, 3])
    vals += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    vals += one_hot(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    )
    vals += [float(atom.GetIsAromatic()), float(atom.GetMass() / 100.0)]
    return np.asarray(vals, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    vals: List[float] = []
    vals += one_hot(
        bond.GetBondType(),
        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
    )
    vals += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    vals += one_hot(int(bond.GetStereo()), [0, 1, 2, 3, 4, 5])
    return np.asarray(vals, dtype=np.float32)


@dataclass
class GraphRecord:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    row_index: int


def smiles_to_graph(smiles: str, y: np.ndarray, row_index: int) -> GraphRecord:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(smiles)
    x = np.stack([atom_features(a) for a in mol.GetAtoms()])
    edges, attrs = ([], [])
    for bond in mol.GetBonds():
        u, v = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        feat = bond_features(bond)
        edges += [(u, v), (v, u)]
        attrs += [feat, feat]
    edge_index = (
        np.asarray(edges, dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_attr = np.stack(attrs) if attrs else np.zeros((0, 14), dtype=np.float32)
    return GraphRecord(
        x,
        edge_index,
        edge_attr,
        y.astype(np.float32),
        interval_labels(y[None, :])[0],
        row_index,
    )


class GraphDataset(Dataset):

    def __init__(self, records: Sequence[GraphRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphRecord:
        return self.records[index]


def collate_graphs(records: Sequence[GraphRecord]) -> Dict[str, torch.Tensor]:
    xs, eis, eas, batches, ys, labels, row_indices = ([], [], [], [], [], [], [])
    atom_offset = 0
    for graph_id, rec in enumerate(records):
        xs.append(torch.from_numpy(rec.x))
        eis.append(torch.from_numpy(rec.edge_index) + atom_offset)
        eas.append(torch.from_numpy(rec.edge_attr))
        batches.append(torch.full((len(rec.x),), graph_id, dtype=torch.long))
        ys.append(torch.from_numpy(rec.y))
        labels.append(torch.from_numpy(rec.labels))
        row_indices.append(rec.row_index)
        atom_offset += len(rec.x)
    return {
        "x": torch.cat(xs),
        "edge_index": torch.cat(eis, dim=1),
        "edge_attr": torch.cat(eas),
        "batch": torch.cat(batches),
        "y": torch.stack(ys),
        "labels": torch.stack(labels),
        "row_index": torch.tensor(row_indices, dtype=torch.long),
    }


class DirectedMPNN(nn.Module):

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden: int,
        depth: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden = hidden
        self.depth = depth
        self.edge_in = nn.Linear(atom_dim + bond_dim, hidden)
        self.edge_update = nn.Linear(hidden, hidden)
        self.atom_out = nn.Linear(atom_dim + hidden, hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden, 2)
        self.classification_head = nn.Linear(hidden, 3)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr, graph_id = (
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["batch"],
        )
        n_atoms = x.shape[0]
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            h0 = torch.relu(self.edge_in(torch.cat([x[src], edge_attr], dim=1)))
            h = h0
            reverse = torch.arange(h.shape[0], device=h.device) ^ 1
            for _ in range(self.depth - 1):
                incoming = torch.zeros(
                    (n_atoms, self.hidden), device=x.device, dtype=x.dtype
                )
                incoming.index_add_(0, dst, h)
                message = incoming[src] - h[reverse]
                h = torch.relu(h0 + self.edge_update(message))
            atom_message = torch.zeros(
                (n_atoms, self.hidden), device=x.device, dtype=x.dtype
            )
            atom_message.index_add_(0, dst, h)
        else:
            atom_message = torch.zeros(
                (n_atoms, self.hidden), device=x.device, dtype=x.dtype
            )
        atom_hidden = torch.relu(self.atom_out(torch.cat([x, atom_message], dim=1)))
        n_graphs = int(graph_id.max().item()) + 1
        pooled = torch.zeros((n_graphs, self.hidden), device=x.device, dtype=x.dtype)
        pooled.index_add_(0, graph_id, atom_hidden)
        counts = (
            torch.bincount(graph_id, minlength=n_graphs)
            .clamp_min(1)
            .to(x.dtype)
            .unsqueeze(1)
        )
        pooled = self.ffn(pooled / counts)
        return (self.regression_head(pooled), self.classification_head(pooled))


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict_dmpnn(
    model: DirectedMPNN,
    records: Sequence[GraphRecord],
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        GraphDataset(records),
        batch_size=args.dmpnn_batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    model.eval()
    preds, probs, indices = ([], [], [])
    for raw in loader:
        batch = move_batch(raw, args.device)
        reg, logits = model(batch)
        preds.append(reg.cpu().numpy() * std + mean)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        indices.append(raw["row_index"].numpy())
    return (np.concatenate(preds), np.concatenate(probs), np.concatenate(indices))


def load_proxy(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    model = DirectedMPNN(
        int(meta["atom_dim"]),
        int(meta["bond_dim"]),
        int(meta["hidden"]),
        int(meta["depth"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return (
        model,
        np.asarray(meta["target_mean"]),
        np.asarray(meta["target_std"]),
        meta,
    )


def proxy_predict(
    smiles: Sequence[str],
    model,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    records = [
        smiles_to_graph(value, np.zeros(2, dtype=np.float32), row_index)
        for row_index, value in enumerate(tqdm(smiles, desc="Building D-MPNN graphs"))
    ]
    args = SimpleNamespace(dmpnn_batch_size=batch_size, device=str(device))
    predictions, probabilities, order = predict_dmpnn(model, records, mean, std, args)
    if not np.array_equal(order, np.arange(len(smiles))):
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        predictions, probabilities = (predictions[inverse], probabilities[inverse])
    return (predictions, probabilities)
