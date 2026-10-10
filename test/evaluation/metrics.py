"""Shared identity checks and the structural/spectral metrics used in results."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from rdkit import Chem

MODALITIES = ("uv", "ir", "raman")


def tanimoto(left, right, scaffold=False):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    a, b = Chem.MolFromSmiles(str(left)), Chem.MolFromSmiles(str(right))
    if a is None or b is None:
        return float("nan")
    if scaffold:
        a, b = MurckoScaffold.GetScaffoldForMol(a), MurckoScaffold.GetScaffoldForMol(b)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(a), generator.GetFingerprint(b)
        )
    )


def dual_good(homo, lumo, config):
    if config["objective"] == "large_gap":
        return (np.asarray(homo) <= config["homo_threshold"]) & (
            np.asarray(lumo) >= config["lumo_threshold"]
        )
    if config["objective"] == "small_gap":
        return (np.asarray(homo) >= config["homo_threshold"]) & (
            np.asarray(lumo) <= config["lumo_threshold"]
        )
    raise ValueError("Unknown gap objective")


@lru_cache(maxsize=200000)
def identity(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "valid": False,
            "canonical_smiles": "",
            "heavy_atoms": None,
            "formal_charge": None,
            "radical_electrons": None,
            "eligible": False,
        }
    charge = Chem.GetFormalCharge(mol)
    radicals = sum((a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": charge,
        "radical_electrons": radicals,
        "eligible": charge == 0 and radicals == 0,
    }


def cosine(x, y):
    x, y = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if x.shape != y.shape:
        raise ValueError("Exact-grid scoring cannot resample mismatched arrays")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / norm, -1, 1)) if norm > 0 else float("nan")



def spectral_pair(query, candidate, modalities=MODALITIES):
    """Raw cosine on the supplied matching grids; no peak-matching diagnostics."""
    scores = {
        "exact_" + m + "_cosine": cosine(query[m], candidate[m])
        for m in modalities
    }
    if len(modalities) > 1:
        scores["exact_multimodal_cosine"] = float(np.mean(list(scores.values())))
    return scores
