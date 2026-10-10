"""Task 4 proxy scoring, deterministic DFT triage and external-result evaluation.

DFT is run outside this package. Fill the exported dft_results.csv in eV, keeping
candidate_id and smiles unchanged. Set status to ok or failed for EVERY row.
Only successful finite DFT results enter the final dual-good denominator.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from test.evaluation.metrics import dual_good, identity
from test.evaluation import properties
from torchmdnet.utils import write_json


def scaffold(smiles):
    """Non-isomeric Bemis–Murcko scaffold; all acyclic molecules share one class."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid scaffold input: {smiles!r}")
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),
                           isomericSmiles=False) or "ACYCLIC"


def reference_scaffolds(config):
    if not config.get("known_reference"):
        raise ValueError("Task 4 requires known_reference: the disclosed dual-good anchor CSV")
    frame = pd.read_csv(config["known_reference"])
    if frame.empty or "smiles" not in frame:
        raise ValueError("known_reference must contain nonempty anchor smiles")
    # Compute from molecules instead of trusting an optional cached scaffold column.
    return {scaffold(s) for s in frame.smiles}


def validate_protocol(config):
    if config.get("objective") not in ("large_gap", "small_gap"):
        raise ValueError("Task 4 objective must be large_gap or small_gap")
    for name in ("homo_threshold", "lumo_threshold"):
        if not np.isfinite(float(config[name])):
            raise ValueError(f"{name} must be finite, in eV")


def score_metrics(evaluated, config, anchors, source):
    """Count distinct scaffolds only among dual-good molecules, not all candidates."""
    evaluated = evaluated.copy()
    evaluated["dual_good"] = dual_good(evaluated.homo_eV, evaluated.lumo_eV, config)
    evaluated["scaffold"] = [scaffold(s) for s in evaluated.smiles]
    evaluated["new_scaffold"] = ~evaluated.scaffold.isin(anchors)
    good_scaffolds = set(evaluated.loc[evaluated.dual_good, "scaffold"])
    count, total = int(evaluated.dual_good.sum()), len(evaluated)
    summary = dict(
        property_source=source, property_unit="eV",
        dual_good_count=count, dual_good_denominator=total,
        denominator_definition=("successful_selected_DFT_molecules" if source == "dft"
                                else "global_unique_valid_proxy_scored_molecules"),
        dual_good_rate=count / total if total else None,
        scaffold_diversity=len(good_scaffolds),
        new_scaffolds=len(good_scaffolds - anchors),
    )
    return summary, evaluated


def rank_candidates(frame, config):
    """Maximize the weaker signed target margin (eV), then their sum.

    Large gap: margins = (HOMO threshold - HOMO, LUMO - LUMO threshold).
    Small gap: reverse both signs. Thus all proxy dual-good candidates precede
    non-dual candidates. Canonical SMILES breaks ties deterministically.
    This is a public default, not a reconstruction of historical DFT panels.
    """
    frame = frame.copy()
    dh = float(config["homo_threshold"]) - frame.homo_eV
    dl = frame.lumo_eV - float(config["lumo_threshold"])
    if config["objective"] == "small_gap":
        dh, dl = -dh, -dl
    frame["proxy_score"] = np.minimum(dh, dl)
    frame["margin_sum"] = dh + dl
    return frame.sort_values(["proxy_score", "margin_sum", "smiles"],
                             ascending=[False, False, True]).drop(columns="margin_sum")


def read_dft_results(config):
    """Join by ID, verify molecular identities, and reject incomplete DFT panels.

    Required columns: candidate_id, smiles, status, homo_eV, lumo_eV.
    Energies must already be converted to eV (Hartree * 27.211386245988).
    Failed calculations retain their rows with blank energies. Missing/pending
    rows cannot silently shrink the denominator or change the selected panel.
    """
    results_path = Path(config["dft_results"])
    selection_path = Path(config.get("selection") or results_path.with_name("dft_selected.csv"))
    selected = pd.read_csv(selection_path, dtype={"candidate_id": str})
    results = pd.read_csv(results_path, dtype={"candidate_id": str}, keep_default_na=False)
    required = {"candidate_id", "smiles", "status", "homo_eV", "lumo_eV"}
    if not required.issubset(results):
        raise ValueError(f"DFT CSV requires columns: {', '.join(sorted(required))}")
    selection_fields = {"candidate_id", "smiles", "objective", "homo_threshold_eV", "lumo_threshold_eV"}
    if selected.empty or not selection_fields.issubset(selected):
        raise ValueError("Use the nonempty dft_selected.csv exported by DFT selection")
    for frame in (selected, results):
        if frame.candidate_id.isna().any() or (frame.candidate_id == "").any() or frame.candidate_id.duplicated().any():
            raise ValueError("Candidate IDs must be nonempty and unique")
    if set(results.candidate_id) != set(selected.candidate_id):
        raise ValueError("DFT IDs must match the selected panel exactly; retain failed rows")
    if not (selected.objective == config["objective"]).all():
        raise ValueError("DFT selection objective differs from the current config")
    for name in ("homo_threshold", "lumo_threshold"):
        if not np.allclose(selected[name + "_eV"], float(config[name]), rtol=0, atol=1e-10):
            raise ValueError("DFT selection thresholds differ from the current config")
    results = results.set_index("candidate_id").loc[selected.candidate_id].reset_index()
    expected = [identity(s)["canonical_smiles"] for s in selected.smiles]
    actual = [identity(s)["canonical_smiles"] for s in results.smiles]
    if not all(expected) or not all(actual) or expected != actual or len(set(expected)) != len(expected):
        raise ValueError("DFT SMILES must match the unique selected molecules for each ID")
    status = results.status.astype(str).str.strip().str.lower()
    if not status.isin(["ok", "failed"]).all():
        raise ValueError("Complete the DFT panel: status must be ok or failed (no pending rows)")
    ok = status == "ok"
    combined = selected.copy().reset_index(drop=True)
    combined["status"] = status.to_numpy()
    for name in ("homo_eV", "lumo_eV"):
        values = results[name].astype(str).str.strip()
        numbers = pd.to_numeric(values, errors="coerce")
        if not np.isfinite(numbers[ok].to_numpy()).all():
            raise ValueError("Successful DFT rows require finite HOMO/LUMO energies in eV")
        if (values[~ok] != "").any():
            raise ValueError("Leave HOMO/LUMO blank for failed DFT calculations")
        combined[name] = numbers.to_numpy()
    if (combined.loc[ok.to_numpy(), "lumo_eV"] < combined.loc[ok.to_numpy(), "homo_eV"]).any():
        raise ValueError("DFT LUMO is below HOMO; check the columns and extracted energies")
    return combined


def evaluate_multiobjective(config):
    validate_protocol(config)
    anchors = reference_scaffolds(config)
    source = config.get("property_source", "proxy")
    if source not in ("proxy", "dft"):
        raise ValueError("property_source must be proxy or dft")
    if config.get("dft_results") and source != "dft":
        raise ValueError("--dft-results requires --property-source dft")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if source == "dft" and config.get("dft_results"):
        # Evaluation reads measured results only; no proxy model is loaded here.
        combined = read_dft_results(config)
        ok = combined.status == "ok"
        summary, scored = score_metrics(combined.loc[ok], config, anchors, "dft")
        summary.update(stage="evaluation", selected_count=len(combined),
                       failed_count=int((~ok).sum()))
        combined = combined.join(scored[["dual_good", "scaffold", "new_scaffold"]])
        combined.to_csv(output / "dft_scored.csv", index=False)
    else:
        raw = pd.read_csv(config["input"]).fillna({"generated_smiles": ""})
        if "generated_smiles" not in raw:
            raise ValueError("Supply the generated candidates.csv, not the A/B/D pair manifest")
        raw["canonical_smiles"] = [identity(s)["canonical_smiles"] for s in raw.generated_smiles]
        smiles = sorted(set(raw.canonical_smiles) - {""})
        predictions = np.asarray(properties.predict_properties(smiles, config))
        if predictions.shape != (len(smiles), 2) or not np.isfinite(predictions).all():
            raise ValueError("Expected finite FG26 proxy HOMO/LUMO predictions in eV")
        evaluated = pd.DataFrame(dict(smiles=smiles, homo_eV=predictions[:, 0], lumo_eV=predictions[:, 1]))
        if source == "proxy":
            summary, scored = score_metrics(evaluated, config, anchors, "proxy")
            scored.to_csv(output / "unique_scored.csv", index=False)
        else:
            topk = config.get("topk", 50)
            if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
                raise ValueError("topk must be a positive integer")
            if not len(evaluated):
                raise ValueError("No valid candidates for DFT selection")
            # Never replace a panel or a result file the user may have filled.
            if any((output / name).exists() for name in ("dft_selected.csv", "dft_results.csv")):
                raise FileExistsError("DFT panel already exists; use a new output directory")
            selected = rank_candidates(evaluated, config).head(topk).reset_index(drop=True)
            selected = selected.rename(columns={"homo_eV": "proxy_homo_eV", "lumo_eV": "proxy_lumo_eV"})
            selected.insert(0, "candidate_id", [f"mol_{i+1:04d}" for i in range(len(selected))])
            selected["objective"] = config["objective"]
            selected["homo_threshold_eV"] = float(config["homo_threshold"])
            selected["lumo_threshold_eV"] = float(config["lumo_threshold"])
            selected.to_csv(output / "dft_selected.csv", index=False)
            template = selected[["candidate_id", "smiles"]].copy()
            template["status"] = "pending"
            template["homo_eV"] = ""
            template["lumo_eV"] = ""
            template.to_csv(output / "dft_results.csv", index=False)
            summary = dict(property_source="dft", stage="selection", requested_count=topk,
                           selected_count=len(selected), unique_valid_count=len(evaluated),
                           ranking="minimum signed target margin in eV; ties: summed margin, SMILES")
    write_json(output / "summary.json", summary)
    return summary
