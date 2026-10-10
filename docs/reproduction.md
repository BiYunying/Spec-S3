# Reproduction notes

See [README](../README.md) for installation and run commands.

## Configuration

- Data use the supplied fixed NPZ train/validation/test splits.
- Task 4 and Task 5 write a `config.yaml` during data preparation. Use that
  generated config for subsequent training, generation and evaluation.
- Initialize training from the relevant pretrained or warmup checkpoint;
  generate and evaluate with the resulting task checkpoint.

## Compact evaluation outputs

| Task | Metrics |
|---|---|
| Task 1: property prediction | MAE |
| Task 2: SMILES translation | Exact Top-1/5/10 |
| Task 3: scaffold editing | Validity, uniqueness, non-recovery, scaffold hopping, Tanimoto and proxy HOMO/LUMO/GAP MAE |
| IR editing | Case generation only |
| Task 4: multi-objective | Dual-good count/rate and distinct/new dual-good scaffold counts |
| Task 5: budget conditioning | Top-10 budget coverage and available spectral cosines; per-query Tanimoto |


## Task 4 DFT CSV format

Choose `property_source: proxy` for direct proxy evaluation, or `dft` to select
proxy-ranked Top-50 candidates (`--topk` controls the count). Selection produces
`dft_selected.csv` and a blank `dft_results.csv` template. Run DFT externally,
then fill the template:

```csv
candidate_id,smiles,status,homo_eV,lumo_eV
mol_0001,c1ccccc1,ok,-7.20,-0.30
mol_0002,C1CCCCC1,failed,,
```

These are format examples. Keep all exported IDs and molecules unchanged.
Set every status to `ok` or `failed`; successful rows require finite energies
in **eV**, and failed rows must have blank energies. Convert Hartree values with
`energy_eV = energy_hartree * 27.211386245988`.

Evaluate using `--property-source dft --dft-results PATH`, with the original
`dft_selected.csv` alongside it and the same generated config. Results are
written to `summary.json` and `dft_scored.csv`.

Dual-good rate uses successfully evaluated molecules as its denominator.
Scaffold counts include only dual-good molecules; new scaffolds are those
absent from the disclosed reference molecules.
