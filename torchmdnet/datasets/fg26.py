"""FG26 processed data; the distributed spectra file has no property labels."""
from pathlib import Path
from .base import ExactProcessedDataset


class FG26(ExactProcessedDataset):
    def __init__(self, root=None, transform=None, dataset_arg=None, processed_path=None, target_index=None):
        self.label_idx = target_index
        super().__init__(processed_path or Path(root) / "processed/fg26_uv_ir_raman_smiles_aligned.pt", transform)

    def get(self, idx):
        sample = super().get(idx).clone()
        if self.label_idx is not None:
            if sample.y is None or sample.y.reshape(1, -1).shape[1] <= self.label_idx:
                raise ValueError("FG26 property prediction requires aligned measured y labels and target_index")
            sample.y = sample.y.reshape(1, -1)[:, self.label_idx:self.label_idx + 1]
        return sample

    def get_atomref(self, max_z=100):
        # No FG26 fitted reference table is available; never borrow QM9S values.
        return None
