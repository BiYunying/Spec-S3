import torch
from .base import ExactProcessedDataset
from torch_geometric.nn.models.schnet import qm9_target_dict


class QM9SP(ExactProcessedDataset):

    def __init__(self, root=None, transform=None, dataset_arg=None, processed_path=None, target_index=None):
        # Per-element least-squares references, not molecular property means.
        # H/C/N/O/F; fit on fixed QM9S train+val (120000 molecules),
        # excluding all 8140 test rows. Regenerate with scripts/fit_atomref.py.
        self.atomref_value = {
            1: [
                1.106180728456,
                8.326802760397,
                6.677487644402,
                3.734717412198,
                2.317789021811,
            ],
            6: [
                0.312124329342,
                0.136969053389,
                0.139644693583,
                0.114309914860,
                0.093204252031,
            ],
            7: [
                -16.432045290011,
                -1036.042398268434,
                -1489.806524106815,
                -2046.987127981167,
                -2717.509273110260,
            ],
            8: [
                -16.421919310276,
                -1036.030657107833,
                -1489.783230850147,
                -2046.960155689343,
                -2717.481744778053,
            ],
            9: [
                -16.421867044973,
                -1036.027836683366,
                -1489.780273316513,
                -2046.957213278489,
                -2717.478754562961,
            ],
            10: [
                -16.445240753818,
                -1036.124762353999,
                -1489.909914198658,
                -2047.096570104938,
                -2717.619707261230,
            ],
            11: [
                1.242003867997,
                2.031094135081,
                2.787934248793,
                3.085908566702,
                3.337618227859,
            ],
        }
        aliases = {"alpha": "isotropic_polarizability", "U": "energy_U", "U0": "energy_U0"}
        self.label = aliases.get(dataset_arg, dataset_arg)
        label2idx = {v: k for k, v in qm9_target_dict.items()}
        self.label_idx = label2idx[self.label] if self.label is not None else target_index
        if target_index is not None and self.label_idx != target_index:
            raise ValueError("dataset_arg and target_index select different properties")
        if self.label_idx is not None and self.label_idx not in qm9_target_dict:
            raise ValueError("Invalid QM9S target_index")
        from pathlib import Path
        path = processed_path or Path(root) / "processed/qm9s_with_uv_ir_raman_smiles.pt"
        super().__init__(path, transform=transform)

    def get(self, idx):
        sample = super().get(idx).clone()
        if self.label_idx is not None:
            if sample.y is None or sample.y.reshape(1, -1).shape[1] <= self.label_idx:
                raise ValueError("Property prediction requires aligned measured y labels")
            sample.y = sample.y.reshape(1, -1)[:, self.label_idx:self.label_idx + 1]
        return sample

    def get_atomref(self, max_z=100):
        if self.label_idx in self.atomref_value:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(
                self.atomref_value[self.label_idx]
            )
            atomref = out.view(-1, 1)
        else:
            atomref = None
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

