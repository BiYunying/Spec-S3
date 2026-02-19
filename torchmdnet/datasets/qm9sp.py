import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict


class QM9SP(QM9_geometric):
    def __init__(self, root, transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )
        self.atomref_value = { 
            1: [ 1.10980119, 8.32047488, 6.67849340, 3.73493822, 2.32304077],
            6: [ 0.31210767, 0.13700274, 0.13962825, 0.11433831, 0.09314660],
            7: [
                -16.43227145, -1036.04206374, -1489.80537863, -2046.98752059,
                -2717.50678526
            ],
            8: [
                -16.42212936, -1036.03035432, -1489.78208138, -2046.96055236,
                -2717.47921738
            ],
            9: [
                -16.42207675, -1036.02753401, -1489.77912374, -2046.95761193,
                -2717.47623487
            ],
            10: [
                -16.44549623, -1036.12438648, -1489.90877533, -2047.09688851,
                -2717.61710661
            ],
            11: [1.24340399, 2.02800400, 2.78931725, 3.08695106, 3.34167177],}
        # self.atomref_value = {
        #     6: [0., 0., 0., 0., 0.],
        #     7: [
        #         -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        #         -2713.48485589
        #     ],
        #     8: [
        #         -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        #         -2713.44632457
        #     ],
        #     9: [
        #         -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        #         -2713.42063702
        #     ],
        #     10: [
        #         -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        #         -2713.88796536
        #     ],
        #     11: [0., 0., 0., 0., 0.],
        # }        
        self.label = dataset_arg
        if dataset_arg == "alpha":   # set this value as placeholder during pre-training
            self.label = "isotropic_polarizability"
        elif dataset_arg in ["U", "U0"]:
            self.label = "energy_" + dataset_arg

        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9SP, self).__init__(root, transform=transform)

    @property
    def processed_file_names(self) -> str:
        # return "eMol9_train_validation_pool.pt"
        # return "data_with_uv_ir_raman_token_ids1.pt"
        # return "data_with_uv_ir_raman_token_ids.pt"
        return "qm9s_with_uv_ir_raman_smiles.pt"
        # return "data_with_uv_ir_raman_smiles.pt"
        # return "qm9s_with_uv_ir_raman_token_ids_.pt"

    def get_atomref(self, max_z=100):
        # atomref = self.atomref_value(self.label_idx)
        if self.label_idx in self.atomref_value:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(self.atomref_value[self.label_idx])
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

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        pass

    def process(self):
        pass


if __name__ == "__main__":
    dataset = QM9SP(root="~/datasets//qm9sp", dataset_arg="homo")
    print(dataset)
    print(dataset[0])
