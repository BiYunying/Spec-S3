"""Shared exact-path loader for the retained processed datasets."""
import os
import torch
from torch_geometric.data import Data

from torch_geometric.data import InMemoryDataset


class ExactProcessedDataset(InMemoryDataset):
    """Load a specific PyG processed .pt file instead of using processed_file_names."""

    def __init__(self, processed_path, transform=None):
        self.processed_path = os.path.abspath(processed_path)
        processed_dir = os.path.dirname(self.processed_path)
        root = (
            os.path.dirname(processed_dir)
            if os.path.basename(processed_dir) == "processed"
            else processed_dir
        )
        self._processed_file_name = os.path.basename(self.processed_path)
        super().__init__(root=root, transform=transform)
        loaded = self._torch_load(self.processed_paths[0])
        if isinstance(loaded, tuple) and len(loaded) >= 2:
            self.data, self.slices = (loaded[0], loaded[1])
        elif isinstance(loaded, list):
            self.data, self.slices = self.collate(loaded)
        elif isinstance(loaded, Data):
            self.data, self.slices = (loaded, None)
        else:
            raise TypeError(
                f"Unsupported processed dataset format in {processed_path}: {type(loaded)}"
            )

    @staticmethod
    def _torch_load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @property
    def processed_dir(self):
        return os.path.dirname(self.processed_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self._processed_file_name

    def download(self):
        pass

    def process(self):
        pass
