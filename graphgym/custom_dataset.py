import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class custom_dataset(InMemoryDataset):
    url = "temp"
    def __init__(
        self,
        name,
        url,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name
        self.url = url
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.path = path

    @property
    def raw_file_names(self) -> str:
        return self.name + '.pt'

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'test_data.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        inputs = torch.load(self.raw_paths[0])
        inputs.process()
        self.data = inputs

