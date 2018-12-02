import random

import numpy as np
import torch.utils.data as data
from path import Path


class BioData(data.Dataset):

    """
    The dataset should be the format of following:
    y: 0.233
    X: (If there are A, B, C and D four elements)
        origin - ABBCD
        formatted -
            1 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1
            A------|B------|B------|C------|D-----|
    """
    def __init__(self, root, seed=None, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.samples = []

        self.parse_data()

    def parse_data(self):
        # TODO load the data and parse it.
        data = []

        for d in data:
            sample = {'y': d[0], 'X': d[1:]}
            self.samples.append(sample)

        pass

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
