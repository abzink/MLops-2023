import math
import os

import pytest
import torch

from tests import _PATH_DATA

# skip test if data files are not found
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestDataset:
    # check the length of the training dataset
    def test_load_traindata(self):
        self.train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
        assert len(self.train_dataset) == math.ceil(25000 / 100)

    # check the length of the training dataset
    def test_load_testdata(self):
        self.test_dataset = torch.load(f"{_PATH_DATA}/processed/test.pt")
        assert len(self.test_dataset) == 5000

    # check the shape of the training dataset
    def test_shape_traindata(self):
        self.train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
        for (images, labels) in self.train_dataset:
            assert (100, 1, 28, 28) == images.shape

    # check if the labels are not None
    def test_labels_traindata(self):
        self.train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
        for (images, labels) in self.train_dataset:
            for label in labels:
                assert labels is not None
