# -*- coding: utf-8 -*-
import glob
import logging
from pathlib import Path

import click
import numpy as np
import torch
#from dotenv import find_dotenv, load_dotenv
from numpy import load
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train_files = glob.glob(f"{input_filepath}/train*.npz")
    test_files = glob.glob(f"{input_filepath}/test*.npz")

    images_train = []
    labels_train = []
    images_test = []
    labels_test = []

    for file in train_files:
        data_loaded = load(file)
        images_train.extend(data_loaded["images"])
        labels_train.extend(data_loaded["labels"])

    for file in test_files:
        data_loaded = load(file)
        images_test.extend(data_loaded["images"])
        labels_test.extend(data_loaded["labels"])

    x_train = torch.Tensor(np.array(images_train))
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[1])
    y_train = torch.from_numpy(np.array(labels_train))

    x_test = torch.Tensor(np.array(images_test))
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[1])
    y_test = torch.from_numpy(np.array(labels_test))

    normalize = transforms.Normalize((0.5,), (0.5,))

    train = DataLoader(
        TensorDataset(normalize(x_train), y_train), shuffle=True, batch_size=100
    )

    test = DataLoader(TensorDataset(normalize(x_test), y_test))

    torch.save(train, f"{output_filepath}/train.pt")
    torch.save(test, f"{output_filepath}/test.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()