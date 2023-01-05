import argparse
import os

import hydra
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim

import wandb

wandb.init(project="mnist-test")


@hydra.main(config_path="../../conf", config_name="config.yaml")
def main(config):
    os.chdir(hydra.utils.get_original_cwd()) # where hydra output are saved
    training_params = config.training
    model_params = config.model
    torch.manual_seed(training_params.seed)
    model = MyAwesomeModel(model_params.dropout1, model_params.dropout2)
    train_set = torch.load(training_params.train_data)
    train_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)
    print("Training day and night")
    wandb.watch(model, log_freq=100)
    for epoch in range(training_params.epochs):
        running_loss = 0
        model.train()
        n_total_steps = len(train_set)
        for i, (images, labels) in enumerate(train_set):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss / 100})
            running_loss += loss.item()

        train_losses.append(running_loss / n_total_steps)

    torch.save(model, "models/model.pth")
    plt.plot(train_losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print("Training day and night")
    main()