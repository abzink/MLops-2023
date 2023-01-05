import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, dropout1, dropout2):
        super(MyAwesomeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            print("{}, {}, {}".format(x.shape[1], x.shape[2], x.shape[3]))
            raise ValueError("Expected each x sample to have shape 1,28,28")

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output