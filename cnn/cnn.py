import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        time_interval: int,
        num_channels: int = 8,
        num_classes: int = 8,
    ):
        """
        Time interval is in ms, should be the same used in the data generation. Ideally use
        a multiple of 8 since you int divide by 8 after 3 pooling layers

        Data has 8 input channels

        There are 8 classes (0-7)
        """
        super(CNN, self).__init__()

        # shape: (batch_size, num_channels, time_interval)
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # shape: (batch_size, 32, time_interval // 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # shape: (batch_size, 64, time_interval // 4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        # shape: (batch_size, 128, time_interval // 8)

        self.flattened_size = 128 * (time_interval // 8)

        # shape: (batch_size, flattened_size)
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)
        # shape: (batch_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_channels, time_interval)
               or (batch_size, time_interval, num_channels) - will be transposed

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Go from (batch_size, 128, time_interval // 8) to (batch_size, 128 * time_interval // 8)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
