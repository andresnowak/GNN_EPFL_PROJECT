import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution kernel size is 7, stride 1, as per Table 2 Tconv2 for each ResNet block
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual) # residual
        out = self.relu(out)

        return out


class SMOTE(nn.Module):
    def __init__(
        self,
        num_eeg_channels=21,
        initial_filters=16,
        resnet_kernel_size=7,
        lstm_hidden_size=220,
        fc1_units=110,
        num_classes=8,
    ):
        super(SMOTE, self).__init__()

        # Initial Temporal Convolution (Tconv from Figure 3, Table 2)
        # Paper Table 2 Tconv: kernel 1x7/1, 16 filters.
        # Output 21x500x16. Interpreting as Conv1D(21, 16, k=7, s=1)
        self.tconv = nn.Sequential(
            nn.Conv1d(
                num_eeg_channels,
                initial_filters,
                kernel_size=resnet_kernel_size,
                stride=1,
                padding=(resnet_kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(initial_filters),
            nn.ReLU(inplace=True),
        )

        # 1D ResNet Module (MODULE 1 from Table 2)
        # ResNet1: 16 filters, stride 1 (first conv in block)
        self.resnet1 = ResNetBlock1D(
            initial_filters, initial_filters, kernel_size=resnet_kernel_size, stride=1
        )
        # ResNet2: 32 filters, stride 2 (first conv in block)
        current_filters = initial_filters
        self.resnet2 = ResNetBlock1D(
            current_filters,
            current_filters * 2,
            kernel_size=resnet_kernel_size,
            stride=2,
        )
        current_filters *= 2  # 32
        # ResNet3: 64 filters, stride 2
        self.resnet3 = ResNetBlock1D(
            current_filters,
            current_filters * 2,
            kernel_size=resnet_kernel_size,
            stride=2,
        )
        current_filters *= 2  # 64
        # ResNet4: 128 filters, stride 2
        self.resnet4 = ResNetBlock1D(
            current_filters,
            current_filters * 2,
            kernel_size=resnet_kernel_size,
            stride=2,
        )
        current_filters *= 2  # 128

        # Average Pooling (Table 2)
        # Stride /2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=2, stride=2
        )  # Matches stride /2 in table for final downsample

        # LSTM Module (MODULE 2 from Table 2)
        # LSTM nodes L=220
        # Input to LSTM: after ResNet4 (128 filters) and AvgPool
        # The features for LSTM are the channels from Conv layers
        self.lstm_input_features = current_filters
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Classification Module (MODULE 3 from Table 2)
        # FC1: 110 units
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_units)
        self.relu_fc1 = nn.ReLU(inplace=True)
        # FC2: 8 units (num_classes)
        self.fc2 = nn.Linear(fc1_units, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape: (batch_size, time_steps, num_eeg_channels) e.g., (B, 500, 21)
        """

        x = x.transpose(-2, -1) # to have [B, num_channels, timesteps]

        # Initial Tconv
        x = self.tconv(x)  # (B, 16, 500)

        # 1D ResNet Module
        x = self.resnet1(x)  # (B, 16, 500)
        x = self.resnet2(x)  # (B, 32, 250)
        x = self.resnet3(x)  # (B, 64, 125)
        x = self.resnet4(x)  # (B, 128, 63)

        # Average Pooling
        x = self.avg_pool(x)  # (B, 128, 31)

        # Prepare for LSTM
        # LSTM expects (batch, seq_len, features)
        # Current x: (batch, features, seq_len)
        x = x.permute(0, 2, 1)  # (B, 31, 128)
        # supppsoedly in the paper they first flatten and then do a reshape, but i don't understand why do a flattening?

        # LSTM
        # lstm_out contains all hidden states for all time steps
        # h_n contains the final hidden state: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # We take the last hidden state from the sequence (or h_n)
        # If using lstm_out:
        # x = lstm_out[:, -1, :] # (B, lstm_hidden_size)
        # If using h_n (more common for classification):
        x = h_n.squeeze(0)  # (B, lstm_hidden_size), assuming num_layers=1

        # Classification Module
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)  # (B, num_classes)

        return x
