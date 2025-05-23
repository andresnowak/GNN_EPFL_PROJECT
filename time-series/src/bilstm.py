import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=1, fc_layers: list[int]=[512]):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        layers = []

        last_dim = hidden_dim * 2

        for fc_dim in fc_layers:
            layers.append(nn.Linear(last_dim, fc_dim))
            layers.append(nn.ReLU())
            last_dim = fc_dim

        layers.append(nn.Linear(last_dim, output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        """
        lstm_out, _ = self.bilstm(x)
        # Use last time step from both directions
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]
        return self.fc(last_output)
