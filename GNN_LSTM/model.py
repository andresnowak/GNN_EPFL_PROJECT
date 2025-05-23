import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GCN_LSTM_Model(nn.Module):
    """
    GCN (PyG) + LSTM model for graph-structured time series.

    Steps to use adjacency matrix with GCNConv:
    1. Convert dense adjacency `adj` ([num_nodes, num_nodes]) to sparse representation:
       ```python
       edge_index, edge_weight = dense_to_sparse(adj)
       ```
    2. Store `edge_index` and `edge_weight` as buffers in the model.
    3. In `forward()`, pass node features and these buffers to `GCNConv`.
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        gcn_hidden: int,
        lstm_hidden: int,
        output_dim: int,
        lstm_layers: int,
        adj: torch.Tensor
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn_hidden = gcn_hidden

        # Convert dense adjacency to sparse format
        # adj: [num_nodes, num_nodes]
        edge_index, edge_weight = dense_to_sparse(adj)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # GCNConv layer from PyTorch Geometric
        self.gcn = GCNConv(in_channels, gcn_hidden)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=num_nodes * gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Final projection per time step
        self.proj = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, num_nodes, in_channels]
        returns: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, num_nodes, in_channels = x.size()
        assert num_nodes == self.num_nodes, (
            f"Expected {self.num_nodes} nodes, got {num_nodes}"
        )

        gcn_seq = []
        for t in range(seq_len):
            # Flatten batch and node dims for PyG
            x_t = x[:, t, :, :].reshape(-1, in_channels)  # [batch_size * num_nodes, in_channels]
            # Apply GCNConv with edge_index and edge_weight
            h = self.gcn(
                x_t,
                self.edge_index,
                edge_weight=self.edge_weight
            )  # [batch_size * num_nodes, gcn_hidden]
            # Restore batch dimension and flatten node embeddings
            h = h.reshape(batch_size, num_nodes * self.gcn_hidden)
            gcn_seq.append(h)

        # Stack along time dimension
        gcn_seq = torch.stack(gcn_seq, dim=1)  # [batch_size, seq_len, num_nodes * gcn_hidden]

        # Pass through LSTM
        lstm_out, _ = self.lstm(gcn_seq)  # [batch_size, seq_len, lstm_hidden]

        # Project to output
        last_timestep = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.proj(last_timestep)  # [batch_size, output_dim]
        return logits
