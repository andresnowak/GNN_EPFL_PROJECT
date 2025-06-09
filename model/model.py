import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, GraphNorm, GATConv
import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

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
        adj: torch.Tensor,
        dropout: float,
        seq_len: float
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
        # Create separate GCNConv for each time step
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels, gcn_hidden) for _ in range(seq_len)
        ])

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=num_nodes * gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
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

            x_t = x[:, t, :, :]
            # Apply GCNConv with edge_index and edge_weight
            gcn = self.gcn_layers[t]
            h = gcn(
                x_t,
                self.edge_index,
                edge_weight=self.edge_weight
            )  # [batch_size * num_nodes, gcn_hidden]
            h  = F.relu(h)
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

class GCN2_LSTM_Model(nn.Module):
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
        adj: torch.Tensor,
        dropout: float,
        seq_len: float
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
        # Create separate GCNConv for each time step -> Layer 1
        self.gcn_layers1 = nn.ModuleList([
            GCNConv(in_channels, gcn_hidden) for _ in range(seq_len)
        ])
        # Create separate GCNConv for each time step -> Layer 2
        self.gcn_layers2 = nn.ModuleList([
            GCNConv(gcn_hidden, gcn_hidden) for _ in range(seq_len)
        ])

        # Normalize the graph output
        self.norm1 = GraphNorm(gcn_hidden)
        self.norm2 = GraphNorm(gcn_hidden)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=num_nodes * gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
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

            x_t = x[:, t, :, :]
            # Apply GCNConv with edge_index and edge_weight
            gcn1 = self.gcn_layers1[t]
            gcn2 = self.gcn_layers2[t]

            x_input = x_t
            h = gcn1(
                x_t,
                self.edge_index,
                edge_weight=self.edge_weight
            )  # [batch_size * num_nodes, gcn_hidden]
            h = self.norm1(h) # [batch_size * num_nodes, gcn_hidden]
            # Skip connection
            h = F.relu(h + x_input)

            x_input2 = h
            h2 = gcn2(
                h,
                self.edge_index,
                edge_weight=self.edge_weight
            )  # [batch_size * num_nodes, gcn_hidden]
            h2 = self.norm2(h2) # [batch_size * num_nodes, gcn_hidden]
            # Skip connection
            h = F.relu(h2 + x_input2)

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


class GAT_LSTM_Model(nn.Module):
    """
    GAT (PyG) + LSTM model for graph-structured time series.

    Steps to use adjacency matrix with GATConv:
    1. Convert dense adjacency `adj` ([num_nodes, num_nodes]) to sparse representation:
       ```python
       edge_index, edge_weight = dense_to_sparse(adj)
       ```
    2. Store `edge_index` and `edge_weight` as buffers in the model.
    3. In `forward()`, pass node features and these buffers to `GATConv`.
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        gat_hidden: int,
        lstm_hidden: int,
        output_dim: int,
        lstm_layers: int,
        num_heads: int,
        dropout: float,
        seq_len: float,
        adj: torch.Tensor,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_hidden = gat_hidden
        self.num_heads = num_heads

        # Convert dense adjacency to sparse format
        # adj: [num_nodes, num_nodes]
        edge_index, edge_weight = dense_to_sparse(adj)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # GATConv layer from PyTorch Geometric
        # Create separate GATConv for each time step
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels, gat_hidden, heads = num_heads) for _ in range(seq_len)
        ])

        # Linear layer to project [gat_hidden * num_heads] to [gat_hidden]
        self.projector = nn.Linear(num_nodes * gat_hidden * num_heads, num_nodes * gat_hidden)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=num_nodes * gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
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

        gat_seq = []
        for t in range(seq_len):

            # Create batches.
            graphs = []
            for i in range(batch_size):
                graphs.append(Data(
                x=x[i, t, :, :], 
                edge_index=self.edge_index.clone(), 
            ))
            batch_graph = Batch.from_data_list(graphs)

            # Apply GATConv with edge_index.
            gat = self.gat_layers[t]
            h = gat(
                batch_graph.x,
                batch_graph.edge_index,
            )  # [batch_size * num_nodes, gat_hidden * num_heads]
            h  = F.relu(h)
            # Restore batch dimension and flatten node embeddings
            h = h.reshape(batch_size, num_nodes * self.gat_hidden * self.num_heads)
            gat_seq.append(h)

        # Stack along time dimension
        gat_seq = torch.stack(gat_seq, dim=1)  # [batch_size, seq_len, num_nodes * gat_hidden * num_heads]

        # Make it compatible with LSTM
        gat_seq = self.projector(gat_seq) # [batch_size, seq_len, num_nodes * gat_hidden]
        gat_seq  = F.relu(gat_seq)

        # Pass through LSTM
        lstm_out, _ = self.lstm(gat_seq)  # [batch_size, seq_len, lstm_hidden]

        # Project to output
        last_timestep = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.proj(last_timestep)  # [batch_size, output_dim]
        return logits

class GAT2_LSTM_Model(nn.Module):
    """
    GAT (PyG) + LSTM model for graph-structured time series.

    Steps to use adjacency matrix with GATConv:
    1. Convert dense adjacency `adj` ([num_nodes, num_nodes]) to sparse representation:
       ```python
       edge_index, edge_weight = dense_to_sparse(adj)
       ```
    2. Store `edge_index` and `edge_weight` as buffers in the model.
    3. In `forward()`, pass node features and these buffers to `GATConv`.
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        gat_hidden: int,
        lstm_hidden: int,
        output_dim: int,
        lstm_layers: int,
        num_heads: int,
        dropout: float,
        seq_len: float,
        adj: torch.Tensor,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_hidden = gat_hidden
        self.num_heads = num_heads

        # Convert dense adjacency to sparse format
        # adj: [num_nodes, num_nodes]
        edge_index, edge_weight = dense_to_sparse(adj)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # GATConv layer from PyTorch Geometric
        # Create separate GATConv for each time step -> Layer 1
        self.gat_layers1 = nn.ModuleList([
            GATConv(in_channels, int(gat_hidden/num_heads), heads = num_heads) for _ in range(seq_len)
        ])

        # Normalize the graph output
        self.norm1 = GraphNorm(gat_hidden)

        # Create separate GATConv for each time step -> Layer 2
        self.gat_layers2 = nn.ModuleList([
            GATConv(gat_hidden, gat_hidden, heads = num_heads) for _ in range(seq_len)
        ])

        # Normalize the graph output
        self.norm2 = GraphNorm(gat_hidden * num_heads)

        # Linear layer to project [gat_hidden * num_heads] to [gat_hidden]
        self.projector = nn.Linear(num_nodes * gat_hidden * num_heads, num_nodes * gat_hidden)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=num_nodes * gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
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

        gat_seq = []
        for t in range(seq_len):

            # Create batches.
            graphs = []
            for i in range(batch_size):
                graphs.append(Data(
                x=x[i, t, :, :], 
                edge_index=self.edge_index.clone(), 
            ))
            batch_graph = Batch.from_data_list(graphs)

            # Apply GATConv with edge_index
            gat1 = self.gat_layers1[t]
            gat2 = self.gat_layers2[t]

            x_input = batch_graph.x
            h = gat1(
                batch_graph.x,
                batch_graph.edge_index,
            )  # [batch_size * num_nodes, gat_hidden]
            h = self.norm1(h, batch_graph.batch)
            # Skip connection
            h = F.relu(h + x_input)
            
            x_input = h
            h = gat2(
                h,
                batch_graph.edge_index,
            )  # [batch_size * num_nodes, gat_hidden * num_heads]
            h = self.norm2(h, batch_graph.batch)

            # Restore batch dimension and flatten node embeddings
            h = h.reshape(batch_size, num_nodes * self.gat_hidden * self.num_heads)
            gat_seq.append(h)

        # Stack along time dimension
        gat_seq = torch.stack(gat_seq, dim=1)  # [batch_size, seq_len, num_nodes * gat_hidden * num_heads]

        # Make it compatible with LSTM
        gat_seq = self.projector(gat_seq) # [batch_size, seq_len, num_nodes * gat_hidden]
        gat_seq  = F.relu(gat_seq)

        # Pass through LSTM
        lstm_out, _ = self.lstm(gat_seq)  # [batch_size, seq_len, lstm_hidden]

        # Project to output
        last_timestep = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.proj(last_timestep)  # [batch_size, output_dim]
        return logits