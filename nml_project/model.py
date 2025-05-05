import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class LSTM_GCN(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_num_layers=3, gcn_hidden=128, gcn_out=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,                   # 1 feature per node per time step
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.gcn1 = GCNConv(lstm_hidden_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_out)
        self.classifier = nn.Linear(gcn_out, 1)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [batch_size, seq_len, num_nodes]
        edge_index: [2, num_edges]
        """
        batch_size, seq_len, num_nodes = x.shape

        # Reshape to [batch_size * num_nodes, seq_len, 1]
        x = x.permute(0, 2, 1)                            # [batch, num_nodes, seq_len]
        x = x.reshape(batch_size * num_nodes, seq_len, 1)

        # Run LSTM over each node's time series
        lstm_out, _ = self.lstm(x)                        # [batch*num_nodes, seq_len, hidden_dim]
        node_feats = lstm_out[:, -1, :]                   # take last time step → [batch*num_nodes, hidden_dim]
        node_feats = node_feats.view(batch_size, num_nodes, -1)  # [batch, num_nodes, hidden_dim]

        # Build batched graphs
        graphs = []
        for i in range(batch_size):
            data = Data(
                x=node_feats[i],  # [num_nodes, hidden_dim]
                edge_index=edge_index.clone(),
                edge_weight=edge_weight.clone() if edge_weight is not None else None
            )
            graphs.append(data)

        batch_graph = Batch.from_data_list(graphs)

        # GCN layers
        x = self.gcn1(batch_graph.x, batch_graph.edge_index)
        x = F.relu(x)
        x = self.gcn2(x, batch_graph.edge_index)
        x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch_graph.batch)  # [batch_size, gcn_out]
        logits = self.classifier(x)  # [batch_size, 1]
        return logits