import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GraphNorm, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class LSTM_GCN(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_num_layers=3, gcn_hidden=256, gcn_out=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,                   # 1 feature per node per time step
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.gcn1 = GCNConv(2*lstm_hidden_dim, gcn_hidden)
        self.norm1 = GraphNorm(gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_out)
        self.norm2 = GraphNorm(gcn_out)
        self.classifier = nn.Linear(gcn_out, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        # Initialize GCN layers
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

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
        lstm_out, _ = self.lstm(x)  # [batch*num_nodes, seq_len, 2*lstm_hidden_dim]
        node_feats = lstm_out.mean(dim=1)  # Mean pooling over time → [batch*num_nodes, 2*lstm_hidden_dim]
        node_feats = node_feats.view(batch_size, num_nodes, -1)  # [batch, num_nodes, 2*lstm_hidden_dim]

        # Build batched graphs
        graphs = []
        for i in range(batch_size):
            data = Data(
                x=node_feats[i],  # [num_nodes, 2*lstm_hidden_dim]
                edge_index=edge_index.clone(),
                edge_weight=edge_weight.clone() if edge_weight is not None else None
            )
            graphs.append(data)

        batch_graph = Batch.from_data_list(graphs)

        # GCN layers
        x_input = batch_graph.x  # Save input for residual
        x = self.gcn1(x_input, batch_graph.edge_index, edge_weight=batch_graph.edge_weight)
        x = self.norm1(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        # GCN block 2
        x_input = x  # residual input
        x = self.gcn2(x, batch_graph.edge_index, edge_weight=batch_graph.edge_weight)
        x = self.norm2(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch_graph.batch)  # [batch_size, gcn_out]
        logits = self.classifier(x)  # [batch_size, 1]
        return logits
    
    
class LSTM_GAT(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_num_layers=3,
                 gat_hidden=256, gat_out=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

        # GATConv input is (in_channels, out_channels), out = out_channels * heads if concat=True
        self.gat1 = GATConv(2 * lstm_hidden_dim, gat_hidden // num_heads, heads=num_heads, dropout=dropout)
        self.norm1 = GraphNorm(gat_hidden)
        self.gat2 = GATConv(gat_hidden, gat_hidden // num_heads, heads=num_heads, dropout=dropout)
        self.norm2 = GraphNorm(gat_hidden)
        # self.gat3 = GATConv(gat_hidden, gat_out // num_heads, heads=num_heads, dropout=dropout)
        # self.norm3 = GraphNorm(gat_out)
        self.classifier = nn.Linear(gat_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        # self.gat3.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, seq_len, num_nodes = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * num_nodes, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        node_feats = lstm_out.mean(dim=1).view(batch_size, num_nodes, -1)

        graphs = []
        for i in range(batch_size):
            data = Data(
                x=node_feats[i],
                edge_index=edge_index.clone(),
                edge_weight=edge_weight.clone() if edge_weight is not None else None
            )
            graphs.append(data)

        batch_graph = Batch.from_data_list(graphs)

        x_input = batch_graph.x
        x = self.gat1(x_input, batch_graph.edge_index)
        x = self.norm1(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x_input = x
        x = self.gat2(x, batch_graph.edge_index)
        x = self.norm2(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)
        
        # x_input = x
        # x = self.gat3(x, batch_graph.edge_index)
        # x = self.norm3(x, batch_graph.batch)
        # x = F.relu(x + x_input)
        # x = self.dropout(x)

        x = global_mean_pool(x, batch_graph.batch)
        logits = self.classifier(x)
        return logits