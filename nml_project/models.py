import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GraphNorm, global_mean_pool, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from collections import deque
from pathlib import Path
from s4.models.s4.s4 import S4Block


class GAT(nn.Module):
    def __init__(
        self,
        gat_input: int   = 354,
        gat_hidden: int  = 352,
        gat_out: int     = 352,
        num_heads: int   = 4,
        dropout: float   = 0.2,
    ):
        super().__init__()
        per_head_hidden = gat_hidden // num_heads
        per_head_out    = gat_out    // num_heads

        self.gat1 = GATConv(gat_input, per_head_hidden, heads=num_heads, dropout=dropout)
        self.norm1 = GraphNorm(gat_hidden)

        self.gat2 = GATConv(gat_hidden, per_head_hidden, heads=num_heads, dropout=dropout)
        self.norm2 = GraphNorm(gat_hidden)

        self.gat3 = GATConv(gat_hidden, per_head_out, heads=num_heads, dropout=dropout)
        self.norm3 = GraphNorm(gat_out)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gat_out, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.gat3.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x, edge_index, edge_weight=None):
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.shape

        graphs = []
        for i in range(batch_size):
            graphs.append(Data(
                x=x[i], 
                edge_index=edge_index, 
                edge_weight=edge_weight
            ))
        batch_graph = Batch.from_data_list(graphs)

        h0 = batch_graph.x
        h  = self.gat1(h0, batch_graph.edge_index)
        h  = self.norm1(h, batch_graph.batch)
        h  = F.relu(h)
        h  = self.dropout(h)

        h0 = h
        h  = self.gat2(h0, batch_graph.edge_index)
        h  = self.norm2(h, batch_graph.batch)
        h  = F.relu(h + h0)
        h  = self.dropout(h)

        h0 = h
        h  = self.gat3(h0, batch_graph.edge_index)
        h  = self.norm3(h, batch_graph.batch)
        h  = F.relu(h + h0)
        h  = self.dropout(h)

        h = global_mean_pool(h, batch_graph.batch)
        logits = self.classifier(h)
        return logits

class LSTM_GCN(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_num_layers=3, gcn_hidden=256, gcn_out=256, dropout=0.2):
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
        self.gcn1 = GCNConv(2*lstm_hidden_dim, gcn_hidden)
        self.norm1 = GraphNorm(gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.norm2 = GraphNorm(gcn_hidden)
        self.gcn3 = GCNConv(gcn_hidden, gcn_out)
        self.norm3 = GraphNorm(gcn_out)
        self.classifier = nn.Linear(gcn_out, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [batch_size, seq_len, num_nodes]
        edge_index: [2, num_edges]
        """
        batch_size, seq_len, num_nodes = x.shape

        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_nodes, seq_len, 1)

        lstm_out, _ = self.lstm(x)
        node_feats = lstm_out.mean(dim=1)
        node_feats = node_feats.view(batch_size, num_nodes, -1)

        # Build batched graphs
        graphs = []
        for i in range(batch_size):
            data = Data(
                x=node_feats[i],  # [num_nodes, 2*lstm_hidden_dim]
                edge_index=edge_index.clone(),
                edge_attr=edge_weight.clone() if edge_weight is not None else None
            )
            graphs.append(data)

        batch_graph = Batch.from_data_list(graphs)

        x_input = batch_graph.x
        x = self.gcn1(x_input, batch_graph.edge_index, edge_weight=batch_graph.edge_weight)
        x = self.norm1(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x_input = x
        x = self.gcn2(x, batch_graph.edge_index, edge_weight=batch_graph.edge_weight)
        x = self.norm2(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x_input = x
        x = self.gcn3(x, batch_graph.edge_index, edge_weight=batch_graph.edge_weight)
        x = self.norm3(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x = global_mean_pool(x, batch_graph.batch)
        logits = self.classifier(x)
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
        self.gat1 = GATConv(2 * lstm_hidden_dim, gat_hidden // num_heads, heads=num_heads, dropout=dropout)
        self.norm1 = GraphNorm(gat_hidden)
        self.gat2 = GATConv(gat_hidden, gat_hidden // num_heads, heads=num_heads, dropout=dropout)
        self.norm2 = GraphNorm(gat_hidden)
        self.gat3 = GATConv(gat_hidden, gat_out // num_heads, heads=num_heads, dropout=dropout)
        self.norm3 = GraphNorm(gat_out)
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
        self.gat3.reset_parameters()
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
        x = self.gat1(x_input, batch_graph.edge_index, edge_attr=batch_graph.edge_attr)
        x = self.norm1(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x_input = x
        x = self.gat2(x, batch_graph.edge_index, edge_attr=batch_graph.edge_attr)
        x = self.norm2(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)
        
        x_input = x
        x = self.gat3(x, batch_graph.edge_index, edge_attr=batch_graph.edge_attr)
        x = self.norm3(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x = global_mean_pool(x, batch_graph.batch)
        logits = self.classifier(x)
        return logits
    
class LSTM_(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=19,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.classifier = nn.Linear(lstm_hidden_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x_pooled = lstm_out.mean(dim=1)
        logits = self.classifier(x_pooled)
        return logits

class CNN_GAT(nn.Module):
    def __init__(self, cnn_out_dim=256, gat_hidden=256, gat_out=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_cnn = nn.Linear(256, cnn_out_dim)

        self.dropout = nn.Dropout(dropout)
        self.gat1 = GATConv(cnn_out_dim, gat_hidden // num_heads, heads=num_heads, dropout=dropout)
        self.norm1 = GraphNorm(gat_hidden)
        self.gat2 = GATConv(gat_hidden, gat_hidden // num_heads, heads=num_heads, dropout=dropout)
        self.norm2 = GraphNorm(gat_hidden)
        self.gat3 = GATConv(gat_hidden, gat_out // num_heads, heads=num_heads, dropout=dropout)
        self.norm3 = GraphNorm(gat_out)

        self.classifier = nn.Linear(gat_out, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.cnn:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.fc_cnn.weight)
        nn.init.constant_(self.fc_cnn.bias, 0)

        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.gat3.reset_parameters()

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, seq_len, num_nodes = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * num_nodes, 1, seq_len)
        cnn_out = self.cnn(x).squeeze(-1)
        node_feats = self.fc_cnn(cnn_out).view(batch_size, num_nodes, -1)

        # Build graph batch
        graphs = []
        for i in range(batch_size):
            data = Data(
                x=node_feats[i],
                edge_index=edge_index.clone(),
                edge_attr=edge_weight.clone() if edge_weight is not None else None
            )
            graphs.append(data)

        batch_graph = Batch.from_data_list(graphs)

        x_input = batch_graph.x
        x = self.gat1(x_input, batch_graph.edge_index, edge_attr=batch_graph.edge_attr)
        x = self.norm1(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x_input = x
        x = self.gat2(x, batch_graph.edge_index, edge_attr=batch_graph.edge_attr)
        x = self.norm2(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x_input = x
        x = self.gat3(x, batch_graph.edge_index, edge_attr=batch_graph.edge_attr)
        x = self.norm3(x, batch_graph.batch)
        x = F.relu(x + x_input)
        x = self.dropout(x)

        x = global_mean_pool(x, batch_graph.batch)
        logits = self.classifier(x)
        return logits
    
class LSTM_GraphTransformer(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_num_layers=3,
                 gtf_hidden=256, gtf_out=256, num_heads=4, dropout=0.2, pos_enc_dim=8):
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

        self.pos_encoder = AddLaplacianEigenvectorPE(
            k=pos_enc_dim,
            normalization='sym'
        )

        in_dim = 2 * lstm_hidden_dim + pos_enc_dim
        self.gtf1 = TransformerConv(in_dim,  gtf_hidden // num_heads, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm1 = GraphNorm(gtf_hidden)
        self.gtf2 = TransformerConv(gtf_hidden, gtf_hidden // num_heads, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm2 = GraphNorm(gtf_hidden)
        self.gtf3 = TransformerConv(gtf_hidden, gtf_out // num_heads, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm3 = GraphNorm(gtf_out)

        self.classifier = nn.Linear(gtf_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        self.gtf1.reset_parameters()
        self.gtf2.reset_parameters()
        self.gtf3.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, edge_index, edge_weight=None):
        device = x.device
        B, T, N = x.size()
        x = x.permute(0, 2, 1).reshape(B * N, T, 1)
        lstm_out, _ = self.lstm(x)
        node_feats = lstm_out.mean(dim=1).view(B, N, -1)
        graphs = []
        for i in range(B):
            data = Data(x=node_feats[i],
                        edge_index=edge_index,
                        edge_attr=edge_weight)
            data = self.pos_encoder(data)
            graphs.append(data)
        batch = Batch.from_data_list(graphs).to(device)
        batch.x = torch.cat([batch.x, batch.laplacian_eigenvector_pe], dim=-1)
        h = batch.x
        for tf, norm in [(self.gtf1, self.norm1),
                         (self.gtf2, self.norm2),
                         (self.gtf3, self.norm3)]:
            h_in = h
            h = tf(h, batch.edge_index, edge_attr=batch.edge_attr)
            h = norm(h, batch.batch)
            h = F.relu(h + h_in)
            h = self.dropout(h)
        out = global_mean_pool(h, batch.batch)
        return self.classifier(out)

class S4_(nn.Module):
    def __init__(self, s4_hidden_dim=128, s4_dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(1, s4_hidden_dim)
        self.s4 = S4Block(
            d_model=s4_hidden_dim,
            dropout=s4_dropout,
            transposed=True,
            l_max=354
        )
        self.dropout = nn.Dropout(s4_dropout)
        self.classifier = nn.Linear(s4_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.s4, "reset_parameters"):
            self.s4.reset_parameters()
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        B, T, N = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * N, T, 1)
        x_proj = self.input_proj(x_flat)
        x_s4_in = x_proj.transpose(1, 2)
        z, _ = self.s4(x_s4_in)
        node_feats = z.mean(dim=-1)
        node_feats = node_feats.view(B, N, -1)
        node_feats = self.dropout(node_feats)
        graph_feats = node_feats.mean(dim=1)
        logits = self.classifier(graph_feats)
        
        return logits


class S4_GCN(nn.Module):
    def __init__(
        self,
        s4_hidden_dim=128,
        s4_dropout=0.2,
        gcn_hidden=256,
        gcn_out=256,
        gcn_dropout=0.2
    ):
        super().__init__()

        self.input_proj = nn.Linear(1, s4_hidden_dim)
        self.s4 = S4Block(
            d_model=s4_hidden_dim,
            dropout=s4_dropout,
            transposed=True,
            l_max=354
        )
        self.gcn1 = GCNConv(s4_hidden_dim, gcn_hidden)
        self.norm1 = GraphNorm(gcn_hidden)
        self.res_proj1 = nn.Linear(s4_hidden_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_out)
        self.norm2 = GraphNorm(gcn_out)
        self.dropout = nn.Dropout(gcn_dropout)
        self.classifier = nn.Linear(gcn_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.s4, "reset_parameters"):
            self.s4.reset_parameters()
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, edge_weight=None):
        B, T, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, T, 1)
        x = self.input_proj(x)
        x = x.transpose(-1, -2)
        z, _ = self.s4(x)
        node_feats = z.mean(dim=-1).view(B, N, -1)

        graphs = []
        for i in range(B):
            graphs.append(Data(
                x=node_feats[i],
                edge_index=edge_index.clone(),
                edge_weight=edge_weight.clone() if edge_weight is not None else None
            ))
        batch = Batch.from_data_list(graphs)

        h0 = batch.x
        h = self.gcn1(h0, batch.edge_index, batch.edge_weight)
        h = self.norm1(h, batch.batch)
        res = self.res_proj1(h0)
        h = F.relu(h + res)
        h = self.dropout(h)

        h1 = h
        h = self.gcn2(h, batch.edge_index, batch.edge_weight)
        h = self.norm2(h, batch.batch)
        h = F.relu(h + h1)
        h = self.dropout(h)

        out = global_mean_pool(h, batch.batch)

        return self.classifier(out)

class S4_GAT(nn.Module):
    def __init__(
        self,
        s4_hidden_dim=128,
        gat_hidden=256,
        gat_out=256,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, s4_hidden_dim)
        self.s4 = S4Block(
            d_model=s4_hidden_dim,
            dropout=dropout,
            transposed=True,
            l_max=354
        )
        self.dropout = nn.Dropout(dropout)
        self.s4_to_gat = nn.Linear(s4_hidden_dim, gat_hidden)

        self.gat1 = GATConv(
            in_channels=gat_hidden,
            out_channels=gat_hidden // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        self.norm1 = GraphNorm(gat_hidden)

        self.gat2 = GATConv(
            in_channels=gat_hidden,
            out_channels=gat_hidden // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        self.norm2 = GraphNorm(gat_hidden)

        self.gat3 = GATConv(
            in_channels=gat_hidden,
            out_channels=gat_out // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        self.norm3 = GraphNorm(gat_out)

        self.classifier = nn.Linear(gat_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.s4, "reset_parameters"):
            self.s4.reset_parameters()
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.s4_to_gat.weight)
        nn.init.zeros_(self.s4_to_gat.bias)
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.gat3.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, edge_weight=None):
        B, T, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, T, 1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        z, _ = self.s4(x)
        node_feats = z.mean(dim=-1).view(B, N, -1)
        node_feats = self.dropout(node_feats)
        node_feats = self.s4_to_gat(node_feats)

        graphs = []
        for i in range(B):
            graphs.append(Data(
                x=node_feats[i],
                edge_index=edge_index.clone(),
                edge_attr=edge_weight.clone() if edge_weight is not None else None
            ))
        batch = Batch.from_data_list(graphs)
        ew = batch.edge_attr

        h0 = batch.x
        h = self.gat1(h0, batch.edge_index, ew)
        h = self.norm1(h, batch.batch)
        h = F.relu(h + h0)
        h = self.dropout(h)

        h1 = h
        h = self.gat2(h, batch.edge_index, ew)
        h = self.norm2(h, batch.batch)
        h = F.relu(h + h1)
        h = self.dropout(h)

        h2 = h
        h = self.gat3(h, batch.edge_index, ew)
        h = self.norm3(h, batch.batch)
        h = F.relu(h + h2)
        h = self.dropout(h)

        out = global_mean_pool(h, batch.batch)
        
        return self.classifier(out)
    
# ---- resnet_lstm -----


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


class SMOTE_BI(nn.Module):
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
            bidirectional=True,
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

        x = x.transpose(-2, -1)  # to have [B, num_channels, timesteps]

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
        h_n = h_n.mean(dim=0)

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

# ---- stgcn -----

# Code obtained from https://github.com/FelixOpolka/STGCN-PyTorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
    

class STGCNClassifier(nn.Module):
    def __init__(self, num_nodes, num_features=1, num_classes=1):
        super(STGCNClassifier, self).__init__()
        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=64,
            spatial_channels=16,
            num_nodes=num_nodes,
        )
        self.block2 = STGCNBlock(
            in_channels=64, out_channels=64, spatial_channels=16, num_nodes=num_nodes
        )
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (nodes, time) → (1, 1)
        self.classifier = nn.Linear(64, num_classes)  # 64 from channels after conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        res_X = X
        out = self.block1(X, A_hat)
        # out += res_X
        # res_X = out
        out = self.block2(out, A_hat)
        # out += res_X
        # res_X = out
        out = self.last_temporal(out)  # shape: (B, N, T, C)
        # out += res_X
        out = out.permute(0, 3, 1, 2)  # (B, C, N, T) for pooling
        out = self.pool(out)  # → (B, C, 1, 1)
        out = out.view(out.size(0), -1)  # → (B, C)
        out = self.classifier(out)  # → (B, 1)
        return out
    

class STGCNClassifier_2(nn.Module):
    def __init__(self, num_nodes, num_features=1, num_classes=1):
        super(STGCNClassifier_2, self).__init__()
        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=64,
            spatial_channels=16,
            num_nodes=num_nodes,
        )
        self.block2 = STGCNBlock(
            in_channels=64, out_channels=64, spatial_channels=16, num_nodes=num_nodes
        )
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (nodes, time) → (1, 1)

        self.linear_1 = nn.Linear(64, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)  # 64 from channels after conv

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        res_X = X
        out = self.block1(X, A_hat)
        # out += res_X
        # res_X = out
        out = self.block2(out, A_hat)
        # out += res_X
        # res_X = out
        out = self.last_temporal(out)  # shape: (B, N, T, C)
        # out += res_X
        out = out.permute(0, 3, 1, 2)  # (B, C, N, T) for pooling
        out = self.pool(out)  # → (B, C, 1, 1)
        out = out.view(out.size(0), -1)  # → (B, C)

        out = F.relu(self.linear_1(out))
        out = F.relu(self.linear_2(out))
        out = self.classifier(out)  # → (B, 1)
        return out


class STGCNClassifier_AttnPool(nn.Module):
    def __init__(self, num_nodes, num_features=1, num_classes=1):
        super().__init__()
        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=64,
            spatial_channels=16,
            num_nodes=num_nodes,
        )
        self.block2 = STGCNBlock(
            in_channels=64, out_channels=64, spatial_channels=16, num_nodes=num_nodes
        )
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

        # attention scorer: maps each channel embedding → a scalar score
        self.attn_scorer = nn.Conv2d(64, 1, kernel_size=1)  

        self.linear_1 = nn.Linear(64, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)  # 64 from channels after conv

    def forward(self, X, A_hat):
        # X: (B, N, T, F)
        out = self.block1(X, A_hat)         # (B, N, T', C)
        out = self.block2(out, A_hat)       # (B, N, T'', C)
        out = self.last_temporal(out)       # (B, N, T''', C)

        # bring to (B, C, N, T) for conv
        z = out.permute(0, 3, 1, 2)         # (B, C, N, T)
        # compute attention scores per node+time
        scores = self.attn_scorer(z)        # (B, 1, N, T)
        weights = torch.softmax(scores.view(z.size(0), -1), dim=1)
        weights = weights.view_as(scores)   # (B,1,N,T)

        # weighted sum: broadcast weights over channels
        z_weighted = (z * weights).sum(dim=[2,3])  # (B, C)
        out = F.relu(self.linear_1(z_weighted))
        out = F.relu(self.linear_2(out))
        out = self.classifier(out) # (B, num_classes)
        return out 


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + torch.diag(torch.ones(A.shape[0], dtype=torch.float32))
    D = torch.sum(A, dim=1).reshape(-1,)
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    A_wave = torch.multiply(torch.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


# --- TGCN ---- 

# code form https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-PyTorch/models/tgcn.py

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
    

class TGCNClassifier(nn.Module):
    def __init__(self, adj, hidden_dim: int, num_classes: int):
        super(TGCNClassifier, self).__init__()

        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn = TGCN(adj, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes)
        tgcn_output = self.tgcn(inputs)  # (batch_size, num_nodes, hidden_dim)

        # Global Average Pooling over nodes (or you can use sum/max/etc.)
        graph_embedding = tgcn_output.mean(dim=1)  # (batch_size, hidden_dim)

        # Classification layer
        logits = self.classifier(graph_embedding)  # (batch_size, num_classes)
        return logits
    

# ---- DTGCN -----


def calculate_laplacian_with_self_loop_2(matrix):
    batch_size, num_nodes, _ = matrix.shape
    eye = torch.eye(num_nodes).expand(batch_size, -1, -1).to(matrix.device)
    matrix = matrix + eye
    row_sum = matrix.sum(dim=2)  # sum over last dimension
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    # Create diagonal matrices for each batch
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    normalized_laplacian = torch.bmm(torch.bmm(matrix, d_mat_inv_sqrt), 
                                    d_mat_inv_sqrt.transpose(-2, -1))

    return normalized_laplacian


class TGCNGraphConvolution_2(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution_2, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state, adj):
        # Ensure adj is on the same device as inputs
        adj = adj.to(inputs.device)
        laplacian = calculate_laplacian_with_self_loop_2(adj)

        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # batch size, num_nodes, gru_units, we want concatenation to still be batch_size
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        # concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        # concatenation = concatenation.reshape(
        #     (num_nodes, (self._num_gru_units + 1) * batch_size)
        # )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        # a_times_concat = a_times_concat.reshape(
        #     (num_nodes, self._num_gru_units + 1, batch_size)
        # )
        a_times_concat = a_times_concat.reshape(
            (batch_size, num_nodes, self._num_gru_units + 1)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell_2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell_2, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGCNGraphConvolution_2(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution_2(
            self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state, adj):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state, adj))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, adj))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
    

class TGCN_2(nn.Module):
    def __init__(self, input_dim, hidden_dim: int, **kwargs):
        super(TGCN_2, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.tgcn_cell = TGCNCell_2(self._input_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state, adj):
        batch_size, num_nodes, seq_len = inputs.shape
        assert self._input_dim == num_nodes
        assert list(hidden_state.shape) == [batch_size, num_nodes * self._hidden_dim]
        # hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
        #     inputs
        # )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, :, i], hidden_state, adj)
            # output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class GraphStructureLearner(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        embed_dim: int,
        alpha: float = 1.0,
    ):
        super(GraphStructureLearner, self).__init__()

        self.alpha = alpha

        # the amount of in_channels is the the size of the node v features
        self.gconv = GCNConv(in_channels=node_feature_dim, out_channels=embed_dim)

        # calculate the dynamic features of src and dst nodes (allows the model to learn directed or asymmetric relationships between node)
        self.linear_de1 = nn.Linear(embed_dim, embed_dim)
        self.linear_de2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, v_h, edge_list, edge_weights):
        # We want batch_size, the number of nodes and then the features for gconv

        df = self.gconv(v_h, edge_list, edge_weights)

        de1 = torch.tanh(self.alpha * self.linear_de1(df))
        de2 = torch.tanh(self.alpha * self.linear_de2(df))

        # Et = RELU(tanh(alpha * (DE1 * DE2^T - DE2 * DE1^T))) - Simplified interpretation
        Et = torch.relu(
            torch.tanh(
                self.alpha * (de1 @ de2.transpose(-2, -1) - de2 @ de1.transpose(-2, -1))
            )
        )

        # mt = Et.mean(dim = 1) # the adjacency matrix at time t will be the average of the windows (that we did on the 12 second window coarse grained lable)

        return Et


class DTGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        gsl_embed_dim: int,
        tgcn_hidden_dim: int,  # The H size
        window_size: int,  # K window size
        gsl_alpha: float,
        gsl_avg_steps: int,
        num_classes: int = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.gsl_embed_dim = gsl_embed_dim
        self.window_size = window_size
        self.gsl_alpha = gsl_alpha
        self.gsl_avg_steps = gsl_avg_steps
        self.tgcn_hidden_dim = tgcn_hidden_dim

        gcn_in_channels = window_size + tgcn_hidden_dim
        self.graph_structure_learner = GraphStructureLearner(
            gcn_in_channels, gsl_embed_dim
        )
        self.tgcn = TGCN_2(
            num_nodes, tgcn_hidden_dim
        )  # accepts at forward always a different graph adjacency and list

        self.classifier = nn.Linear(tgcn_hidden_dim * num_nodes, num_classes)

    def forward(self, x, static_edge_index, static_edge_weight=None):
        """
        Args:
            x (Tensor): Input time series. Shape: [B, T, N] (Batch, Time, Nodes)
                        Assumes node_input_feat_dim=1, so T is time steps.
            static_edge_index (LongTensor): Shared predefined graph connectivity. Shape: [2, E_static]
            static_edge_weight (Tensor, optional): Shared predefined graph weights. Shape: [E_static]
        Returns:
            out (Tensor): Classification logits. Shape: [B, num_classes]
        """
        x = x.transpose(-2, -1)
        if x.dim() == 3:  # Assume [B, N, T] -> [B, N, T, 1]
            x = x.unsqueeze(-1)
        batch_size, N, time_steps, F_in = x.shape
        assert N == self.num_nodes
        # assert F_in == self.node_input_feat_dim
        device = x.device
        H_dim = self.tgcn_hidden_dim

        # --- Initialize States ---
        # TGCN hidden state shape [B*N, H]. Initialize correctly.
        h_prev_flat = torch.zeros(batch_size, N * H_dim, device=device)

        E_history_deque = deque(maxlen=self.gsl_avg_steps)

        # --- Prepare input windows V_t ---
        # Use unfold to create sliding windows V_t = {x_{t-K+1}, ..., x_t}
        # Input x: [B, N, T, F_in]
        # We need to window along the time dimension (dim=2)
        # Permute to put Time dim first for unfold: [B, N, F_in, T]
        x_permuted = x.permute(0, 1, 3, 2)  # [B, N, F_in, T]
        # Unfold along the last dim (T): size=K, step=1 (for sliding window)
        # Output shape: [B, N, F_in, num_windows, K] where num_windows = T - K + 1
        v_t_unfolded = x_permuted.unfold(
            dimension=-1, size=self.window_size, step=self.window_size
        )

        # Permute and reshape V_t to be [B, num_windows, N, K*F_in] for easier iteration
        v_t = v_t_unfolded.permute(0, 3, 1, 4, 2).reshape(
            batch_size, -1, N, self.window_size * F_in
        )
        num_windows = v_t.shape[1]

        # --- Temporal Loop ---
        outputs = []
        for t in range(num_windows):
            # Get current window features V_t: [B, N, K*F_in]
            v_t_step = v_t[:, t, :, :]

            # Reshape previous hidden state for GSL input: [B*N, H] -> [B, N, H]
            h_prev = h_prev_flat.view(batch_size, N, H_dim)

            # Concatenate V_t and H_{t-1} to form I_t: [B, N, K*F_in + H]
            I_t = torch.cat([v_t_step, h_prev], dim=-1)

            # --- Graph Structure Learning ---
            # Using the simplified GSL that takes [B, N, F]
            # Pass I_t [B, N, K*F_in + H]
            data_list = []
            for i in range(batch_size):
                data_list.append(
                    Data(
                        x=I_t[i],
                        edge_index=static_edge_index,
                        edge_weight=static_edge_weight,
                    )
                )
            batch_graph = Batch.from_data_list(data_list)
            Et = self.graph_structure_learner(
                I_t, static_edge_index, static_edge_weight
            )  # Output Et: [B, N, N]

            E_history_deque.append(Et)

            # --- Average Dynamic Graph ---
            # Use the history buffer to compute Mt (averaged Et)
            if len(E_history_deque) > 0:
                # Shape becomes [current_deque_len, B, N, N]
                relevant_Et_stack = torch.stack(list(E_history_deque), dim=0)
                # Average along the time dimension (dim=0)
                Mt_batch = torch.mean(relevant_Et_stack, dim=0)  # [B, N, N]
            else:
                # Should not happen if loop runs at least once and n>=1
                # Handle gracefully just in case (e.g., use static graph or identity?)
                # Using Et directly if history is empty (only first step if n=1)
                Mt_batch = Et

            # --- Prepare Inputs for TGCN Cell ---
            # TGCN expects node features x_t: [B*N, F_tgcn_in]
            # F_tgcn_in = K * F_in (windowed features per node)

            data_list = []
            # Hidden state h_prev_flat is already [B*N, H]

            # --- Apply TGCN Cell ---
            # Uses x_t, h_{t-1}, and the *averaged* dynamic graph M_t
            h_t_flat = self.tgcn(
                v_t_step,
                h_prev_flat,
                Mt_batch,
            )  # Output h_t_flat: [B*N, H]

            # h_t_flat = self.tgcn(
            #     X=x_tgcn_in,
            #     H=h_prev_flat,
            #     edge_index=batched_edge_index,
            #     edge_weight=batched_edge_weight,
            # )  # Output h_t_flat: [B*N, H]

            # Update previous hidden state for the next loop iteration
            h_prev_flat = h_t_flat
            # Store the output hidden state (optional, e.g., for pooling later)
            # outputs.append(h_t_flat.view(batch_size, N, H_dim))

        # --- Final Classification ---
        # Use the *last* hidden state h_t_flat from the loop
        # Reshape from [B*N, H] to [B, N*H] for the linear classifier
        final_h_reshaped = h_prev_flat.view(batch_size, -1)  # [B, N*H]
        logits = self.classifier(final_h_reshaped)  # [B, num_classes]

        return logits


# ---- DCRNN -----

# code form https://github.com/tsy935/eeg-gnn-ssl/blob/main/model/model.py

class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        """
        Diffusion graph convolution
        Args:
            num_supports: number of supports, 1 for 'laplacian' filter and 2
                for 'dual_random_walk'
            input_dim: input feature dim
            hid_dim: hidden units
            num_nodes: number of nodes in graph
            max_diffusion_step: maximum diffusion step
            output_dim: output feature dim
            filter_type: 'laplacian' for undirected graph, and 'dual_random_walk'
                for directed graph
        """
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(
                size=(
                    self._input_size *
                    num_matrices,
                    output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))

        # Initialize with normally distributed parameters
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes,
        # input_dim/hidden_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # (batch, num_nodes, input_dim+hidden_dim)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x0 = inputs_and_state  # (batch, num_nodes, input_dim+hidden_dim)
        # (batch, 1, num_nodes, input_dim+hidden_dim)
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                # (batch, num_nodes, input_dim+hidden_dim)
                x1 = torch.matmul(support, x0)
                # (batch, _, num_nodes, input_dim+hidden_dim)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    # (batch, num_nodes, input_dim+hidden_dim)
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(
                        x, x2)  # (batch, _, num_nodes, input_dim+hidden_dim)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * \
            self._max_diffusion_step + 1  # Adds for x itself
        # (batch, num_nodes, num_matrices, input_hidden_size)
        x = torch.transpose(x, dim0=1, dim1=2)
        # (batch, num_nodes, input_hidden_size, num_matrices)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.reshape(
            x,
            shape=[
                batch_size *
                self._num_nodes,
                input_size *
                num_matrices])
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
    

class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
            self,
            input_dim,
            num_units,
            max_diffusion_step,
            num_nodes,
            filter_type="laplacian",
            nonlinearity='tanh',
            use_gc_for_ru=True):
        """
        Args:
            input_dim: input feature dim
            num_units: number of DCGRU hidden units
            max_diffusion_step: maximum diffusion step
            num_nodes: number of nodes in the graph
            filter_type: 'laplacian' for undirected graph, 'dual_random_walk' for directed graph
            nonlinearity: 'tanh' or 'relu'. Default is 'tanh'
            use_gc_for_ru: decide whether to use graph convolution inside rnn. Default True
        """
        super(DCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        if filter_type == "laplacian":  # ChebNet graph conv
            self._num_supports = 1
        elif filter_type == "random_walk":  # Forward random walk
            self._num_supports = 1
        elif filter_type == "dual_random_walk":  # Bidirectional random walk
            self._num_supports = 2
        else:
            self._num_supports = 1

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def forward(self, supports, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state: (B, num_nodes * num_units)
        Returns:
            output: (B, num_nodes * output_dim)
            state: (B, num_nodes * num_units)
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(supports, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(
                output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        # batch_size, self._num_nodes * output_size
        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)



class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)

def last_relevant_pytorch(output, lengths, batch_first=True):
    # Get the last output in 'output'.
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output

class DCRNNModel_classification(nn.Module):
    def __init__(self, num_nodes, num_rnn_layers, rnn_units, input_dim, num_classes, max_diffusion_step, dcgru_activation, filter_type, dropout, device=None):
        super(DCRNNModel_classification, self).__init__()

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(input_dim=input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=dcgru_activation,
                                    filter_type=filter_type)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(
            batch_size).to(self._device) # (num_layers, batch, num_nodes * rnn_units)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        _, final_hidden = self.encoder(input_seq, init_hidden_state, supports)
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # extract last relevant output
        last_out = last_relevant_pytorch(
            output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        last_out = last_out.to(self._device)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits



# ---- GNN_LSTM -----

from torch_geometric.utils import dense_to_sparse

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