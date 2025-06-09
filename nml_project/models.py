import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GraphNorm, global_mean_pool, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import AddLaplacianEigenvectorPE

import sys
sys.path.append("/path/to/spaces")
from models.s4.s4 import S4Block


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