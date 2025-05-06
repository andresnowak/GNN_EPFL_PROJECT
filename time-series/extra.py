from torch_geometric.nn import GCNConv, TGNMemory
from torch import nn
import torch

# The TGCN cell
class GraphGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gcn_x = GCNConv(in_channels, hidden_channels)
        self.gcn_uh = GCNConv(hidden_channels * 2, hidden_channels)
        self.gcn_ch = GCNConv(hidden_channels * 2, hidden_channels)

    def forward(self, x, edge_index, h_prev):
        x_gcn = self.gcn_x(x, edge_index)  # θ_G(I_t)
        combined = torch.cat([x_gcn, h_prev], dim=1)

        u = torch.sigmoid(self.gcn_uh(combined, edge_index))  # Update gate
        r = torch.sigmoid(
            self.gcn_uh(combined, edge_index)
        )  # Reset gate (you may define separate parameters)
        r_h = r * h_prev
        combined_r = torch.cat([x_gcn, r_h], dim=1)
        c = torch.tanh(self.gcn_ch(combined_r, edge_index))  # Candidate hidden state

        h = u * h_prev + (1 - u) * c
        return h


class SeizureAttentionModule(nn.Module):
    def __init__(self, num_nodes: int, num_heads: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.head_dim = num_nodes // num_heads

        assert self.head_dim * num_heads == num_nodes, (
            "num_nodes must be divisible by num_heads"
        )

        self.W_Q = nn.Linear(num_nodes, num_nodes)
        self.W_K = nn.Linear(num_nodes, num_nodes)
        self.W_V = nn.Linear(num_nodes, num_nodes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        # X shape: [batch_size, num_nodes (N), time_steps (T)]
        X_permuted = X.permute(0, 2, 1)  # [B, T, N]

        # Project to Q, K, V
        Q = self.W_Q(X_permuted)  # [B, T, N]
        K = self.W_K(X_permuted)
        V = self.W_V(X_permuted)

        # Split into heads
        batch_size, T, _ = Q.size()
        Q = Q.view(batch_size, T, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, T, head_dim]
        K = K.view(batch_size, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        S = self.softmax(scores)

        # Apply attention to V
        X_prime = torch.matmul(S, V)  # [B, num_heads, T, head_dim]
        X_prime = (
            X_prime.transpose(1, 2).contiguous().view(batch_size, T, self.num_nodes)
        )
        X_prime = X_prime.permute(0, 2, 1)  # [B, N, T]

        return X_prime