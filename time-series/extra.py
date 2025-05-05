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
