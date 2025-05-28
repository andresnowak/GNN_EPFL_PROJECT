from torch_geometric.nn import GCNConv
from torch import nn
import torch
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Batch, Data
from collections import deque


class TGCNWrapper(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim 

        self.tgcn_1 = TGCN(input_dim, hidden_dim)
        self.tgcn_2 = TGCN(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)


    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x (torch.Tensor): shape: (batch_size, time_steps, num_eeg_channels) e.g., (B, 500, 21)
        """

        B, T, N = x.shape
        outputs = []

        # Pre-allocate tensors for hidden states
        H_1 = None          # hidden state for layer1
        H_2 = None          # hidden state for layer2

        for t in range(T):
            x_t = x[:, t, :].unsqueeze(-1)  # [batch, num_nodes, 1]

            data_list = []
            for i in range(B):
                data_list.append(
                    Data(x=x_t[i], edge_index=edge_index, edge_weight=edge_weight)
                )
            batch_graph = Batch.from_data_list(data_list)
            
            # Update hidden states
            H_1 = self.tgcn_1(batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight, H_1)
            H_2 = self.tgcn_2(H_1, batch_graph.edge_index, batch_graph.edge_weight, H_2)

            out = H_2.view(B, N, -1)
            outputs.append(out)

        out_seq = torch.stack(outputs, dim=1) # [B, T, N, H]
        graph_out = out_seq.mean(dim=2) # [B, T, H]
        seq_out   = graph_out.mean(dim=1).squeeze() # [B, H]
        logits    = self.classifier(seq_out) # [B, num_classes]

        return logits