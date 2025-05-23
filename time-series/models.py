from torch_geometric.nn import GCNConv
from torch import nn
import torch
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric.utils.sparse import dense_to_sparse
from collections import deque

# from tgcn import TGCN


class GraphStructureLearner(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        embed_dim: int,
        alpha: float = 1.0,
        device: str = "cpu",
    ):
        super(GraphStructureLearner, self).__init__()

        self.alpha = alpha

        # the amount of in_channels is the the size of the node v features
        self.gconv = GCNConv(in_channels=node_feature_dim, out_channels=embed_dim).to(
            device
        )

        # calculate the dynamic features of src and dst nodes (allows the model to learn directed or asymmetric relationships between node)
        self.linear_de1 = nn.Linear(embed_dim, embed_dim).to(device)
        self.linear_de2 = nn.Linear(embed_dim, embed_dim).to(device)

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
        num_classes: int,
        device: str = "cpu",
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
            gcn_in_channels, gsl_embed_dim, device=device
        )
        self.tgcn = TGCN(window_size, tgcn_hidden_dim).to(
            device
        )  # accepts at forward always a different graph adjacency and list

        self.classifier = nn.Linear(tgcn_hidden_dim * num_nodes, num_classes).to(device)

    def forward(self, x, static_edge_index, static_edge_weight=None):
        """
        Args:
            x (Tensor): Input time series. Shape: [B, N, T] (Batch, Nodes, Time)
                        Assumes node_input_feat_dim=1, so T is time steps.
                        If node_input_feat_dim > 1, shape is [B, N, T, F_in]
            static_edge_index (LongTensor): Shared predefined graph connectivity. Shape: [2, E_static]
            static_edge_weight (Tensor, optional): Shared predefined graph weights. Shape: [E_static]
        Returns:
            out (Tensor): Classification logits. Shape: [B, num_classes]
        """
        if x.dim() == 3:  # Assume [B, N, T] -> [B, N, T, 1]
            x = x.unsqueeze(-1)
        batch_size, N, time_steps, F_in = x.shape
        assert N == self.num_nodes
        # assert F_in == self.node_input_feat_dim
        device = x.device
        H_dim = self.tgcn_hidden_dim

        # --- Initialize States ---
        # TGCN hidden state shape [B*N, H]. Initialize correctly.
        h_prev_flat = torch.zeros(batch_size * N, H_dim, device=device)

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

            # --- Convert Averaged Graph to Batched Sparse Format ---
            batched_edge_index, batched_edge_weight = batched_adj_to_edge_list(Mt_batch)

            # --- Prepare Inputs for TGCN Cell ---
            # TGCN expects node features x_t: [B*N, F_tgcn_in]
            # F_tgcn_in = K * F_in (windowed features per node)
            x_tgcn_in = v_t_step.reshape(batch_size * N, -1)
            # Hidden state h_prev_flat is already [B*N, H]

            # --- Apply TGCN Cell ---
            # Uses x_t, h_{t-1}, and the *averaged* dynamic graph M_t
            h_t_flat = self.tgcn(
                X=x_tgcn_in,
                H=h_prev_flat,
                edge_index=batched_edge_index,
                edge_weight=batched_edge_weight,
            )  # Output h_t_flat: [B*N, H]

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
    


def batched_adj_to_edge_list(adj, threshold=1e-8):
    """
    Converts a batched adjacency matrix to a batched edge list and edge weights.

    Args:
        adj (Tensor): Batched adjacency matrix of shape (batch_size, num_nodes, num_nodes).
        threshold (float): Threshold value to filter edges.

    Returns:
        edge_index (LongTensor): Edge indices in COO format with shape (2, num_edges).
        edge_weights (Tensor): Edge weights with shape (num_edges,).
    """
    # Apply threshold to create a mask for edges to keep
    mask = adj > threshold

    # Get indices of non-zero elements (edges) in the batched adjacency matrix
    batch_indices, rows, cols = mask.nonzero(as_tuple=True)

    # Extract the corresponding edge weights (gradients preserved here)
    edge_weights = adj[batch_indices, rows, cols]

    # Calculate global node indices accounting for the batch
    num_nodes = adj.size(1)
    offset = batch_indices * num_nodes
    global_rows = rows + offset
    global_cols = cols + offset

    # Stack to form the edge_index in COO format
    edge_index = torch.stack([global_rows, global_cols], dim=0)

    return edge_index, edge_weights
