from torch_geometric.nn import GCNConv
from torch import nn
import torch
from torch_geometric.utils.sparse import dense_to_sparse
from collections import deque
from torch_geometric.data import Batch, Data

# from tgcn import TGCN
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize


def calculate_laplacian_with_self_loop(matrix):
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


class TGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
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
        laplacian = calculate_laplacian_with_self_loop(adj)

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


class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
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
    

class TGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.tgcn_cell = TGCNCell(self._input_dim, self._hidden_dim)

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
        self.tgcn = TGCN(
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