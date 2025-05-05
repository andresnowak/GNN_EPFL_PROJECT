from torch_geometric.nn import GCNConv, TGNMemory
from torch import nn
import torch
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric.utils.sparse import dense_to_sparse

class GraphStructureLearner(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        embed_dim: int,
        alpha: float=1.0,
        device: str="cpu",
    ):
        super(GraphStructureLearner, self).__init__()

        self.alpha = alpha

        # the amount of in_channels is the the size of the node v features
        self.gconv = GCNConv(in_channels=node_feature_dim, out_channels=embed_dim).to(device)

        # calculate the dynamic features of src and dst nodes (allows the model to learn directed or asymmetric relationships between node)
        self.linear_de1 = nn.Linear(embed_dim, embed_dim).to(device)
        self.linear_de2 = nn.Linear(embed_dim, embed_dim).to(device)
    
    def forward(self, v_h, edge_list, edge_weights):
        # We want batch_size, the number of nodes and then the features for gconv

        df = self.gconv(v_h, edge_list, edge_weights)

        de1 = torch.tanh(self.alpha * self.linear_de1(df))
        de2 = torch.tanh(self.alpha * self.linear_de2(df))

        # Et = RELU(tanh(alpha * (DE1 * DE2^T - DE2 * DE1^T))) - Simplified interpretation
        Et = torch.relu(torch.tanh(self.alpha * (de1 @ de2.transpose(-2, -1) - de2 @ de1.transpose(-2, -1))))

        # mt = Et.mean(dim = 1) # the adjacency matrix at time t will be the average of the windows (that we did on the 12 second window coarse grained lable)

        return Et
    

class DTGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_input_feat_dim: int,  # Features per node at each time step
        gsl_node_feat_dim: int,  # Features per node FOR THE GSL WINDOW
        gsl_embed_dim: int,
        tgcn_hidden_dim: int, # The H size
        window_size: int, # K window size
        gsl_alpha: float,
        gsl_avg_steps: int,
        num_classes: int,
        device: str = "cpu",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_input_feat_dim = node_input_feat_dim
        self.gsl_node_feat_dim = gsl_node_feat_dim
        self.gsl_embed_dim = gsl_embed_dim
        self.window_size = window_size
        self.gsl_alpha = gsl_alpha
        self.gsl_avg_steps = gsl_avg_steps
        self.tgcn_hidden_dim = tgcn_hidden_dim

        gcn_in_channels = window_size + tgcn_hidden_dim
        self.graph_structure_learner = GraphStructureLearner(gcn_in_channels,gsl_embed_dim, device=device)
        self.tgcn = TGCN(1, tgcn_hidden_dim).to(device) # accepts at forward always a different graph adjacency and list

        self.classifier = nn.Linear(tgcn_hidden_dim * num_nodes, num_classes).to(device)

    def _init_gsl_history(self, batch_size, device):
        """Initializes the history state for Et averaging."""
        N = self.num_nodes
        return {
            "E_history": torch.zeros(
                batch_size, self.gsl_avg_steps, N, N, device=device
            ),
            "history_idx": torch.zeros(batch_size, dtype=torch.long, device=device),
            "history_full": torch.zeros(batch_size, dtype=torch.bool, device=device),
        }
    
    def _update_history_and_average(self, Et_batch, history_state):
        """Updates history buffer with Et_batch and computes Mt_batch."""
        batch_size = Et_batch.shape[0]
        device = Et_batch.device
        avg_steps = self.gsl_avg_steps

        # Retrieve current history state
        E_history = history_state['E_history']
        history_idx = history_state['history_idx']
        history_full = history_state['history_full']

        mt_batch = torch.zeros_like(Et_batch) # To store results [B, N, N]
        next_history_idx = torch.zeros_like(history_idx)
        next_history_full = history_full.clone()

        for i in range(batch_size):
            current_idx_i = history_idx[i].item()
            E_history[i, current_idx_i] = Et_batch[i] # Store current Et
            next_idx_i = (current_idx_i + 1) % avg_steps
            next_history_idx[i] = next_idx_i
            if not next_history_full[i] and next_idx_i == 0:
                next_history_full[i] = True
            # Calculate average Mt for item i
            if next_history_full[i]:
                mt_batch[i] = torch.mean(E_history[i], dim=0)
            else:
                mt_batch[i] = torch.mean(E_history[i, :current_idx_i+1], dim=0)

        new_history_state = {
            'E_history': E_history,
            'history_idx': next_history_idx.to(device),
            'history_full': next_history_full.to(device)
        }
        return mt_batch, new_history_state

    def forward(self, x, static_edge_index, static_edge_weight=None):
        """
        Args:
            x (Tensor): Input time series. Shape: [batch, num_nodes, time_steps]
            static_edge_index (LongTensor): Shared predefined graph connectivity. Shape: [2, E_static]
            static_edge_weight (Tensor, optional): Shared predefined graph weights. Shape: [E_static]
        Returns:
            out (Tensor): Classification logits. Shape: [batch, num_classes]
        """
        batch_size, N_main, time_steps = x.shape
        device = x.device
        N_gsl = self.num_nodes
        H_dim = self.tgcn_hidden_dim

        # --- Initialize States ---
        gsl_history_state = self._init_gsl_history(batch_size, device)
        # TGCN hidden state shape usually [B*N, H]. Initialize correctly.
        h_prev = torch.zeros(batch_size * self.num_nodes, H_dim, device=device)

        # --- Temporal Processing ---
        for t in range(time_steps):
            # --- Prepare Inputs for GSL ---
            # 1a. Get Window Vt -> features gsl_node_features [B, N, F_gsl]
            start_idx = max(0, t - self.window_size + 1)
            window = x[:, :, start_idx : t + 1]  # [B, win_len, N, F_in]
            win_len_actual = window.shape[2]
            # Assuming N_gsl == N_main
            if N_main != N_gsl:
                raise NotImplementedError("GSL nodes != main nodes")
            # Prepare features [B, N, F_gsl=K] (assuming F_in=1)
    
            # gsl_node_features_padded = window.permute(0, 2, 1, 3)  # [B, N, W, 1]
            if win_len_actual < self.window_size:
                pad_shape = (
                    batch_size,
                    N_main,
                    self.window_size - win_len_actual,
                )
                padding = torch.zeros(pad_shape, device=device)
                window = torch.cat(
                    [padding, window], dim=2
                )

        
            # Reshape based on definition F_gsl = K * F_in
            gsl_node_features = window.reshape(
                batch_size, N_main, -1
            )  # [B, N, F_gsl]

            # 1b. Get Previous Hidden State Ht-1, reshape for concat: [B*N, H] -> [B, N, H]
            h_prev_reshaped = h_prev.reshape(batch_size, N_main, H_dim)

            # 1c. Concatenate features It = Vt || Ht-1
            It_batch = torch.cat(
                [gsl_node_features, h_prev_reshaped], dim=2
            )  # [B, N, F_gsl + H]


            # --- GSL Step ---
            # 2. Get Instantaneous Graph Et_batch
            Et_batch = self.graph_structure_learner(
                It_batch, static_edge_index, static_edge_weight
            )  # Shape: [B, N_gsl, N_gsl]

            # --- Averaging Step ---
            # 3. Update History & Compute Average Mt_batch
            Mt_batch, gsl_history_state = self._update_history_and_average(
                Et_batch, gsl_history_state
            )
            # Shape: [B, N_gsl, N_gsl]

            # --- Prepare Inputs for TGCN ---
            # 4. Convert Mt_batch to Sparse PyG format for TGCN
            dynamic_edge_index, dynamic_edge_weight = dense_to_sparse(
                Mt_batch
            )

            # 5. Get Current Input x_t (Batched) -> [B*N, F_in]
            # x_t shape: [B, N] (signal value at time t for each node)
            x_t = x[:, :, t]
            # Reshape x_t for TGCN: [B, N] -> [B*N, 1] (feature dim is 1)
            x_t_flat = x_t.reshape(-1, 1)

            # --- TGCN Step ---
            # 6. Update TGCN State using Mt (via sparse graph)

            h_next = self.tgcn(
                x_t_flat, dynamic_edge_index, dynamic_edge_weight, H=h_prev
            )
            h_prev = h_next  # Update hidden state

        # --- Classification ---
        # 7. Classify based on final state
        final_state_flat = h_prev.reshape(batch_size, -1)  # [B, N*H]
        out = self.classifier(final_state_flat)

        return out  # Shape: [batch, num_classes]