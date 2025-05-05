import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Modules ---


class GCNLayer(nn.Module):
    """Basic Graph Convolution Layer"""

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        # x shape: (Batch, NumNodes, InFeatures) or (NumNodes, InFeatures) if no batch
        # adj shape: (Batch, NumNodes, NumNodes) or (NumNodes, NumNodes)
        support = self.linear(x)  # Batch, NumNodes, OutFeatures
        # Basic GCN: A * X * W
        # Ensure dimensions work for batch matrix multiplication
        if support.dim() == 3 and adj.dim() == 3:
            output = torch.bmm(adj, support)  # Batch, NumNodes, OutFeatures
        elif support.dim() == 2 and adj.dim() == 2:
            output = torch.mm(adj, support)  # NumNodes, OutFeatures
        else:
            raise ValueError("Input dimensions for GCN not standard")
        return output


# --- Core Components from Paper ---


class SeizureAttention(nn.Module):
    """
    Implements the Multi-Head Attention for reconstruction (Section 3.3)
    and calculates the constrained reconstruction loss (Lre).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SeizureAttention, self).__init__()
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Assuming input features (M channels) = embed_dim for simplicity
        # If not, add Linear layers for Q, K, V projections here or ensure input matches

    def forward(self, x):
        # x shape: (Batch, SequenceLength N, Features M (embed_dim))
        # In self-attention Q, K, V are the same
        attn_output, _ = self.multihead_attn(x, x, x)
        # attn_output is the reconstructed signal X'
        return attn_output  # Shape: (Batch, N, M)

    def calculate_reconstruction_loss(self, x, x_prime, fine_grained_labels, tau):
        # x, x_prime shape: (Batch, N, M)
        # fine_grained_labels: (Batch, N), contains 0 or 1 for each time step
        # tau: pre-determined threshold (scalar)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # 1. Standard Reconstruction Loss (L2 norm per batch item)
        l2_diff_per_step = torch.linalg.norm(x - x_prime, dim=2)  # Shape: (Batch, N)
        reconstruction_loss_mse = F.mse_loss(x_prime, x, reduction="none").mean(
            dim=[1, 2]
        )  # Shape: (Batch,)

        # 2. Constraint based on fine-grained labels
        loss_constraint = torch.zeros(batch_size, device=x.device)

        for i in range(batch_size):
            # T1: Ground truth seizure steps
            t1_indices = torch.where(fine_grained_labels[i] == 1)[0]
            crad_t1 = len(t1_indices)

            # T2: Model-predicted seizure steps based on reconstruction error > tau
            t2_indices = torch.where(l2_diff_per_step[i] > tau)[0]
            crad_t2 = len(t2_indices)

            # Avoid division by zero if seq_len (n in paper Eq 5) is 0, though unlikely
            if seq_len > 0:
                # Difference in counts (cardinality)
                # Paper uses (crad(T1) - crad(T2)) / n
                # Using abs() might be more stable than direct difference for loss
                # loss_constraint[i] = (crad_t1 - crad_t2) / seq_len
                loss_constraint[i] = (
                    torch.abs(torch.tensor(crad_t1 - crad_t2, dtype=torch.float32))
                    / seq_len
                )

        # Combine: Lre = ||X – X'||2 + (crad(T1) – crad(T2)) / n (per batch item)
        # Using MSE for ||X-X'||2 part averaged over batch
        total_lre_loss = reconstruction_loss_mse + loss_constraint

        return total_lre_loss.mean()  # Average Lre over the batch


class GraphStructureLearner(nn.Module):
    """
    Learns the dynamic graph structure (Section 3.4).
    """

    def __init__(
        self, num_nodes, embed_dim, gru_hidden_dim, alpha=1.0, averaging_steps=4
    ):
        super(GraphStructureLearner, self).__init__()
        self.num_nodes = num_nodes  # M
        self.embed_dim = embed_dim  # Dimension of features per node in window K
        self.gru_hidden_dim = gru_hidden_dim
        self.averaging_steps = averaging_steps
        self.alpha = alpha  # Saturation control

        # Input to Gconv is It = Vt || Ht-1 (potentially)
        # Vt shape: (Batch, K, M), Ht-1 shape: (Batch, M, gru_hidden_dim) ? Need clarification
        # Let's assume It represents features per node: (Batch, M, FeatureDim)
        # FeatureDim could be K + gru_hidden_dim if concatenating flattened window + hidden state per node

        # Simplified: Assume input feature dim reflects window info + hidden state passed appropriately
        gconv_input_dim = embed_dim  # Example: Needs adjustment based on It definition
        self.gconv = GCNLayer(gconv_input_dim, embed_dim)  # Output dim = embed_dim

        # Learnable node embeddings (E_t in paper, used implicitly here)
        # Alternatively, use a predefined graph adjacency matrix
        self.predefined_adj = nn.Parameter(
            torch.randn(num_nodes, num_nodes), requires_grad=True
        )  # Example learnable graph

        self.linear_de1 = nn.Linear(embed_dim, embed_dim)
        self.linear_de2 = nn.Linear(embed_dim, embed_dim)

        self.dynamic_adj_buffer = []  # Store last 'n' Et matrices

    def forward(self, node_features_it):
        # node_features_it: Features per node at time t, shape (Batch, M, FeatureDim)
        # Assume gconv_input_dim matches FeatureDim

        batch_size = node_features_it.shape[0]

        # 1. Graph Conv on predefined/learnable graph (Eq 8, simplified)
        # Use self.predefined_adj, ensure it's normalized/processed if needed (e.g., softmax)
        predefined_graph_processed = F.softmax(
            self.predefined_adj, dim=1
        )  # Example processing
        if node_features_it.dim() == 3:
            predefined_graph_processed = predefined_graph_processed.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        df = self.gconv(
            node_features_it, predefined_graph_processed
        )  # (Batch, M, embed_dim)

        # 2. Calculate Dynamic Features DE1, DE2 (Eq 9, 10)
        # Note: Paper uses DFt for both DE1 and DE2 source/target? Assume typo, use DF for both.
        de1 = torch.tanh(self.alpha * self.linear_de1(df))  # (Batch, M, embed_dim)
        de2 = torch.tanh(self.alpha * self.linear_de2(df))  # (Batch, M, embed_dim)

        # 3. Calculate Dynamic Adjacency Et (Eq 11)
        # Et = RELU(tanh(alpha * (DE1 * DE2^T - DE2 * DE1^T))) - Simplified interpretation
        # Let's use dot product similarity between node pairs
        de1_t = de1.transpose(1, 2)  # (Batch, embed_dim, M)
        et_raw = torch.bmm(de1, de1_t)  # Similarity matrix (Batch, M, M)

        # Apply activation: RELU(tanh(alpha * raw_similarity))
        et = F.relu(torch.tanh(self.alpha * et_raw))  # (Batch, M, M)

        # Ensure sparsity (e.g., keep top-k edges per node, optional based on paper details)
        # TODO: Implement top-k or other sparsity methods if needed

        # 4. Average over time (Eq 12)
        if len(self.dynamic_adj_buffer) >= self.averaging_steps:
            self.dynamic_adj_buffer.pop(0)
        self.dynamic_adj_buffer.append(et)

        if not self.dynamic_adj_buffer:  # Should not happen after first step
            avg_adj = et
        else:
            avg_adj = torch.mean(
                torch.stack(self.dynamic_adj_buffer, dim=0), dim=0
            )  # Average over buffer

        return avg_adj  # M^t


class TGCNCell(nn.Module):
    """Temporal Graph Convolutional Network Cell (Combines GCN with GRU gates)"""

    def __init__(self, node_feature_dim, hidden_dim):
        super(TGCNCell, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim

        # GCN part within GRU gates (theta_prime in paper)
        self.gcn_update_gate = GCNLayer(node_feature_dim + hidden_dim, hidden_dim)
        self.gcn_reset_gate = GCNLayer(node_feature_dim + hidden_dim, hidden_dim)
        self.gcn_candidate = GCNLayer(node_feature_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, adj):
        # x_t: Node features at time t, shape (Batch, NumNodes M, FeatureDim)
        # h_prev: Previous hidden state, shape (Batch, NumNodes M, HiddenDim)
        # adj: Dynamic graph adjacency M^t, shape (Batch, NumNodes M, NumNodes M)

        combined_input = torch.cat(
            [x_t, h_prev], dim=2
        )  # (Batch, M, FeatureDim + HiddenDim)

        # Calculate gates using GCN
        update_gate = torch.sigmoid(
            self.gcn_update_gate(combined_input, adj)
        )  # ut (Eq 13)
        reset_gate = torch.sigmoid(
            self.gcn_reset_gate(combined_input, adj)
        )  # rt (Eq 14)

        # Calculate candidate hidden state
        candidate_input = torch.cat([x_t, reset_gate * h_prev], dim=2)
        candidate = torch.tanh(self.gcn_candidate(candidate_input, adj))  # ct (Eq 15)

        # Update hidden state (Eq 16)
        h_next = (1.0 - update_gate) * h_prev + update_gate * candidate

        return h_next


# --- Main Model ---


class DTGCN(nn.Module):
    """Dynamic Temporal Graph Convolutional Network"""

    def __init__(
        self,
        num_nodes,
        input_feature_dim,
        hidden_dim,
        num_classes,
        num_attn_heads=8,
        window_size_k=10,
        graph_avg_steps_n=4,
        lambda_lre=0.1,
        dropout=0.1,
        use_fine_grained=True,
    ):
        super(DTGCN, self).__init__()

        self.num_nodes = num_nodes  # M
        self.input_feature_dim = (
            input_feature_dim  # Should match M if raw EEG channels used directly
        )
        self.hidden_dim = hidden_dim
        self.window_size_k = window_size_k  # K for Graph Learner input window
        self.use_fine_grained = use_fine_grained
        self.lambda_lre = lambda_lre

        if self.use_fine_grained:
            # Embed dim for attention should match input features if no projection used
            self.seizure_attention = SeizureAttention(
                input_feature_dim, num_attn_heads, dropout
            )
            # Input to TGCNCell comes from reconstructed signal X'
            tgcn_node_feature_dim = input_feature_dim
        else:
            self.seizure_attention = None
            # Input to TGCNCell comes from original signal X
            tgcn_node_feature_dim = input_feature_dim

        # Graph Structure Learner needs features per node - careful about dimensions
        # Simplified: Assume TGCN cell handles feature extraction internally for now
        # TODO: Refine input feature definition for GraphStructureLearner
        graph_learner_embed_dim = (
            hidden_dim  # Example dimension for graph learner internal features
        )
        self.graph_learner = GraphStructureLearner(
            num_nodes,
            graph_learner_embed_dim,
            hidden_dim,
            averaging_steps=graph_avg_steps_n,
        )

        self.tgcn_cell = TGCNCell(tgcn_node_feature_dim, hidden_dim)

        # Output layers
        # Use final hidden state pooled over nodes or flattened
        self.fc_detection = nn.Linear(
            num_nodes * hidden_dim, 1
        )  # Binary seizure detection
        self.fc_classification = nn.Linear(
            num_nodes * hidden_dim, num_classes
        )  # Multi-class classification

    def forward(self, x, fine_grained_labels=None, tau=None):
        # x: Input EEG, shape (Batch, SequenceLength N, NumNodes M)
        # fine_grained_labels: (Batch, N), required if use_fine_grained=True
        # tau: Threshold for Lre calculation, required if use_fine_grained=True

        batch_size, seq_len, num_nodes = x.shape
        lre_loss = torch.tensor(0.0, device=x.device)  # Initialize loss

        # 1. Seizure Attention (Optional)
        if self.use_fine_grained:
            if fine_grained_labels is None or tau is None:
                raise ValueError(
                    "fine_grained_labels and tau required when use_fine_grained is True"
                )
            # Assume input_feature_dim = num_nodes M for attention
            x_prime = self.seizure_attention(x)
            lre_loss = self.seizure_attention.calculate_reconstruction_loss(
                x, x_prime, fine_grained_labels, tau
            )
            temporal_input = x_prime  # Use reconstructed signal for TGCN
        else:
            temporal_input = x  # Use original signal for TGCN

        # 2. Temporal Dynamics (TGCN)
        h_t = torch.zeros(
            batch_size, num_nodes, self.hidden_dim, device=x.device
        )  # Initial hidden state
        self.graph_learner.dynamic_adj_buffer = []  # Reset buffer for each forward pass

        # Iterate through time steps for TGCN
        # Need to define how inputs It and node features for graph learner are derived
        # Simplified: Use hidden state for graph learner input and raw step for TGCN input
        for t in range(seq_len):
            # Prepare input for Graph Learner
            # TODO: Define how features for Graph Learner (node_features_it) are derived
            # Using h_t as a placeholder for features passed to graph learner
            graph_learner_input_features = (
                h_t  # Shape (Batch, M, HiddenDim) - Adjust if needed
            )

            # Get dynamic graph M^t
            dynamic_adj_mt = self.graph_learner(graph_learner_input_features)

            # Prepare input for TGCN Cell
            x_t_step = temporal_input[:, t, :].unsqueeze(
                2
            )  # (Batch, M, 1) -> need FeatureDim match
            # Assuming input feature dim = 1 per node per step? Or use original M dim?
            # Let's assume tgcn_node_feature_dim = M (input_feature_dim)
            x_t_step_features = temporal_input[:, t, :]  # Shape (Batch, M)
            # Need to match TGCNCell input (Batch, M, FeatureDim)
            # Example: treat channels as features -> (Batch, M, 1) - needs adjustment
            # Correct approach: TGCN likely operates on node embeddings or processed features.
            # Placeholder: Assuming TGCNCell input dim matches hidden_dim for simplicity now
            # This part needs careful matching with TGCNCell's node_feature_dim definition
            tgcn_input_features = (
                h_t  # Using hidden state as input feature proxy - NEEDS REFINEMENT
            )

            # Run TGCN Cell
            h_t = self.tgcn_cell(tgcn_input_features, h_t, dynamic_adj_mt)

        # 3. Final Classification
        # Use the last hidden state h_t
        # Pool or flatten the node dimension
        final_features = h_t.reshape(
            batch_size, -1
        )  # Flatten: (Batch, NumNodes * HiddenDim)

        detection_logits = self.fc_detection(final_features)  # (Batch, 1)
        classification_logits = self.fc_classification(
            final_features
        )  # (Batch, NumClasses)

        # Apply sigmoid for detection probability if needed outside the model
        # Apply softmax/logsoftmax for classification probability if needed outside

        return detection_logits, classification_logits, lre_loss


# --- Example Usage ---

if __name__ == "__main__":
    # Hyperparameters (Example Values)
    BATCH_SIZE = 16
    SEQ_LEN_N = 100  # Example sequence length (e.g., 0.5 seconds at 200Hz)
    NUM_NODES_M = 24  # Number of EEG channels
    INPUT_FEAT_DIM = NUM_NODES_M  # Assuming input features are the channels directly
    HIDDEN_DIM = 64
    NUM_CLASSES = 4  # E.g., CF, GN, AB, CT
    NUM_ATTN_HEADS = 4
    WINDOW_K = 10
    AVG_N = 4
    LAMBDA = 0.2  # Weight for Lre loss
    TAU = 0.5  # Example Threshold (NEEDS TO BE DETERMINED from validation set)

    # --- Fine-Grained Example ---
    print("--- Testing Fine-Grained Mode ---")
    model_fine = DTGCN(
        num_nodes=NUM_NODES_M,
        input_feature_dim=INPUT_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_attn_heads=NUM_ATTN_HEADS,
        window_size_k=WINDOW_K,
        graph_avg_steps_n=AVG_N,
        lambda_lre=LAMBDA,
        use_fine_grained=True,
    )

    # Dummy Input Data
    dummy_eeg = torch.randn(BATCH_SIZE, SEQ_LEN_N, NUM_NODES_M)
    dummy_fine_labels = (
        torch.rand(BATCH_SIZE, SEQ_LEN_N) > 0.8
    ).float()  # Random 0/1 labels
    dummy_coarse_labels_detect = (
        (torch.rand(BATCH_SIZE) > 0.5).float().unsqueeze(1)
    )  # Binary detection target
    dummy_coarse_labels_class = torch.randint(
        0, NUM_CLASSES, (BATCH_SIZE,)
    )  # Classification target

    # Forward Pass (Fine-Grained)
    detect_logits_fine, class_logits_fine, lre_loss_fine = model_fine(
        dummy_eeg, dummy_fine_labels, TAU
    )

    # Calculate Loss (Fine-Grained)
    criterion_detect = nn.BCEWithLogitsLoss()  # For binary detection
    criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

    loss_detect_fine = criterion_detect(detect_logits_fine, dummy_coarse_labels_detect)
    loss_class_fine = criterion_class(class_logits_fine, dummy_coarse_labels_class)

    total_loss_fine = (
        loss_detect_fine + loss_class_fine + model_fine.lambda_lre * lre_loss_fine
    )

    print(f"Fine-Grained Mode:")
    print(f"  Detection Logits shape: {detect_logits_fine.shape}")
    print(f"  Classification Logits shape: {class_logits_fine.shape}")
    print(f"  Lre Loss: {lre_loss_fine.item():.4f}")
    print(f"  Detection Loss: {loss_detect_fine.item():.4f}")
    print(f"  Classification Loss: {loss_class_fine.item():.4f}")
    print(f"  Total Loss: {total_loss_fine.item():.4f}")

    # --- Coarse-Grained Example ---
    print("\n--- Testing Coarse-Grained Mode ---")
    model_coarse = DTGCN(
        num_nodes=NUM_NODES_M,
        input_feature_dim=INPUT_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_attn_heads=NUM_ATTN_HEADS,  # Not used, but param needed
        window_size_k=WINDOW_K,
        graph_avg_steps_n=AVG_N,
        lambda_lre=LAMBDA,  # Not used, but param needed
        use_fine_grained=False,  # KEY DIFFERENCE
    )

    # Forward Pass (Coarse-Grained) - No fine labels or tau needed
    detect_logits_coarse, class_logits_coarse, lre_loss_coarse = model_coarse(dummy_eeg)

    # Calculate Loss (Coarse-Grained) - Lre is ignored (should be 0)
    loss_detect_coarse = criterion_detect(
        detect_logits_coarse, dummy_coarse_labels_detect
    )
    loss_class_coarse = criterion_class(class_logits_coarse, dummy_coarse_labels_class)

    # Total loss doesn't include Lre term
    total_loss_coarse = loss_detect_coarse + loss_class_coarse

    print(f"Coarse-Grained Mode:")
    print(f"  Detection Logits shape: {detect_logits_coarse.shape}")
    print(f"  Classification Logits shape: {class_logits_coarse.shape}")
    print(f"  Lre Loss (should be 0): {lre_loss_coarse.item():.4f}")
    print(f"  Detection Loss: {loss_detect_coarse.item():.4f}")
    print(f"  Classification Loss: {loss_class_coarse.item():.4f}")
    print(f"  Total Loss: {total_loss_coarse.item():.4f}")
