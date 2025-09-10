import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

class SimpleCoeffGNN(nn.Module):
    def __init__(self, num_vars: int, rank: int, hidden_dim: int = 128, residual_scale: float = 0.1):
        """
        Simple GNN coefficient generator for one lag with low-rank factors.
        - num_vars: number of variables (p)
        - rank: low-rank factor size
        - hidden_dim: hidden layer size of MLP
        - residual_scale: scaling factor for residual identity
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.residual_scale = residual_scale

        # Learnable adjacency, small initialization
        self.adj = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)

        # 2-layer MLP for low-rank factors
        self.fc1 = nn.Linear(num_vars, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_vars * rank)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)

    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    def forward(self, x):
        """
        x: (B, p) input at one lag
        returns: coeffs_k (B, p, p)
        """
        # Row-normalized adjacency to prevent exploding messages
        adj_norm = F.softmax(self.adj, dim=1)
        h = torch.matmul(adj_norm, x.unsqueeze(2)).squeeze(2)  # (B, p)

        # Optional clamping to avoid huge values
        h = torch.clamp(h, -10.0, 10.0)

        # 2-layer MLP
        h = F.relu(self.fc1(h))
        out = self.fc2(h)

        # Split into U and V for low-rank reconstruction
        U_flat, V_flat = torch.split(out, self.num_vars * self.rank, dim=1)
        U = U_flat.view(-1, self.num_vars, self.rank)
        V = V_flat.view(-1, self.num_vars, self.rank)
        coeffs_k = torch.bmm(U, V.transpose(1, 2))  # (B, p, p)


        return coeffs_k
    

class AttentionCoeffGNN(nn.Module):
    def __init__(self, num_vars: int, rank: int, hidden_dim: int = 128):
        """
        Attention-based GNN coefficient generator for one lag.
        - num_vars: number of variables (p)
        - rank: low-rank factor size
        - hidden_dim: hidden layer size for attention MLP
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank

        # Linear layers to compute queries, keys, values
        self.q = nn.Linear(num_vars, hidden_dim)
        self.k = nn.Linear(num_vars, hidden_dim)
        self.v = nn.Linear(num_vars, hidden_dim)

        # MLP to project aggregated features to U and V
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_vars * rank)

        # Optional scaling parameter
        self.global_scale = nn.Parameter(torch.tensor(0.1))

    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)
            
    def forward(self, x):
        """
        x: (B, p) input at one lag
        returns: coeffs_k (B, p, p)
        """
        B, p = x.shape

        # Compute Q, K, V
        Q = self.q(x)       # (B, num_vars)
        K = self.k(x)       # (B, num_vars)
        V = self.v(x)       # (B, num_vars)
        attn_logits = torch.bmm(Q.unsqueeze(2), K.unsqueeze(1)) / (self.num_vars ** 0.5)  # (B, num_vars, num_vars)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate values
        h = torch.bmm(attn_weights, V.unsqueeze(2)).squeeze(2)  # (B, hidden_dim)

        # 2-layer MLP to predict low-rank factors
        h = F.relu(self.fc1(h))
        out = self.fc2(h)

        # Split into U and V and reconstruct coefficient matrix
        U_flat, V_flat = torch.split(out, self.num_vars * self.rank, dim=1)
        U = U_flat.view(-1, self.num_vars, self.rank)
        V = V_flat.view(-1, self.num_vars, self.rank)
        coeffs_k = torch.bmm(U, V.transpose(1, 2))

        # Optional residual identity + scaling
        #coeffs_k = coeffs_k * self.global_scale + torch.eye(self.num_vars, device=x.device)

        return coeffs_k



class AttentionCoeffGNN_multihead(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=16, heads=2, extra_layers=1):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.heads = heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.head_dim = hidden_dim // heads

        # Q, K, V projections
        self.q = nn.Linear(1, hidden_dim)
        self.k = nn.Linear(1, hidden_dim)
        self.v = nn.Linear(1, hidden_dim)

        # LayerNorm after attention
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Build MLP as one ModuleList
        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.ReLU())
        for _ in range(extra_layers):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_dim, 2 * num_vars * rank))
        self.mlp = nn.ModuleList(mlp_layers)

        # Optional scaling parameter
        self.global_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: (B, p)
        returns: coeffs_k: (B, p, p)
        """
        B, p = x.shape

        x_unsq = x.unsqueeze(-1)  # (B, p, 1)

        Q = self.q(x_unsq)  # (B, p, hidden_dim)
        K = self.k(x_unsq)  # (B, p, hidden_dim)
        V = self.v(x_unsq)  # (B, p, hidden_dim)

        # Split into heads
        Q = Q.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        K = K.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        V = V.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        
        # Compute scaled dot-product attention for each head
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.num_vars ** 0.5)  # (B, heads, p, p)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate values
        h = torch.matmul(attn_weights, V)  # (B, heads, p, head_dim)

        # Merge heads: (B, p, hidden_dim)
        h = h.transpose(1, 2).contiguous().view(B, p, self.heads * self.head_dim)

        # Mean-pool across p to get global vector: (B, hidden_dim)
        h = h.mean(dim=1)

        # Norm
        h = self.norm1(h)

        # Pass through MLP
        for layer in self.mlp:
            h = layer(h)

        # Split into U and V
        U_flat, V_flat = torch.split(h, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        # Reconstruct coefficient matrix
        coeffs_k = torch.bmm(U, V.transpose(1, 2))
        return coeffs_k
    
    
class AttentionCoeffGNN_multihead_fixed(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=128, heads=4, extra_layers=1):
        """
        Multi-head attention coefficient generator (fixed version).
        - Avoids mean pooling to preserve mid-ranked signals.
        - Adds residual connection to maintain weaker correlations.
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.heads = heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.head_dim = hidden_dim // heads

        # Q, K, V projections
        self.q = nn.Linear(1, hidden_dim)
        self.k = nn.Linear(1, hidden_dim)
        self.v = nn.Linear(1, hidden_dim)

        # LayerNorm after attention
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Residual projection from input
        self.residual = nn.Linear(num_vars, hidden_dim)

        # Build MLP as nn.Sequential (simpler than ModuleList loop)
        mlp_layers = [nn.Linear(num_vars * hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(extra_layers):
            mlp_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        mlp_layers.append(nn.Linear(hidden_dim, 2 * num_vars * rank))
        self.mlp = nn.Sequential(*mlp_layers)

        # Optional scaling parameter
        self.global_scale = nn.Parameter(torch.tensor(0.1))
    
    def _init_weights(self):
        # Initialize Linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, attn_dropout=0.1):
        """
        x: (B, p)
        returns: coeffs_k: (B, p, p)
        """
        B, p = x.shape
        x_unsq = x.unsqueeze(-1)  # (B, p, 1)

        # Project Q, K, V
        Q = self.q(x_unsq)
        K = self.k(x_unsq)
        V = self.v(x_unsq)

        # Split into heads
        Q = Q.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        K = K.view(B, p, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, p, self.heads, self.head_dim).transpose(1, 2)

        # Head-wise LayerNorm
        Q = F.layer_norm(Q, (self.head_dim,))
        K = F.layer_norm(K, (self.head_dim,))
        V = F.layer_norm(V, (self.head_dim,))

        # Attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.num_vars ** 0.5)
        tau = 0.3  # temperature for sharper attention
        attn_weights = F.softmax(attn_logits / tau, dim=-1)  # try tau ∈ {0.3, 0.5, 0.7, 1.0}

        # Attention dropout
        attn_weights = F.dropout(attn_weights, p=attn_dropout, training=self.training)

        h = torch.matmul(attn_weights, V)  # (B, heads, p, head_dim)

        # Merge heads
        h = h.transpose(1, 2).contiguous().view(B, p, self.heads * self.head_dim)  # (B, p, hidden_dim)

        # Residual connection with scaling
        res = self.residual(x).unsqueeze(1)  # (B, 1, hidden_dim)
        alpha = 0.9  # scale residual to stabilize
        h = h + alpha * res  # broadcast over p dimension

        # LayerNorm over hidden_dim after residual
        h = self.norm1(h)

        # Flatten across variables before MLP
        h_flat = h.view(B, -1)
        h_mlp = self.mlp(h_flat)

        # Split into U and V
        U_flat, V_flat = torch.split(h_mlp, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        # Reconstruct coefficient matrix
        coeffs_k = torch.bmm(U, V.transpose(1, 2))
        attn_mean = attn_weights.mean(dim=1)
        return coeffs_k,attn_mean

class RecurrentAttentionCoeffGNN__(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=64, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Shared GNN coefficient extractor per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)  # project flattened coeffs
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project hidden state to coefficient adjustment
        self.context_proj = nn.Linear(hidden_dim, num_vars * num_vars)
    def _init_weights(self):
        # Initialize Linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, inputs: torch.Tensor):
        """
        inputs: (B, order, num_vars)
        returns:
            preds: (B, num_vars)
            coeffs: (B, order, num_vars, num_vars)
        """
        B, O, P = inputs.shape
        if (O, P) != (self.order, self.num_vars):
            print("WARNING: inputs should be of shape BS x K x p")

        # --- Step 1: compute per-lag coefficients ---
        coeffs_seq = []
        for k in range(O):
            coeff_k = self.base_net(inputs[:, k, :])    # (B, P, P)
            coeffs_seq.append(coeff_k.view(B, -1))      # flatten

        # Sequence: (B, O, P*P)
        coeffs_seq = torch.stack(coeffs_seq, dim=1)

        # --- Step 2: process sequence with RNN ---
        rnn_input = self.in_proj(coeffs_seq)  # (B, O, hidden_dim)
        h_seq, h_final = self.rnn(rnn_input)  # h_seq: (B, O, hidden_dim), h_final: (num_layers, B, hidden_dim)
        h_final = h_final[-1]                 # (B, hidden_dim) last layer final hidden

        # --- Step 3: project hidden state to coefficient adjustment ---
        context_adjust = self.context_proj(h_final).view(B, P, P)  # (B, P, P)

        # --- Step 4: add global context to each lag coefficient ---
        coeffs_rnn = coeffs_seq.view(B, O, P, P) + context_adjust.unsqueeze(1)  # broadcast across lags

        # --- Step 5: compute predictions ---
        preds = torch.zeros((B, P), device=self.device)
        for k in range(O):
            preds += torch.matmul(coeffs_rnn[:, k, :, :], inputs[:, k, :].unsqueeze(-1)).squeeze(-1)

        return preds, coeffs_rnn

class RecurrentAttentionCoeffGNN_chunks(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=32, proj_dim=32, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Shared GNN coefficient extractor per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project hidden state to coefficient adjustment
        self.context_proj = nn.Linear(hidden_dim, num_vars * num_vars)

        # 🔑 Projection for coefficients to smaller space
        self.coeff_proj = nn.Linear(num_vars * num_vars, proj_dim)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 16, return_coeffs: bool = True):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = [] if return_coeffs else None

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: compute per-lag coefficients ---
            coeffs_seq = []
            for k in range(O):
                coeff_k = self.base_net(x_chunk[:, k, :])    # (B_chunk, P, P)
                coeffs_seq.append(coeff_k.view(B_chunk, -1)) # (B_chunk, P*P)

            coeffs_seq = torch.stack(coeffs_seq, dim=1)      # (B_chunk, O, P*P)

            # --- Step 2: process sequence with RNN ---
            rnn_input = self.in_proj(coeffs_seq)             # (B_chunk, O, hidden_dim)
            h_seq, h_final = self.rnn(rnn_input)
            h_final = h_final[-1]                            # (B_chunk, hidden_dim)

            # --- Step 3: project hidden state to coefficient adjustment ---
            context_adjust = self.context_proj(h_final).view(B_chunk, P, P)

            # --- Step 4: add global context ---
            coeffs_rnn = coeffs_seq.view(B_chunk, O, P, P) + context_adjust.unsqueeze(1)

            # --- Step 5: compute predictions ---
            preds_chunk = torch.zeros((B_chunk, P), device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_rnn[:, k, :, :], x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)

            if return_coeffs:
                # 🔑 project coeffs to smaller space before storing
                coeffs_proj = self.coeff_proj(coeffs_rnn.view(B_chunk, O, -1))  # (B_chunk, O, proj_dim)
                coeffs_list.append(coeffs_proj)

        preds = torch.cat(preds_list, dim=0)
        coeffs_seq = torch.cat(coeffs_list, dim=0) if return_coeffs else None

        return preds, coeffs_seq
    

class RecurrentAttentionCoeffGNN(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=32, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Base network per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.rnn = nn.GRU(
            input_size=num_vars * num_vars,  # flattened output from base_net
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project hidden state to prediction
        self.pred_proj = nn.Linear(hidden_dim, num_vars)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k = self.base_net(x_chunk[:, k, :])    # (B_chunk, P, P)
                features_seq.append(feat_k.view(B_chunk, -1))  # flatten to (B_chunk, P*P)

            features_seq = torch.stack(features_seq, dim=1)  # (B_chunk, O, P*P)

            # --- Step 2: process with RNN ---
            _, h_final = self.rnn(features_seq)
            h_final = h_final[-1]  # (B_chunk, hidden_dim)

            # --- Step 3: predict from hidden state ---
            preds_chunk = self.pred_proj(h_final)  # (B_chunk, P)
            preds_list.append(preds_chunk)

        preds = torch.cat(preds_list, dim=0)  # (B, P)
        return preds, None
    

class RecurrentAttentionGNN_Attn_nocoeff(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=64, num_heads=4, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device

        # Base GNN per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # Project flattened GNN output to hidden dimension for attention
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # Temporal attention across lags
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Output projection from aggregated hidden state
        self.pred_proj = nn.Linear(hidden_dim, num_vars)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k = self.base_net(x_chunk[:, k, :])     # (B_chunk, P, P)
                features_seq.append(feat_k.view(B_chunk, -1))  # flatten to (B_chunk, P*P)

            features_seq = torch.stack(features_seq, dim=1)   # (B_chunk, O, P*P)

            # --- Step 2: project to hidden_dim for attention ---
            attn_input = self.in_proj(features_seq)          # (B_chunk, O, hidden_dim)

            # --- Step 3: temporal attention ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)  # (B_chunk, O, hidden_dim)

            # --- Step 4: aggregate across lags ---
            # simple mean pooling across lags
            agg_hidden = attn_out.mean(dim=1)               # (B_chunk, hidden_dim)

            # --- Step 5: predict from aggregated hidden state ---
            preds_chunk = self.pred_proj(agg_hidden)        # (B_chunk, P)
            preds_list.append(preds_chunk)

        preds = torch.cat(preds_list, dim=0)               # (B, P)
        return preds, None


class RecurrentAttentionGNN_Attn(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order  # number of lags
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale  # scale factor for positional embedding

        # Base GNN per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # Project flattened GNN output to hidden_dim for attention
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # Temporal attention across lags
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Map attention hidden state back to coefficients
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)

        # Learnable positional embedding (small initial scale)
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = []

        # Use positional encoding
        pos_embeddings = self.pos_enc[:, :O, :]  # shape (1, O, hidden_dim)

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k, _ = self.base_net(x_chunk[:, k, :])        # (B_chunk, P, P)
                features_seq.append(feat_k.view(B_chunk, -1))   # (B_chunk, P*P)
            features_seq = torch.stack(features_seq, dim=1)      # (B_chunk, O, P*P)

            # --- Step 2: project to hidden_dim for attention + add positional encoding ---
            attn_input = self.in_proj(features_seq) + pos_embeddings  # (B_chunk, O, hidden_dim)

            # --- Step 3: temporal attention ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)  # (B_chunk, O, hidden_dim)

            # --- Step 4: map attention outputs back to coeffs ---
            coeffs_seq = self.coeff_proj(attn_out)               # (B_chunk, O, P*P)
            coeffs_seq = coeffs_seq.view(B_chunk, O, P, P)       # (B_chunk, O, P, P)

            # --- Step 5: prediction using coeffs ---
            preds_chunk = torch.zeros((B_chunk, P), device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_seq[:, k, :, :], 
                                            x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)

        preds = torch.cat(preds_list, dim=0)                    # (B, P)
        coeffs = torch.cat(coeffs_list, dim=0)                  # (B, O, P, P)

        return preds, coeffs, None 


class RecurrentAttentionGNN_Attn_fourier(nn.Module):
    """
    Time domain path: exactly like your current RecurrentAttentionGNN_Attn.
    Freq domain path: rFFT over lags (per variable) -> magnitude -> per-bin GNN -> attention over bins.
    Fusion: gated blend of (time coeffs) and (freq coeffs).
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale

        # Shared per-slice GNN (use your fixed variant; swap if needed)
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Freq path ---
        # rFFT along lags ⇒ F = order//2 + 1 bins
        self.in_proj_freq = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_freq = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_freq = nn.Parameter(torch.randn(1, (order // 2) + 1, hidden_dim) * pe_scale)

        # --- Gated fusion (global context → α in [0,1]) ---
        # take simple stats from the window as context
        ctx_dim = 2 * num_vars  # mean & std per variable, concatenated
        self.fusion_gate = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    
    def _init_weights(self):
        """Initialize weights for linear layers and attention projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch MHA already uses xavier for in_proj_weight internally, but can re-init if needed
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
                    
    @torch.no_grad()
    def _context_stats(self, x_win):  # x_win: (B, O, P)
        mean = x_win.mean(dim=1)       # (B, P)
        std = x_win.std(dim=1)         # (B, P)
        return torch.cat([mean, std], dim=-1)  # (B, 2P)

    def _time_path(self, x_chunk):  # x_chunk: (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            # base_net expects (B, P) and returns (B, P, P)
            coeff_k, _ = self.base_net(x_chunk[:, k, :])            # (Bc, P, P)
            feats.append(coeff_k.view(Bc, -1))                      # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                              # (Bc, O, P*P)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)  # (Bc, O, H)
        coeffs_seq = self.coeff_proj_time(attn_out).view(Bc, O, P, P)     # (Bc, O, P, P)
        # prediction using coeffs per lag
        preds = torch.zeros((Bc, P), device=x_chunk.device)
        for k in range(O):
            preds += (coeffs_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs_seq

    def _freq_path(self, x_chunk):  # x_chunk: (B_chunk, O, P)
        """
        rFFT over the lag axis (dim=1). We use magnitude per variable & bin: (B, F, P).
        Each frequency bin plays the role of a "lag slice" for base_net.
        """
        Bc, O, P = x_chunk.shape
        X = torch.fft.rfft(x_chunk, dim=1)                     # (Bc, F, P) complex
        Xmag = X.abs()                                         # (Bc, F, P) real magnitudes
        Fbins = Xmag.size(1)

        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(Xmag[:, f, :])          # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))                 # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                        # (Bc, F, P*P)

        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)   # (Bc, F, H)
        coeffs_seq = self.coeff_proj_freq(attn_out).view(Bc, Fbins, P, P)  # (Bc, F, P, P)

        # collapse over frequency bins (attention already weighs them; mean works well)
        coeffs_collapsed = coeffs_seq.mean(dim=1)                              # (Bc, P, P)

        # one-step prediction proxy: apply collapsed coeffs to the *latest* lag input
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)  # (Bc, P)
        return preds, coeffs_seq, coeffs_collapsed

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []
        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- time path ---
            preds_t, coeffs_t = self._time_path(x_chunk)             # (Bch,P), (Bch,O,P,P)

            # --- freq path ---
            preds_f, coeffs_f_seq, coeffs_f_collapsed = self._freq_path(x_chunk)  # (Bch,P), (Bch,F,P,P), (Bch,P,P)

            # --- fusion gate ---
            ctx = self._context_stats(x_chunk)                        # (Bch, 2P)
            alpha = self.fusion_gate(ctx)                             # (Bch, 1) in [0,1]

            # fuse predictions (optional; used mainly for training signal)
            preds = alpha * preds_t + (1 - alpha) * preds_f

            # fuse coefficients: time path gives per-lag; freq path is collapsed
            # broadcast alpha to match (Bch, O, 1, 1) and (Bch, 1, 1)
            alpha_time = alpha.view(-1, 1, 1, 1)
            alpha_freq = (1 - alpha).view(-1, 1, 1)

            # add freq coeffs into each lag as a global periodic prior
            alpha_freq = alpha_freq.unsqueeze(-1)  # [131, 1, 1, 1]
            coeffs_fused = alpha_time * coeffs_t + alpha_freq * coeffs_f_collapsed[:, None, :, :]  # (Bch,O,P,P)

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_fused)
            coeffs_freq_out.append(coeffs_f_seq)  # keep raw per-bin seq if you want diagnostics

        preds = torch.cat(preds_out, dim=0)                     # (B, P)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)    # (B, O, P, P)  (fused)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)     # (B, F, P, P)

        return preds, coeffs_time_like, coeffs_freq_seq


class RecurrentAttentionGNN_Attn_crossattn(nn.Module):
    """
    Time domain path: like original RecurrentAttentionGNN_Attn.
    Freq domain path: rFFT over lags (per variable) -> magnitude -> per-bin GNN -> attention over bins.
    Cross-attention: time <-> freq interaction before prediction.
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale

        # Shared GNN per slice
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Freq path ---
        self.in_proj_freq = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_freq = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_freq = nn.Parameter(torch.randn(1, (order // 2) + 1, hidden_dim) * pe_scale)

        # --- Cross-attention ---
        self.cross_attn_time_to_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_freq_to_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def _init_weights(self):
        """Initialize weights for linear layers and attention projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch MHA already uses xavier for in_proj_weight internally, but can re-init if needed
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
                    
    @torch.no_grad()
    def _context_stats(self, x_win):  # (B, O, P)
        mean = x_win.mean(dim=1)      # (B, P)
        std = x_win.std(dim=1)        # (B, P)
        return torch.cat([mean, std], dim=-1)  # (B, 2P)

    def _time_path(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            coeff_k, _ = self.base_net(x_chunk[:, k, :])  # (Bc, P, P)
            feats.append(coeff_k.view(Bc, -1))            # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                   # (Bc, O, P*P)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)  # (Bc, O, H)
        return attn_out, seq  # return seq for potential diagnostics

    def _freq_path(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        X = torch.fft.rfft(x_chunk, dim=1)             # (Bc, F, P) complex
        Xmag = X.abs()                                 # (Bc, F, P)
        Fbins = Xmag.size(1)

        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(Xmag[:, f, :])  # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))         # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                # (Bc, F, P*P)
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)  # (Bc, F, H)
        return attn_out, seq, Xmag

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape

        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- encode time & freq ---
            attn_time, _ = self._time_path(x_chunk)  # (Bch, O, H)
            attn_freq, _, _ = self._freq_path(x_chunk)  # (Bch, F, H)

            # --- cross-attention ---
            attn_time_cross, _ = self.cross_attn_time_to_freq(attn_time, attn_freq, attn_freq)  # (Bch, O, H)
            attn_freq_cross, _ = self.cross_attn_freq_to_time(attn_freq, attn_time, attn_time)  # (Bch, F, H)

            # --- project to coeffs ---
            coeffs_time_seq = self.coeff_proj_time(attn_time_cross)  # (Bch, O, P*P)
            coeffs_time_seq = coeffs_time_seq.view(-1, O, P, P)

            coeffs_freq_seq = self.coeff_proj_freq(attn_freq_cross)  # (Bch, F, P*P)
            Fbins = coeffs_freq_seq.size(1)
            coeffs_freq_seq = coeffs_freq_seq.view(-1, Fbins, P, P)
            coeffs_freq_collapsed = coeffs_freq_seq.mean(dim=1)     # (Bch, P, P)

            # --- prediction ---
            preds_time = torch.zeros((x_chunk.size(0), P), device=x_chunk.device)
            for k in range(O):
                preds_time += (coeffs_time_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
            preds_freq = (coeffs_freq_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)

            # --- final fused prediction ---
            preds = 0.5 * preds_time + 0.5 * preds_freq  # can adjust weighting or use a learned gate

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_time_seq)
            coeffs_freq_out.append(coeffs_freq_seq)

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)

        return preds, coeffs_time_like, coeffs_freq_seq


class RecurrentAttentionGNN_Attn_legendre(nn.Module):
    """
    Time domain path: same as original.
    Freq domain path: Legendre polynomial projection over lags (using numpy) -> per-basis GNN -> attention over basis.
    Fusion: gated blend of time and polynomial coefficients.
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01, num_basis=None):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        self.num_basis = num_basis or (order // 2 + 1)  # default similar to FFT bins

        # Shared per-slice GNN
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Legendre path ---
        self.in_proj_legendre = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_legendre = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_legendre = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_legendre = nn.Parameter(torch.randn(1, self.num_basis, hidden_dim) * pe_scale)

        # --- Fusion gate ---
        ctx_dim = 2 * num_vars  # mean & std per variable
        self.fusion_gate = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Precompute Legendre basis using NumPy
        self.register_buffer("legendre_basis", self._compute_legendre_basis(order, self.num_basis))

    def _compute_legendre_basis(self, order, num_basis):
        """
        Generate Legendre polynomials using NumPy.
        Returns a tensor of shape (order, num_basis)
        """
        x = np.linspace(-1, 1, order)
        basis = []
        for n in range(num_basis):
            Pn = np.polynomial.legendre.Legendre([0]*n + [1])(x)  # n-th Legendre polynomial
            basis.append(Pn)
        basis = np.stack(basis, axis=1)  # (order, num_basis)
        return torch.tensor(basis, dtype=torch.float32)

    @torch.no_grad()
    def _context_stats(self, x_win):
        mean = x_win.mean(dim=1)
        std = x_win.std(dim=1)
        return torch.cat([mean, std], dim=-1)

    def _time_path(self, x_chunk):
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            coeff_k, _ = self.base_net(x_chunk[:, k, :])
            feats.append(coeff_k.view(Bc, -1))
        seq = torch.stack(feats, dim=1)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)
        coeffs_seq = self.coeff_proj_time(attn_out).view(Bc, O, P, P)

        preds = torch.zeros((Bc, P), device=x_chunk.device)
        for k in range(O):
            preds += (coeffs_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs_seq

    def _legendre_path(self, x_chunk):
        # x_chunk: (B, O, P)
        Bc, O, P = x_chunk.shape
        order, num_basis = self.legendre_basis.shape

        # Resize legendre basis to match O
        legendre_resized = []
        for b in range(num_basis):
            leg_b = self.legendre_basis[:, b].cpu().numpy()           # shape (order,)
            leg_b_resized = np.interp(np.linspace(0, order-1, O), np.arange(order), leg_b)
            legendre_resized.append(leg_b_resized)
        legendre_resized = torch.tensor(np.stack(legendre_resized, axis=1), device=x_chunk.device, dtype=torch.float32)  # (O, num_basis)

        # Project input
        Xproj = torch.einsum('bok,kf->bfo', x_chunk, legendre_resized)

        feats = []
        for f in range(self.num_basis):
            coeff_f, _ = self.base_net(Xproj[:, f, :])
            feats.append(coeff_f.view(Bc, -1))
        seq = torch.stack(feats, dim=1)

        attn_in = self.in_proj_legendre(seq) + self.pos_enc_legendre[:, :self.num_basis, :]
        attn_out, _ = self.temporal_attn_legendre(attn_in, attn_in, attn_in)
        coeffs_seq = self.coeff_proj_legendre(attn_out).view(Bc, self.num_basis, P, P)

        coeffs_collapsed = coeffs_seq.mean(dim=1)
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs_seq, coeffs_collapsed


    def forward(self, inputs, batch_chunk_size=1000):
        B, O, P = inputs.shape
        preds_out, coeffs_time_out, coeffs_legendre_out = [], [], []
        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]

            preds_t, coeffs_t = self._time_path(x_chunk)
            preds_l, coeffs_l_seq, coeffs_l_collapsed = self._legendre_path(x_chunk)

            ctx = self._context_stats(x_chunk)
            alpha = self.fusion_gate(ctx)

            preds = alpha * preds_t + (1 - alpha) * preds_l

            alpha_time = alpha.view(-1, 1, 1, 1)
            alpha_leg = (1 - alpha).view(-1, 1, 1).unsqueeze(-1)
            coeffs_fused = alpha_time * coeffs_t + alpha_leg * coeffs_l_collapsed[:, None, :, :]

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_fused)
            coeffs_legendre_out.append(coeffs_l_seq)

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_legendre_seq = torch.cat(coeffs_legendre_out, dim=0)

        return preds, coeffs_time_like, coeffs_legendre_seq

class RecurrentAttentionGNN_Attn_______(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale

        # Base GNN per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # Project GNN output to hidden_dim per lag
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # Temporal attention across lags
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Residual connection for temporal attention
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Map attention hidden state back to coefficients
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = []

        # Use positional encoding
        pos_embeddings = self.pos_enc[:, :O, :]  # shape (1, O, hidden_dim)

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k, _ = self.base_net(x_chunk[:, k, :])        # (B_chunk, P, P)
                # Flatten GNN output
                feat_k_flat = feat_k.view(B_chunk, -1)
                # Add input residual, tiled to match size
                feat_k_flat = feat_k_flat + x_chunk[:, k, :].repeat(1, P)
                features_seq.append(feat_k_flat)                   # (B_chunk, P*P)
            features_seq = torch.stack(features_seq, dim=1)       # (B_chunk, O, P*P)

            # --- Step 2: project to hidden_dim for attention + add positional encoding ---
            attn_input = self.in_proj(features_seq) + pos_embeddings  # (B_chunk, O, hidden_dim)

            # --- Step 3: temporal attention ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)  # (B_chunk, O, hidden_dim)

            # --- Step 4: map attention outputs back to coeffs ---
            coeffs_seq = self.coeff_proj(attn_out)               # (B_chunk, O, P*P)
            coeffs_seq = coeffs_seq.view(B_chunk, O, P, P)       # (B_chunk, O, P, P)

            # --- Step 5: prediction using coeffs ---
            preds_chunk = torch.zeros((B_chunk, P), device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_seq[:, k, :, :], 
                                            x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)

        preds = torch.cat(preds_list, dim=0)                    # (B, P)
        coeffs = torch.cat(coeffs_list, dim=0)                  # (B, O, P, P)

        return preds, coeffs, None



class RecurrentAttentionGNN_Attn___(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=4,
                 attention_heads=6, attention_dim=128, pe_scale=0.05, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale

        # --- Base GNN per lag ---
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Project GNN output to attention hidden_dim ---
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # --- Learnable positional embedding ---
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Temporal multi-head attention (residual + normalization) ---
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)

        # --- Dilated temporal convolution (optional, helps multi-scale) ---
        self.dilated_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)

        # --- Map attention output back to coefficients ---
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list, coeffs_list, attn_seq = [], [], []

        pos_embeddings = self.pos_enc[:, :O, :]

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]
            B_chunk = x_chunk.size(0)

            # --- Step 1: base GNN per lag ---
            features_seq = []
            attn_lags = []
            for k in range(O):
                feat_k, attn_k = self.base_net(x_chunk[:, k, :])
                features_seq.append(feat_k.view(B_chunk, -1))
                attn_lags.append(attn_k)
            features_seq = torch.stack(features_seq, dim=1)  # (B_chunk, O, P*P)
            attn_lags = torch.stack(attn_lags, dim=1)        # (B_chunk, O, P, P)

            # --- Step 2: project to hidden_dim + positional encoding ---
            attn_input = self.in_proj(features_seq) + pos_embeddings

            # --- Step 3: temporal attention with residual ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)
            attn_out = self.attn_layer_norm(attn_out + attn_input)

            # --- Step 4: dilated temporal conv (multi-scale) ---
            attn_out = self.dilated_conv(attn_out.transpose(1, 2)).transpose(1, 2)

            # --- Step 5: map back to coefficient matrix ---
            coeffs_seq = self.coeff_proj(attn_out).view(B_chunk, O, P, P)

            # --- Step 6: prediction ---
            preds_chunk = torch.zeros(B_chunk, P, device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_seq[:, k, :, :], x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)
            attn_seq.append(attn_lags)

        preds = torch.cat(preds_list, dim=0)
        coeffs = torch.cat(coeffs_list, dim=0)
        attn_seq = torch.cat(attn_seq, dim=0)  # (B, O, P, P)

        return preds, coeffs, attn_seq


import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGNN(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=64, heads=8, extra_layers=1, temporal_hidden=32):
        """
        TemporalGNN:
        - Spatial: AttentionCoeffGNN_multihead
        - Temporal: small GRUCell over spatial embeddings
        - Output: preds + coeffs
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.temporal_hidden = temporal_hidden

        # Spatial GNN
        self.spatial_gnn = AttentionCoeffGNN_multihead(
            num_vars=num_vars,
            rank=rank,
            hidden_dim=hidden_dim,
            heads=heads,
            extra_layers=extra_layers
        )

        # Temporal projection (small recurrent model)
        self.proj = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_rnn = nn.GRUCell(hidden_dim, temporal_hidden)

        # Final MLP to produce U, V for coeffs
        self.final_mlp = nn.Sequential(
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.ReLU(),
            nn.Linear(temporal_hidden, 2 * num_vars * rank)
        )

    def forward(self, x_seq):
        """
        x_seq: (B, order, num_vars)
        returns: preds, coeffs: (B, num_vars, num_vars)
        """
        B, order, p = x_seq.shape
        device = x_seq.device

        # Init temporal hidden state
        h_t = torch.zeros(B, self.temporal_hidden, device=device)

        for t in range(order):
            x_t = x_seq[:, t, :]  # (B, num_vars)

            # Spatial GNN produces coeffs
            coeffs_k = self.spatial_gnn(x_t)  # (B, p, p)

            # Flatten and project to hidden_dim
            h_embed = self.proj(coeffs_k.view(B, -1))  # (B, hidden_dim)

            # Update recurrent state
            h_t = self.temporal_rnn(h_embed, h_t)

        # Decode final hidden state into U, V
        h_final = self.final_mlp(h_t)
        U_flat, V_flat = torch.split(h_final, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        coeffs = torch.bmm(U, V.transpose(1, 2))  # (B, p, p)
        preds = coeffs  # optionally you can apply some post-processing for preds

        return preds, coeffs



    

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Lightweight Temporal Encoder (efficient) ----------
class TemporalDilatedEncoderFast(nn.Module):
    def __init__(self, num_vars, hidden_dim=128, dilations=(1,2,4,8)):
        super().__init__()
        self.proj = nn.Linear(num_vars, hidden_dim)
        # Use small stack of depthwise separable friendly convs (fast)
        self.layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in dilations])

    def forward(self, x):  # x: (B,O,P)
        h = self.proj(x)                # (B,O,H)
        h = h.transpose(1,2)            # (B,H,O)
        for conv, ln in zip(self.layers, self.norms):
            z = conv(h)                 # (B,H,O)
            z = F.gelu(z)
            # LayerNorm expects (B,O,H) so transpose
            z_t = z.transpose(1,2)      # (B,O,H)
            z_t = ln(z_t)
            z = z_t.transpose(1,2)     # (B,H,O)
            h = h + z                  # residual (same shapes)
        h = h.transpose(1,2)            # (B,O,H)
        return h

# ---------- Deterministic Graph Learner (fast, cached) ----------
class GraphLearnerDeterministic(nn.Module):
    def __init__(self, num_vars, hidden_dim=128):
        super().__init__()
        self.num_vars = num_vars
        self.src = nn.Parameter(torch.randn(num_vars, hidden_dim) * 0.01)
        self.dst = nn.Parameter(torch.randn(num_vars, hidden_dim) * 0.01)
        self.scorer = nn.Linear(hidden_dim, 1)
        # cache
        self._cached_A = None
        self._cached_device = None
        self._cached_version = 0  # bump when force-recompute

    def _compute_logits(self):
        P = self.num_vars
        src = self.src.unsqueeze(1).expand(P, P, -1)   # (P,P,H)
        dst = self.dst.unsqueeze(0).expand(P, P, -1)   # (P,P,H)
        h = torch.tanh(src + dst)                      # (P,P,H)
        logits = self.scorer(h).squeeze(-1)            # (P,P)
        mask_eye = torch.eye(P, device=logits.device).bool()
        logits = logits.masked_fill(mask_eye, -1e9)
        return logits

    def forward(self, force_recompute=False):
        # compute deterministic adjacency via sigmoid(logits)
        device = self.src.device
        if (self._cached_A is None) or force_recompute or (self._cached_device != device):
            logits = self._compute_logits().to(device)
            A = torch.sigmoid(logits)   # (P,P) deterministic
            self._cached_A = A
            self._cached_device = device
            self._cached_version += 1
        else:
            A = self._cached_A
        # l0 proxy (mean) can be used if you want a reg term externally
        l0 = torch.mean(A)
        return A, l0

# ---------- Fast single-hop Message Passing (vectorized) ----------
class GraphMessagePassingFast(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.msg_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_nodes, A):
        # h_nodes: (B,O,P,H)
        B, O, P, H = h_nodes.shape
        # flatten (B*O,P,H) for single matmul
        h_flat = h_nodes.view(B*O, P, H)                 # (B*O,P,H)
        # neighbor aggregation via A^T (fast single matmul)
        # new_nodes[bop] = A^T @ h_flat[bop]  -> (B*O, P, H)
        new_nodes = torch.matmul(A.transpose(0,1), h_flat)  # (B*O,P,H)
        new_nodes = self.msg_out(new_nodes)               # (B*O,P,H)
        out = (h_flat + new_nodes).view(B, O, P, H)        # residual
        return self.norm(out)

# ---------- Lightweight "MoE-lite" -> small residual MLP ----------
class MoELite(nn.Module):
    def __init__(self, hidden_dim=128, n_hidden=1):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):  # (B,O,P,H)
        out = self.net(h)
        return self.norm(h + out)

# ---------- Hybrid Coefficient Decoder (same idea, vectorized) ----------
class HybridCoeffDecoderFast(nn.Module):
    def __init__(self, num_vars, hidden_dim=128, rank=32):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.u_head = nn.Linear(hidden_dim, num_vars * rank)
        self.v_head = nn.Linear(hidden_dim, num_vars * rank)
        self.diag_head = nn.Linear(hidden_dim, num_vars)
        # remove huge sparse_head for speed (optional); keep lowrank+diag only
        # if you want sparse residual keep a small head, but it's heavier.
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h):  # h: (B,O,H)
        B, O, H = h.shape
        P = self.num_vars
        U = self.u_head(h).view(B, O, P, self.rank)    # (B,O,P,r)
        V = self.v_head(h).view(B, O, P, self.rank)    # (B,O,P,r)
        # lowrank via einsum
        lowrank = torch.einsum("boir,bojr->boij", U, V)   # (B,O,P,P)
        diag = torch.diag_embed(self.diag_head(h).view(B, O, P))  # (B,O,P,P)
        return lowrank + diag

# ---------- Fast Temporal Causal Model (vectorized and cached A) ----------
class TemporalCausalMoE(nn.Module):
    def __init__(self, num_vars, order, rank=32, hidden_dim=128,
                 dilations=(1,2,4,8), heads=8, n_experts=4, pe_scale=0.1,
                 l0_lambda=1e-4, fast_mode=True, use_fp16=False):
        """
        Same external signature as before but optimized.
        Set fast_mode=True to enable the fast path (default).
        """
        super().__init__()
        self.num_vars = num_vars
        self.order = order
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.fast_mode = fast_mode
        self.use_fp16 = use_fp16

        # lightweight temporal encoder
        self.temporal = TemporalDilatedEncoderFast(num_vars, hidden_dim, dilations)

        # positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # deterministic graph learner (cached)
        self.graph = GraphLearnerDeterministic(num_vars, hidden_dim)

        # message passing (fast single matmul)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gmp = GraphMessagePassingFast(hidden_dim)

        # light MoE replacement
        self.moe = MoELite(hidden_dim, n_hidden=1)

        # coefficient decoder (lowrank + diag)
        self.coeff_decoder = HybridCoeffDecoderFast(num_vars, hidden_dim, rank)

    def forward(self, inputs, force_recompute_A=False):  # inputs: (B,O,P)
        import contextlib

        # optionally run in half precision
        if self.use_fp16 and inputs.is_cuda:
            dtype_switch = torch.cuda.amp.autocast(enabled=True)
        else:
            dtype_switch = contextlib.nullcontext()

        with dtype_switch:
            B, O, P = inputs.shape

            # 1) temporal features (vectorized)
            h_lag = self.temporal(inputs)                          # (B,O,H)
            # add pos enc (broadcast)
            pos = self.pos_enc[:, :O, :].to(h_lag.dtype).to(h_lag.device)
            h_lag = h_lag + pos

            # 2) broadcast to nodes (clone to avoid view inplace issues)
            h_nodes = self.node_proj(h_lag).unsqueeze(2).expand(B, O, P, self.hidden_dim).clone()  # (B,O,P,H)

            # 3) deterministic adjacency (cached) + message passing
            A, l0 = self.graph(force_recompute_A)   # (P,P), scalar proxy
            h_nodes = self.gmp(h_nodes, A)          # (B,O,P,H)

            # 4) light MoE (small MLP)
            h_nodes = self.moe(h_nodes)             # (B,O,P,H)

            # 5) aggregate nodes per lag
            h_agg = h_nodes.mean(dim=2)             # (B,O,H)

            # 6) decode coefficients vectorized
            coeffs_seq = self.coeff_decoder(h_agg)  # (B,O,P,P)

            # 7) vectorized prediction: einsum over lag and input var -> (B,P)
            preds = torch.einsum("boij,boj->bi", coeffs_seq, inputs)  # (B,P)

        return preds, coeffs_seq



