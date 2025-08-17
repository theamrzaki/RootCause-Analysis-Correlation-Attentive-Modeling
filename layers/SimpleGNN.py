import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, num_vars, rank, hidden_dim=64, heads=8, extra_layers=1):
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

    def forward(self, x):
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
        Q = Q.view(B, p, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, p, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, p, self.heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.num_vars ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        h = torch.matmul(attn_weights, V)  # (B, heads, p, head_dim)

        # Merge heads
        h = h.transpose(1, 2).contiguous().view(B, p, self.heads * self.head_dim)

        # Residual connection (project input)
        # Project input to hidden_dim
        res = self.residual(x)  # (B, hidden_dim)

        # Unsqueeze to match h's variable dimension
        res = res.unsqueeze(1)  # (B, 1, hidden_dim)

        # Add residual
        h = h + res  # broadcast over p dimension (variables) # preserves weaker correlations

        # Norm
        h = self.norm1(h)

        # Flatten across variables before MLP (preserves all signals)
        h_flat = h.view(B, -1)
        h_mlp = self.mlp(h_flat)

        # Split into U and V
        U_flat, V_flat = torch.split(h_mlp, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        # Reconstruct coefficient matrix
        coeffs_k = torch.bmm(U, V.transpose(1, 2))
        return coeffs_k

class RecurrentAttentionCoeffGNN(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=64, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Shared GNN coeff extractor
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)  # project coeffs to hidden_dim
        self.rnn = nn.GRU(
            input_size=hidden_dim,   # flatten adjacency
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project back to coeff matrix
        self.out = nn.Linear(hidden_dim, num_vars * num_vars)

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

        coeffs_seq = []
        for k in range(O):
            coeff_k = self.base_net(inputs[:, k, :])    # (B, P, P)
            coeffs_seq.append(coeff_k.view(B, -1))

        # Sequence of coeffs from GNN: (B, O, P*P)
        coeffs_seq = torch.stack(coeffs_seq, dim=1)

        # Process with RNN
        coeffs_seq = self.in_proj(coeffs_seq)  # (B, O, hidden_dim)
        h, _ = self.rnn(coeffs_seq)  # (B, O, hidden_dim)
        coeffs_rnn = self.out(h).view(B, O, P, P)

        # Predictions (like your original loop)
        preds = torch.zeros((B, P), device=self.device)
        for k in range(O):
            preds = preds + torch.matmul(
                coeffs_rnn[:, k, :, :], inputs[:, k, :].unsqueeze(-1)
            ).squeeze(-1)

        return preds, coeffs_rnn


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



    




