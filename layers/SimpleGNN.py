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
    def __init__(self, num_vars: int, rank: int, hidden_dim: int = 64):
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
        Q = self.q(x)  # (B, hidden_dim)
        K = self.k(x)  # (B, hidden_dim)
        V = self.v(x)  # (B, hidden_dim)

        # Compute attention weights (scaled dot-product)
        attn_logits = torch.bmm(Q.unsqueeze(2), K.unsqueeze(1)) / (self.rank ** 0.5)  # (B, p, p)
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




