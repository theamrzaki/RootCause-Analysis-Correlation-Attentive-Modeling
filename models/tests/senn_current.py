import torch.nn as nn
import torch
import torch.nn.functional as F

class SENNGC_attention_per_lag(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        super(SENNGC___, self).__init__()

        self.num_vars = num_vars  # p
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.device = device

        self.context_radius = 1  # neighbors on each side
        self.window_size = 2 * self.context_radius + 1  # typically 3

        # gating networks per lag: input is flattened local window (window_size * p), output attention over offsets
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_size * self.num_vars, self.window_size),
                nn.Softmax(dim=1)
            )
            for _ in range(order)
        ])

        # coefficient nets per lag: input is the attended summary vector of size p
        self.coeff_nets = nn.ModuleList()
        for _ in range(order):
            layers = [nn.Linear(self.num_vars, hidden_layer_size), nn.ReLU()]
            for _ in range(self.num_hidden_layers - 1):
                layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()]
            layers += [nn.Linear(hidden_layer_size, self.num_vars ** 2), nn.Tanh()]
            self.coeff_nets.append(nn.Sequential(*layers))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def _gather_local_window(self, inputs, k):
        # returns [B, window_size, p]
        B, K, p = inputs.shape
        r = self.context_radius
        window = []
        for offset in range(-r, r + 1):
            idx = k + offset
            if 0 <= idx < K:
                window.append(inputs[:, idx, :])  # [B, p]
            else:
                window.append(torch.zeros((B, p), device=inputs.device))
        return torch.stack(window, dim=1)  # [B, window_size, p]

    def forward(self, inputs: torch.Tensor):
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(f"Expected input shape [B, {self.order}, {self.num_vars}], got {tuple(inputs.shape)}")
        B = inputs.shape[0]
        device = inputs.device

        preds = torch.zeros((B, self.num_vars), device=device)
        coeffs_list = []

        for k in range(self.order):
            local_window = self._gather_local_window(inputs, k)  # [B, window_size, p]
            flat_window = local_window.view(B, -1)  # [B, window_size * p]
            attn_weights = self.gates[k](flat_window)  # [B, window_size]
            attn_weights = attn_weights.unsqueeze(-1)  # [B, window_size, 1]
            summary = (attn_weights * local_window).sum(dim=1)  # [B, p]

            coeffs_k = self.coeff_nets[k](summary)  # [B, p^2]
            coeffs_k = coeffs_k.view(B, self.num_vars, self.num_vars)  # [B, p, p]
            coeffs_list.append(coeffs_k.unsqueeze(1))

            lag_vec = inputs[:, k, :].unsqueeze(-1)  # [B, p, 1]
            preds = preds + torch.matmul(coeffs_k, lag_vec).squeeze(-1)

        coeffs = torch.cat(coeffs_list, dim=1)  # [B, order, p, p]
        return preds, coeffs



import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.nn as nn
import torch
import torch.nn.functional as F



class LowRankCoeffNet(nn.Module):
    def __init__(self, p, hidden_dim, num_hidden_layers, rank):
        super().__init__()
        layers = [nn.Linear(p, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.left = nn.Linear(hidden_dim, p * rank)
        self.right = nn.Linear(hidden_dim, rank * p)
        self.p = p
        self.rank = rank

    def forward(self, x):  # x: [B, p]
        h = self.net(x)  # [B, hidden_dim]
        left = self.left(h).view(-1, self.p, self.rank)  # [B, p, r]
        right = self.right(h).view(-1, self.rank, self.p)  # [B, r, p]
        coeff_matrix = torch.bmm(left, right)  # [B, p, p]
        return coeff_matrix


class GRUCoeffNet(nn.Module):
    def __init__(self, num_vars, hidden_layer_size, num_hidden_layers, latent_dim):
        super().__init__()
        self.num_vars = num_vars
        self.latent_dim = latent_dim
        
        # GRU layer(s) to process the lag sequence
        self.gru = nn.GRU(input_size=num_vars, hidden_size=latent_dim, num_layers=num_hidden_layers, batch_first=True)
        
        # Map GRU outputs (latent_dim) to flattened coeffs (num_vars^2)
        self.fc = nn.Linear(latent_dim, num_vars * num_vars)

    def forward(self, x):
        """
        x: shape (batch, order, num_vars)
        returns: coeffs tensor of shape (batch, order, num_vars, num_vars)
        """
        gru_out, _ = self.gru(x)  # (batch, order, latent_dim)
        coeffs_flat = self.fc(gru_out)  # (batch, order, num_vars^2)
        coeffs = coeffs_flat.view(x.size(0), x.size(1), self.num_vars, self.num_vars)  # reshape
        return coeffs
    
    
class SENNGC(nn.Module):
    def __init__(
        self,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        device: torch.device,
        attn_blend_init: float = 0.5,
        num_heads: int = 2,
        use_attention: str = "both",  # options: "none", "global", "self", "both"
    ):
        super().__init__()
        print(f"use_attention: {use_attention}")
        assert use_attention in {"global","self","both", "none"}, "Invalid use_attention flag"
        self.num_vars = num_vars
        self.order = order
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_attention = use_attention

        # Per-lag coefficient generators
        self.coeff_nets = nn.ModuleList()
        for _ in range(order):
            layers = [nn.Linear(self.num_vars, hidden_layer_size), nn.ReLU()]
            for _ in range(self.num_hidden_layers - 1):
                layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()]
            layers += [nn.Linear(hidden_layer_size, self.num_vars ** 2)]
            self.coeff_nets.append(nn.Sequential(*layers))
        #rank = 512  # or smaller/larger depending on tradeoff
        #self.coeff_nets = nn.ModuleList([
        #    LowRankCoeffNet(self.num_vars, self.hidden_layer_size, self.num_hidden_layers, rank)
        #    for _ in range(order)
        #])
        # Scalar lag attention
        if use_attention in {"global", "both"}:
            self.lag_attn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, self.hidden_layer_size),
                nn.ReLU(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # Self-attention over lags
        if use_attention in {"self", "both"}:
            self.self_attn = nn.MultiheadAttention(embed_dim=self.num_vars, num_heads=num_heads, batch_first=True)
            self.self_attn_score = nn.Sequential(
                nn.Linear(self.num_vars, self.hidden_layer_size),
                nn.ReLU(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # blend parameter between learned attention and uniform
        self.attn_blend = nn.Parameter(torch.tensor(attn_blend_init))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, order, p]
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(f"Expected input shape [B, {self.order}, {self.num_vars}], got {tuple(inputs.shape)}")
        B = inputs.shape[0]
        device = inputs.device

        lag_outputs = []
        scalar_logits = []
        coeffs_list = []

        for k in range(self.order):
            lag_vec = inputs[:, k, :]  # [B, p]
            coeff_flat = self.coeff_nets[k](lag_vec)  # [B, p^2]
            coeff_k = coeff_flat.view(B, self.num_vars, self.num_vars)  # [B, p, p]
            coeffs_list.append(coeff_k.unsqueeze(1))

            yk = torch.matmul(coeff_k, lag_vec.unsqueeze(-1)).squeeze(-1)  # [B, p]
            lag_outputs.append(yk.unsqueeze(1))  # [B,1,p]

            if self.use_attention in {"global", "both"}:
                score_k = self.lag_attn(lag_vec.unsqueeze(1))  # [B,1]
                scalar_logits.append(score_k)
            else:
                scalar_logits.append(torch.zeros(B, 1, device=device))

        lag_outputs = torch.cat(lag_outputs, dim=1)     # [B, order, p]
        coeffs = torch.cat(coeffs_list, dim=1)          # [B, order, p, p]
        scalar_logits = torch.cat(scalar_logits, dim=1)  # [B, order]

        # compute scalar attention weights
        if self.use_attention in {"global", "both"}:
            scalar_attn = F.softmax(scalar_logits, dim=1)  # [B, order]
        else:
            scalar_attn = torch.full((B, self.order), 0.0, device=device)

        # compute self-attention weights
        if self.use_attention in {"self", "both"}:
            self_attn_out, _ = self.self_attn(inputs, inputs, inputs, need_weights=False)  # [B, order, p]
            self_scores = self.self_attn_score(self_attn_out).squeeze(-1)  # [B, order]
            self_attn_weights = F.softmax(self_scores, dim=1)
        else:
            self_attn_weights = torch.full((B, self.order), 0.0, device=device)

        # combine according to flag
        if self.use_attention == "none":
            final_attn = torch.full((B, self.order), 1.0 / self.order, device=device)
        else:
            learned = scalar_attn + self_attn_weights  # sum contributions (handles scalar/self/both)
            learned = learned / (learned.sum(dim=1, keepdim=True) + 1e-8)  # normalize
            uniform = torch.full((B, self.order), 1.0 / self.order, device=device)
            mix = torch.sigmoid(self.attn_blend)
            final_attn = mix * learned + (1 - mix) * uniform  # [B, order]

        attn_weights = final_attn.unsqueeze(-1)  # [B, order, 1]
        preds = (attn_weights * lag_outputs).sum(dim=1)  # [B, p]

        return preds, coeffs, lag_outputs, attn_weights
