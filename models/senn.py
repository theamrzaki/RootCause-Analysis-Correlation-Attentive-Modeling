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

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device, use_attention: bool = True):
        super().__init__()
        self.num_vars = num_vars  # p
        self.order = order        # K
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_attention = use_attention

        # Per-lag coefficient generators: x_{t-k} -> flattened p x p matrix
        self.coeff_nets = nn.ModuleList()
        for _ in range(order):
            layers = [nn.Linear(self.num_vars, hidden_layer_size), nn.ReLU()]
            for _ in range(self.num_hidden_layers - 1):
                layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()]
            layers += [nn.Linear(hidden_layer_size, self.num_vars ** 2)]
            self.coeff_nets.append(nn.Sequential(*layers))

        # Attention over lags: score each lag vector to get weights
        self.lag_attn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1),  # local context across variables
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # aggregate to single feature per channel
            nn.Flatten(),             # [B, 16]
            nn.Linear(16, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, 1)  # scalar score per lag
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_original_output(self, inputs: torch.Tensor):
        # inputs: [B, order, p]
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(f"Expected input shape [B, {self.order}, {self.num_vars}], got {tuple(inputs.shape)}")
        B = inputs.shape[0]
        device = inputs.device

        lag_outputs = []   # per-lag y^{(k)}: [B, 1, p]
        attn_logits = []  # [B, order]
        coeffs_list = []  # to build coeffs tensor

        for k in range(self.order):
            lag_vec = inputs[:, k, :]  # [B, p]
            coeff_flat = self.coeff_nets[k](lag_vec)  # [B, p^2]
            coeff_k = coeff_flat.view(B, self.num_vars, self.num_vars)  # [B, p, p]
            coeffs_list.append(coeff_k.unsqueeze(1))  # for stacking later

            yk = torch.matmul(coeff_k, lag_vec.unsqueeze(-1)).squeeze(-1)  # [B, p]
            lag_outputs.append(yk.unsqueeze(1))  # [B,1,p]

            score_k = self.lag_attn(lag_vec)  # [B,1]
            attn_logits.append(score_k)

        lag_outputs = torch.cat(lag_outputs, dim=1)     # [B, order, p]
        attn_logits = torch.cat(attn_logits, dim=1)     # [B, order]
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, order, 1]

        # Weighted sum across lags
        preds = (attn_weights * lag_outputs).sum(dim=1)  # [B, p]

        coeffs = torch.cat(coeffs_list, dim=1)  # [B, order, p, p]
        return preds, coeffs

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, order, p]
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(...)
        B = inputs.shape[0]

        lag_outputs = []   # per-lag contributions y^{(k)}: [B, 1, p]
        attn_logits = []   # raw scores before softmax: [B, order]
        coeffs_list = []   # will become [B, order, p, p]

        for k in range(self.order):
            lag_vec = inputs[:, k, :]  # [B, p]
            coeff_flat = self.coeff_nets[k](lag_vec)  # [B, p^2]
            coeff_k = coeff_flat.view(B, self.num_vars, self.num_vars)  # [B, p, p]
            coeffs_list.append(coeff_k.unsqueeze(1))

            yk = torch.matmul(coeff_k, lag_vec.unsqueeze(-1)).squeeze(-1)  # [B, p]
            lag_outputs.append(yk.unsqueeze(1))  # [B,1,p]

            if self.use_attention:
                x_for_attn = lag_vec.unsqueeze(1)  # [B, 1, p] for Conv1d
                score_k = self.lag_attn(x_for_attn)  # [B,1]
                attn_logits.append(score_k)
            else:
                attn_logits.append(torch.zeros((B, 1), device=inputs.device))

        lag_outputs = torch.cat(lag_outputs, dim=1)     # [B, order, p]
        attn_logits = torch.cat(attn_logits, dim=1)     # [B, order]
        if self.use_attention:
            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, order, 1]
        else:
            attn_weights = torch.full((B, self.order, 1), 1.0 / self.order, device=inputs.device)


        preds = (attn_weights * lag_outputs).sum(dim=1)  # [B, p]
        coeffs = torch.cat(coeffs_list, dim=1)           # [B, order, p, p]

        return preds, coeffs, lag_outputs, attn_weights
