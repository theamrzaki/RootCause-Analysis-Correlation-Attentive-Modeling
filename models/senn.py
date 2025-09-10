import torch.nn as nn
import torch

from layers.SimpleGNN import AttentionCoeffGNN, AttentionCoeffGNN_multihead, AttentionCoeffGNN_multihead_fixed, TemporalGNN, RecurrentAttentionCoeffGNN, RecurrentAttentionGNN_Attn,RecurrentAttentionCoeffGNN_chunks
from layers.trend_seasonal import TS_Model

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int,
                 args: dict,  device: torch.device):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.
        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        super(SENNGC, self).__init__()
        self.args = args

        if args["coeff_architecture"] == "deep_mlp":
            # Networks for amortising generalised coefficient matrices.
            self.coeff_nets = nn.ModuleList()

            ## Instantiate coefficient networks
            for k in range(order):
                modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
                if num_hidden_layers > 1:
                    for j in range(num_hidden_layers - 1):
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
                self.coeff_nets.append(nn.Sequential(*modules))

        elif args["coeff_architecture"] == "gnn_attention":
            self.rank = 20
            self.coeff_nets = nn.ModuleList()
            for k in range(order):
                self.coeff_nets.append(AttentionCoeffGNN(num_vars=num_vars, rank=self.rank ))
        
        elif args["coeff_architecture"] == "AttentionCoeffGNN_multihead":
            # time projection
            self.rank = 5
            self.coeff_nets = nn.ModuleList()
            for k in range(order):
                self.coeff_nets.append(AttentionCoeffGNN_multihead(num_vars=num_vars, rank=self.rank))

            # freq projection
            #self.coeff_nets_freq = nn.ModuleList()              
            #for k in range(order):
            #    self.coeff_nets_freq.append(AttentionCoeffGNN_multihead(num_vars=num_vars, rank=self.rank))
        
        elif args["coeff_architecture"] == "AttentionCoeffGNN_multihead_fixed":
            self.rank = 20
            self.coeff_nets = nn.ModuleList()
            for k in range(order):
                self.coeff_nets.append(AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=self.rank))


        elif args["coeff_architecture"] == "cross_time_freq":
            self.rank = 20
            self.coeff_nets_time = nn.ModuleList()
            for k in range(order):
                self.coeff_nets_time.append(AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=self.rank))

            # freq branch (shared across all orders)
            self.coeff_nets_freq = AttentionCoeffGNN_multihead_fixed(
                num_vars=num_vars, rank=self.rank
            )

            d_q = 16  # query dimension
            d_k = 16  # key dimension
            d_v = 32  # value dimension
            self.time_Q = nn.Linear(num_vars, d_q)
            self.time_K = nn.Linear(num_vars, d_k)
            self.time_V = nn.Linear(num_vars, d_v)
            self.freq_Q = nn.Linear(num_vars, d_q)
            self.freq_K = nn.Linear(num_vars, d_k)
            self.freq_V = nn.Linear(num_vars, d_v)
            self.time_out = nn.Linear(d_v, num_vars)
            self.freq_out = nn.Linear(d_v, num_vars)

            total_params = sum(p.numel() for p in self.coeff_nets_time.parameters() if p.requires_grad)
            total_freq_params = sum(p.numel() for p in self.coeff_nets_freq.parameters() if p.requires_grad)
            print(f"Total parameters for temporal : {total_params+total_freq_params}")
        

        elif args["coeff_architecture"] == "cross_attention_single_coeff_network":
            self.rank = 20

            # single time branch coefficient (shared across all lags)
            self.coeff_net_time = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=self.rank)

            # frequency branch coefficient (shared across all lags)
            self.coeff_net_freq = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=self.rank)

            # cross-attention projections
            self.d_q = 16  # query dimension
            self.d_k = 16  # key dimension
            self.d_v = 32  # value dimension

            self.time_Q = nn.Linear(num_vars, self.d_q)
            self.time_K = nn.Linear(num_vars, self.d_k)
            self.time_V = nn.Linear(num_vars, self.d_v)

            self.freq_Q = nn.Linear(num_vars, self.d_q)
            self.freq_K = nn.Linear(num_vars, self.d_k)
            self.freq_V = nn.Linear(num_vars, self.d_v)

            self.time_out = nn.Linear(self.d_v, num_vars)
            self.freq_out = nn.Linear(self.d_v, num_vars)

            # --- Attention GRU for temporal correlation ---
            hidden_dim = 64
            self.attn_gru = nn.GRU(input_size=num_vars*num_vars, hidden_size=hidden_dim, batch_first=True)
            self.coeff_adjust_proj = nn.Linear(hidden_dim, num_vars*num_vars)  # project to next-step preds

            total_params = sum(p.numel() for p in self.coeff_net_time.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.coeff_net_freq.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.time_Q.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.time_K.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.time_V.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.freq_Q.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.freq_K.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.freq_V.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.time_out.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.freq_out.parameters() if p.requires_grad)

            print(f"Total parameters for cross_attention_single_coeff_network: {total_params}")

        elif args["coeff_architecture"] == "cross_attention_single_coeff_network_multi_head":
            self.rank = 20
            self.n_heads = 4  # number of attention heads
            self.d_model = 32  # total feature dimension for attention

            # --- single time branch coefficient (shared across all lags) ---
            self.coeff_net_time = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=self.rank)

            # --- frequency branch coefficient (shared across all lags) ---
            self.coeff_net_freq = AttentionCoeffGNN_multihead(num_vars=num_vars, rank=self.rank)

            # --- cross-attention projections ---
            self.time_Q = nn.Linear(num_vars, self.d_model)
            self.time_K = nn.Linear(num_vars, self.d_model)
            self.time_V = nn.Linear(num_vars, self.d_model)

            self.freq_Q = nn.Linear(num_vars, self.d_model)
            self.freq_K = nn.Linear(num_vars, self.d_model)
            self.freq_V = nn.Linear(num_vars, self.d_model)

            self.time_out = nn.Linear(self.d_model, num_vars)
            self.freq_out = nn.Linear(self.d_model, num_vars)

            # --- learnable skip connection weight ---
            self.alpha = nn.Parameter(torch.tensor(0.5))

            # --- optional layer norm ---
            self.ln = nn.LayerNorm(self.d_model)

            # --- Positional encoding ---
            self.pos_emb = nn.Parameter(torch.randn(order, self.d_model))

            # --- Total parameters count ---
            total_params = sum(p.numel() for p in self.coeff_net_time.parameters() if p.requires_grad)
            total_params += sum(p.numel() for p in self.coeff_net_freq.parameters() if p.requires_grad)
            for m in [self.time_Q, self.time_K, self.time_V, self.freq_Q, self.freq_K, self.freq_V, self.time_out, self.freq_out]:
                total_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

            print(f"Total parameters for cross_attention_single_coeff_network_multi_head: {total_params}")

        elif args["coeff_architecture"] == "TemporalGNN":
            self.rank = 20
            #### 🚨 only ONE temporal model, not one per lag
            ###self.temporal = TemporalGNN(num_vars=num_vars, rank=self.rank, hidden_dim=hidden_layer_size)
            
            self.coeff_net = RecurrentAttentionCoeffGNN_chunks(
                num_vars=num_vars,
                rank=self.rank,
                order=order,
                device=device
            )
            total_params = sum(p.numel() for p in self.coeff_net.parameters() if p.requires_grad)
            print(f"Total parameters for temporal : {total_params}")


        elif args["coeff_architecture"] == "TemporalGNN_Attention":
            self.rank = 51
            
            self.coeff_net = RecurrentAttentionGNN_Attn(
                num_vars=num_vars,
                rank=self.rank,
                order=order,
                device=device,
                hidden_dim = args.get("outer_hidden_dim", 64),  # default to 64 if not specified
                num_heads = args.get("outer_heads_num", 4),  # default to 4 heads if not specified
                attention_heads = args.get("num_attention_heads", 4),  # default to 4 heads if not specified
                attention_dim = args.get("attention_dim", 64)  # default to 64 if not specified
            )
            total_params = sum(p.numel() for p in self.coeff_net.parameters() if p.requires_grad)
            print(f"Total parameters for temporal : {total_params}")

        elif args["coeff_architecture"] == "trend_seasonal":
            self.coeff_net = TS_Model( seq_len=order, num_nodes=num_vars, d_model=64)
            total_params = sum(p.numel() for p in self.coeff_net.parameters() if p.requires_grad)
            print(f"Total parameters for temporal : {total_params}")


        if args["coeff_architecture"] not in  ["ht","epsilon_diagnosis","rcd","TemporalGNN","cross_time_freq","cross_attention_single_coeff_network","TemporalGNN_Attention","trend_seasonal","rcd"]:
            total_params = sum(p.numel() for net in self.coeff_nets for p in net.parameters())
            print(f"Total parameters for {order} lags: {total_params}")
        
        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device


    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward_normal(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            # coeffs[:, k, :, :] = coeffs_k
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs, None
    

    def forward_gnn(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)            
            # coeffs[:, k, :, :] = coeffs_k
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs
    
    
    def forward_cross_time_freq(self, inputs: torch.Tensor, corr_type="dual_guided"):
        """
        Forward pass for cross time-frequency AttentionCoeffGNN.

        Args:
            inputs: Tensor of shape (B, order, num_vars)
            corr_type: str, type of combination:
                - "simple": linear correlation
                - "weighted": learnable weighted combination
                - "cross_attention": time branch attends to freq branch
        Returns:
            preds: Tensor (B, num_vars)
            coeffs_combined: Tensor (B, order, num_vars, num_vars)
        """
        B, order, p = inputs.shape
        device = inputs.device

        # --- Time branch ---
        coeffs_time = []
        for k in range(order):
            coeff_t = self.coeff_nets_time[k](inputs[:, k, :])
            coeff_t = coeff_t.view(B, p, p)
            coeffs_time.append(coeff_t)
        coeffs_time = torch.stack(coeffs_time, dim=1)  # (B, order, p, p)

        # --- Frequency branch ---
        inputs_freq = torch.fft.rfft(inputs, dim=1).real  # (B, freq_bins, p)
        inputs_freq = inputs_freq.mean(dim=1)             # aggregate to (B, p)
        coeff_freq = self.coeff_nets_freq(inputs_freq).view(B, p, p)

        # --- Combine time & freq ---
        coeffs_combined = torch.zeros_like(coeffs_time)
        for k in range(order):
            c_time = coeffs_time[:, k, :, :]

            if corr_type == "simple":
                # linear correlation
                coeffs_combined[:, k, :, :] = (c_time @ coeff_freq.T) / p

            elif corr_type == "weighted":
                alpha = getattr(self, "alpha", 0.5)
                coeffs_combined[:, k, :, :] = alpha * c_time + (1 - alpha) * coeff_freq

            elif corr_type == "cross_attention":
                # scaled matmul attention + softmax
                attn_scores = torch.matmul(c_time, coeff_freq) / (p ** 0.5)
                attn_scores = torch.softmax(attn_scores, dim=-1)
                coeffs_combined[:, k, :, :] = c_time + torch.matmul(attn_scores, coeff_freq)
            
            elif corr_type == "dual_attention":
                # Time → Freq attention
                attn_time2freq = torch.matmul(c_time, coeff_freq) / (p ** 0.5)
                attn_time2freq = torch.softmax(attn_time2freq, dim=-1)
                time2freq = torch.matmul(attn_time2freq, coeff_freq)

                # Freq → Time attention
                attn_freq2time = torch.matmul(coeff_freq, c_time) / (p ** 0.5)
                attn_freq2time = torch.softmax(attn_freq2time, dim=-1)
                freq2time = torch.matmul(attn_freq2time, c_time)

                # Combine dual attention with residual
                coeffs_combined[:, k, :, :] = c_time + time2freq + freq2time

            elif corr_type == "dual_guided":
                # --- Linear projections for Time branch ---
                Q_time = self.time_Q(c_time)   # (B, p, d_q)
                K_time = self.time_K(c_time)   # (B, p, d_k)
                V_time = self.time_V(c_time)   # (B, p, d_v)

                # --- Linear projections for Frequency branch ---
                Q_freq = self.freq_Q(coeff_freq)  # (B, p, d_q)
                K_freq = self.freq_K(coeff_freq)  # (B, p, d_k)
                V_freq = self.freq_V(coeff_freq)  # (B, p, d_v)

                # --- Frequency -> Time attention ---
                attn_f2t = torch.softmax(Q_time @ K_freq.transpose(-2, -1) / math.sqrt(K_freq.size(-1)), dim=-1)
                guided_time = attn_f2t @ V_freq  # (B, p, d_v)

                # --- Time -> Frequency attention ---
                attn_t2f = torch.softmax(Q_freq @ K_time.transpose(-2, -1) / math.sqrt(K_time.size(-1)), dim=-1)
                guided_freq = attn_t2f @ V_time  # (B, p, d_v)

                # --- Project back to original coeff dimension (p x p) ---
                guided_time_proj = self.time_out(guided_time)   # (B, p, p)
                guided_freq_proj = self.freq_out(guided_freq)   # (B, p, p)

                # --- Fuse both with skip connections ---
                coeffs_combined[:, k, :, :] = c_time + coeff_freq + guided_time_proj + guided_freq_proj

            
            else:
                raise ValueError(f"Unknown corr_type: {corr_type}")

        # --- Predictions ---
        preds = torch.zeros((B, p), device=device)
        for k in range(order):
            preds += (coeffs_combined[:, k, :, :] @ inputs[:, k, :].unsqueeze(-1)).squeeze(-1)

        return preds, coeffs_combined


    def forward_cross_attention_single_coeff_network(self, inputs: torch.Tensor):
        """
        Forward pass for cross-domain time-frequency attention with single time coeff.

        Args:
            inputs: Tensor of shape (B, order, num_vars)

        Returns:
            preds: Tensor of shape (B, num_vars)
            coeffs_combined: Tensor of shape (B, order, num_vars, num_vars)
        """
        B, order, p = inputs.shape
        device = inputs.device

        # --- Step 1: Compute single time/frequency coefficients ---
        coeff_time = self.coeff_net_time(inputs[:, 0, :]).view(B, p, p)
        inputs_freq = torch.fft.rfft(inputs, dim=1).real.mean(dim=1)  # aggregate frequency
        coeff_freq = self.coeff_net_freq(inputs_freq).view(B, p, p)

        # --- Step 2: Cross-attention ---
        Q_time = self.time_Q(coeff_time)
        K_time = self.time_K(coeff_time)
        V_time = self.time_V(coeff_time)

        Q_freq = self.freq_Q(coeff_freq)
        K_freq = self.freq_K(coeff_freq)
        V_freq = self.freq_V(coeff_freq)

        attn_weights_tf = torch.softmax(Q_time @ K_freq.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        guided_time = attn_weights_tf @ V_freq

        attn_weights_ft = torch.softmax(Q_freq @ K_time.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        guided_freq = attn_weights_ft @ V_time

        fused_time = self.time_out(guided_time) + coeff_time
        fused_freq = self.freq_out(guided_freq) + coeff_freq

        # --- Step 3: Combine time-frequency features as GRU input ---
        # Flatten per sample: (B, p*p) and repeat for each order
        fused_flat = (fused_time + fused_freq).view(B, 1, -1).repeat(1, order, 1)  # (B, order, p*p)

        # --- Step 4: Attention-GRU over order dimension ---
        h_seq, h_final = self.attn_gru(fused_flat)  # h_seq: (B, order, hidden_dim)
        h_last = h_final[-1]                         # (B, hidden_dim)

        # Project hidden state to adjust coefficients
        coeff_adjust = self.coeff_adjust_proj(h_last).view(B, 1, p, p)  # (B, 1, p, p)
        coeffs_combined = fused_flat.view(B, order, p, p) + coeff_adjust  # broadcast across order

        # --- Step 5: Compute predictions ---
        preds = torch.zeros((B, p), device=device)
        for k in range(order):
            preds += torch.bmm(coeffs_combined[:, k, :, :], inputs[:, k, :].unsqueeze(-1)).squeeze(-1)

        return preds, coeffs_combined

    def forward_temporal(self, inputs: torch.Tensor):
        """
        inputs: (B, order, num_vars)
        TemporalGNN processes the entire lag sequence recurrently.
        """
        preds, coeffs, attn_weights = self.coeff_net(inputs)  # let TemporalGNN return preds + coeffs
        return preds, coeffs, attn_weights
    
    def forward(self, inputs: torch.Tensor):
        if self.args["coeff_architecture"] == "deep_mlp":
            return self.forward_normal(inputs)
        elif self.args["coeff_architecture"] == "gnn_attention" or self.args["coeff_architecture"] == "AttentionCoeffGNN_multihead" or self.args["coeff_architecture"] == "AttentionCoeffGNN_multihead_fixed":                                                                                                                                                    
            return self.forward_gnn(inputs)
        elif self.args["coeff_architecture"] in ["TemporalGNN", "TemporalGNN_Attention","trend_seasonal"]:
            return self.forward_temporal(inputs)
        elif self.args["coeff_architecture"] == "cross_time_freq":
            return self.forward_cross_time_freq(inputs)
        elif self.args["coeff_architecture"] == "cross_attention_single_coeff_network":
            return self.forward_cross_attention_single_coeff_network(inputs)
        elif self.args["coeff_architecture"] == "trend_seasonal":
            return self.forward_trend_seasonal(inputs)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat

# ---------- Mamba Blocks (from your provided code) ----------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class ResidualBlock(nn.Module):
    def __init__(self, configs, d_inner, dt_rank):
        super().__init__()
        self.mixer = MambaBlock(configs, d_inner, dt_rank)
        self.norm = RMSNorm(configs.d_model)
    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class MambaBlock(nn.Module):
    def __init__(self, configs, d_inner, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.in_proj = nn.Linear(configs.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=configs.d_conv,
                                padding=configs.d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + configs.d_ff * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, configs.d_ff + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, configs.d_model, bias=False)

        self.configs = configs

    def forward(self, x):  # x: [B, L, d_model]
        b, l, d = x.shape
        x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())       # [d_in, n]
        D = self.D.float()                        # [d_in]
        x_dbl = self.x_proj(x)                    # [B, L, dt_rank + 2*n]
        delta, B, C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))   # [B, L, d_in]
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA  = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaBu = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")
        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaBu[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)
        y = torch.stack(ys, dim=1)               # [B, L, d_in]
        y = y + u * D
        return y

# ---------- Efficient Spatial Encoder (shared across lags) ----------
class SpatialEncoder(nn.Module):
    """
    Lightweight 'spatial' mixing shared across all lags.
    Efficient: O(p*d) linear + gated residual; optional norm.
    """
    def __init__(self, num_vars: int, d_model: int):
        super().__init__()
        self.lin1 = nn.Linear(num_vars, d_model, bias=True)
        self.lin2 = nn.Linear(d_model, d_model, bias=True)
        self.gate = nn.Linear(num_vars, d_model, bias=True)
        self.norm = RMSNorm(d_model)

    def forward(self, x_t):  # x_t: [B, p]
        h = F.gelu(self.lin1(x_t))               # [B, d_model]
        h = self.lin2(h)                         # [B, d_model]
        g = torch.sigmoid(self.gate(x_t))        # [B, d_model]
        h = self.norm(h * g)                     # gated residual-style
        return h                                  # [B, d_model]

# ---------- Spatio-Temporal Mamba VAR (most efficient) ----------
class ST_Mamba_VAR(nn.Module):
    """
    Most efficient spatio-temporal approach:
    - ONE shared SpatialEncoder for all lags (no per-lag duplication)
    - Mamba over lag sequence (temporal)
    - Low-rank or full coefficient decoder per lag
    """
    def __init__(self, num_vars: int, order: int, mamba_configs,
                 d_model: int, use_low_rank: bool = True, rank: int = 4):
        super().__init__()
        self.num_vars = num_vars
        self.order = order
        self.d_model = d_model
        self.use_low_rank = use_low_rank
        self.rank = rank

        # Spatial encoder shared across all lags
        self.spatial = SpatialEncoder(num_vars, d_model)

        # Temporal encoder (stack of residual Mamba blocks)
        d_inner = mamba_configs.d_model * mamba_configs.expand
        dt_rank = math.ceil(mamba_configs.d_model / 16)
        self.blocks = nn.ModuleList([ResidualBlock(mamba_configs, d_inner, dt_rank)
                                     for _ in range(mamba_configs.e_layers)])
        self.post_norm = RMSNorm(mamba_configs.d_model)

        # Decoder: map each time step to coefficients
        if use_low_rank:
            # Output 2 * p * r (U and V)
            self.proj = nn.Linear(d_model, 2 * num_vars * rank, bias=False)
        else:
            # Output p^2
            self.proj = nn.Linear(d_model, num_vars * num_vars, bias=False)

    def forward(self, inputs: torch.Tensor):
        """
        inputs: [B, K, p] where K = order
        returns:
          preds:  [B, p]
          coeffs: [B, K, p, p]
        """
        B, K, p = inputs.shape
        assert K == self.order and p == self.num_vars, "inputs must be [B, order, num_vars]"

        # Spatial encode each lag (shared weights)
        # h_seq: [B, K, d_model]
        h_seq = torch.stack([self.spatial(inputs[:, k, :]) for k in range(K)], dim=1)

        # Temporal Mamba over sequence
        for block in self.blocks:
            h_seq = block(h_seq)
        h_seq = self.post_norm(h_seq)  # [B, K, d_model]

        # Decode coefficients per lag
        if self.use_low_rank:
            out = self.proj(h_seq)                                   # [B, K, 2*p*r]
            U_flat, V_flat = out.split(self.num_vars * self.rank, dim=-1)
            U = U_flat.view(B, K, self.num_vars, self.rank)          # [B, K, p, r]
            V = V_flat.view(B, K, self.num_vars, self.rank)          # [B, K, p, r]
            coeffs = torch.matmul(U, V.transpose(-1, -2))            # [B, K, p, p]
        else:
            coeffs = self.proj(h_seq).view(B, K, self.num_vars, self.num_vars)  # [B, K, p, p]

        # Prediction: sum_k A_k x_k
        preds = torch.einsum('bkpq,bkq->bp', coeffs, inputs)          # [B, p]
        return preds, coeffs

# ---------- How to plug into your SENNGC ----------
class SENNGC___(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int,
                 args: dict, device: torch.device, mamba_configs=None):
        super().__init__()
        self.args = args
        self.num_vars = num_vars
        self.order = order
        self.device = device

        arch = args.get("coeff_architecture", "deep_mlp")

        if arch == "deep_mlp":
            self.coeff_nets = nn.ModuleList()
            for _ in range(order):
                layers = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
                for _ in range(max(0, num_hidden_layers - 1)):
                    layers.append(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                layers.append(nn.Sequential(nn.Linear(hidden_layer_size, num_vars * num_vars), nn.Tanh()))
                self.coeff_nets.append(nn.Sequential(*layers))

        elif arch == "st_mamba":
            from types import SimpleNamespace

            # Minimal Mamba configs adapted to SENNGC
            mamba_configs = SimpleNamespace(
                task_name='long_term_forecast',
                pred_len=1,
                enc_in=num_vars,     # num_vars
                c_out=num_vars,      # num_vars
                d_model=64,
                expand=2,
                embed='fixed',
                freq='h',
                dropout=0.1,
                e_layers=2,    # number of residual Mamba blocks
                d_conv=3,
                d_ff=16
            )

            use_low_rank = args.get("use_low_rank", True)
            rank = args.get("rank", 4)
            d_model = mamba_configs.d_model
            self.st_mamba = ST_Mamba_VAR(num_vars, order, mamba_configs,
                                         d_model=d_model, use_low_rank=use_low_rank, rank=rank)

        else:
            raise ValueError(f"Unknown coeff_architecture: {arch}")

    def forward(self, inputs: torch.Tensor):
        if self.args.get("coeff_architecture") == "deep_mlp":
            B = inputs.size(0)
            preds = torch.zeros((B, self.num_vars), device=inputs.device)
            coeffs_all = []
            for k in range(self.order):
                A_k = self.coeff_nets[k](inputs[:, k, :]).view(B, self.num_vars, self.num_vars)
                coeffs_all.append(A_k)
                preds = preds + torch.matmul(A_k, inputs[:, k, :].unsqueeze(-1)).squeeze(-1)
            coeffs = torch.stack(coeffs_all, dim=1)  # [B, K, p, p]
            return preds, coeffs

        elif self.args.get("coeff_architecture") == "st_mamba":
            return self.st_mamba(inputs)

        else:
            raise ValueError("Unsupported architecture in forward()")
