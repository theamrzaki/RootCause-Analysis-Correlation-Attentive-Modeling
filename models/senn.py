import torch.nn as nn
import torch


from layers.vlinear_arch import vlinear
from layers.cLSTM import cLSTM
from models.causalrca import causalrca
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

        if args["coeff_architecture"] == "deep_mlp" or args["coeff_architecture"] == "GVAR":
            # Networks for amortising generalised coefficient matrices.
            self.coeff_nets = nn.ModuleList()

            ## Instantiate coefficient networks
            for k in range(order-1):
                modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
                if num_hidden_layers > 1:
                    for j in range(num_hidden_layers - 1):
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
                self.coeff_nets.append(nn.Sequential(*modules))

        elif args["coeff_architecture"] == "causalrca":
            import numpy as np
            num_nodes = args.get("num_vars")
            adj_A = np.zeros((num_nodes, num_nodes))

            self.coeff_net = causalrca(
                num_vars=num_nodes,                       # SAME as encoder
                order=order,                              # same as original order
                adj_A=adj_A,                              # SAME adjacency input
                hidden_dim=args.get("outer_hidden_dim", 64),  # SAME hidden dimension
                n_hid=args.get("outer_hidden_dim", 64),   # same as input embedding dim
                n_out=1,   # SAME latent dimensionality
                tol=args.get("lr", 64),                   # keep default
                device=device,                            # same device
            )

            total_params = sum(p.numel() for p in self.coeff_net.parameters() if p.requires_grad)
            print(f"Total parameters for temporal : {total_params}")
     
        if args["coeff_architecture"] not in  ["ht","epsilon_diagnosis","rcd","TemporalGNN","cross_time_freq","cross_attention_single_coeff_network","TemporalGNN_Attention","trend_seasonal","rcd","TemporalGNN_Attention_fourier","TemporalGNN_Attention_crossattn","TemporalGNN_Attention_crossattn_Legendre","TemporalGNN_Attention_crossattn_enhanced","causalrca","cuts_mlp","cuts_lstm","GVAR","vlinear","nsigma","baro","circa","torai","cLSTM"]:
            total_params = sum(p.numel() for net in self.coeff_nets for p in net.parameters())
            print(f"Total parameters for {order} lags: {total_params}")
        
        if args["coeff_architecture"] in "vlinear":
            self.coeff_net = vlinear(
                num_vars=num_vars,
                order=order,
                device=device,
                options = args  # default to None if not specified
            )

        if args["coeff_architecture"] == "cLSTM":
            self.coeff_net = cLSTM(num_vars, hidden_layer_size)

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
        # Shape check using your preferred style
        if inputs.shape[1] != (self.order - 1):
            print(f"WARNING: Expected {self.order-1} steps, got {inputs.shape[1]}")

        coeffs = [] # Using a list is faster than torch.cat in a loop
        preds = torch.zeros((inputs.shape[0], self.num_vars), device=self.device)
        
        for k in range(self.order - 1):
            coeffs_k = self.coeff_nets[k](inputs[:, k, :])
            coeffs_k = coeffs_k.view(inputs.shape[0], self.num_vars, self.num_vars)
            
            coeffs.append(coeffs_k.unsqueeze(1))
            
            # Prediction: A_k * x_k
            # Squeeze(2) ensures we go from [BS, 30, 1] back to [BS, 30]
            preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(2)).squeeze(2)

        return preds, torch.cat(coeffs, dim=1), None
       
    
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
    
    def forward_temporal_causalrca(self, inputs: torch.Tensor):
        """
        inputs: (B, order, num_vars)
        TemporalGNN processes the entire lag sequence recurrently.
        """
        preds, coeffs, attn_weights, aux_vars = self.coeff_net(inputs)  # let TemporalGNN return preds + coeffs
        return preds, coeffs, (attn_weights, aux_vars)
    
    def forward_simple_nextstep(self, inputs: torch.Tensor):
        """
        Simple forward pass for next-step prediction without returning coefficients.
        """
        preds,_ = self.coeff_net(inputs)
        return preds
    
    def forward(self, inputs: torch.Tensor):
        if self.args["coeff_architecture"] == "deep_mlp" or self.args["coeff_architecture"] == "GVAR":
            return self.forward_normal(inputs)

        elif self.args["coeff_architecture"] in ["vlinear"]:
            return self.forward_temporal(inputs)
        elif self.args["coeff_architecture"] == "causalrca":
            return self.forward_temporal_causalrca(inputs)
        
        elif self.args["coeff_architecture"] == "cLSTM":
            return self.forward_simple_nextstep(inputs), None, None
