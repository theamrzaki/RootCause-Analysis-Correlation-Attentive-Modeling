import torch
import torch.nn as nn
import numpy as np
import os
from numpy.linalg import eigh
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
import os
from numpy.linalg import eigh

class OrthTransform(nn.Module):
    def __init__(self, dataset_obj, save_path, time_lag, device):
        super().__init__()
        self.device = device
        self.time_lag = time_lag
        # Use the data_dir from the dataset object to define the matrix path
        # This ensures the Q matrix is tied to the specific dataset processing
        filename = "swat_q_matrix"
        self.matrix_path = os.path.join(save_path, f'{filename}_lag{time_lag}.npy')
        
        if not os.path.isfile(self.matrix_path):
            print(f"Matrix not found at {self.matrix_path}. Computing from dataset memory...")
            # We pass the pre-loaded normal data from the dataset object
            q_mat = self._compute_q_matrix(dataset_obj.data_dict['x_n_list'], time_lag, save_path)
        else:
            print(f"Loading precomputed Q matrix from {self.matrix_path}")
            q_mat = np.load(self.matrix_path)
            
        self.register_buffer('Q', torch.from_numpy(q_mat.astype(np.float32)))

    def _compute_q_matrix(self, train_data, time_lag, save_path):
        """
        Computes Q based on the pre-processed 'x_n_list' (Samples, Window, Vars)
        """
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        # train_data shape is [Samples, Window, Vars]
        # We need to flatten the temporal aspect for covariance or use the windowed samples
        # For SWaT, we usually compute the temporal covariance across the window
        S, W, V = train_data.shape
        
        sigma_list = []
        for feature_idx in range(V):
            # Extract the specific feature across all windows
            # Shape: [Samples, Window]
            feat_windows = train_data[:, :, feature_idx]
            
            # Compute covariance across the temporal dimension (Window size)
            cov = np.cov(feat_windows.T) 
            diag = np.diag(cov)
            
            if (diag < 1e-6).any(): continue
                
            cov = cov / (np.sqrt(np.outer(diag, diag)) + 1e-9) 
            sigma_list.append(cov)

        if not sigma_list:
            raise ValueError("No valid features found to compute OrthTransform. Check data variance.")

        sigma_mean = np.mean(sigma_list, axis=0)
        eigenvalues, eigenvectors = eigh(sigma_mean)
        
        # Sort descending
        q_mat = np.flip(eigenvectors.T, axis=0)
        
        np.save(self.matrix_path, q_mat)
        return q_mat

    def forward(self, x, disable_orth=False):
        # x: [Batch, Window, Channels] (e.g., 20, 36, 51)
        target_len = self.Q.shape[0] # 1000
        current_len = x.shape[1]    # 36
        disable_orth = False
        if disable_orth:
            # IDENTITY MODE: Pure temporal pass-through
            # No spectral mixing happens here.
            return x.transpose(1, 2)
        
        # --- ORTHOGONAL MODE ---
        if current_len < target_len:
            # Pad the temporal dimension to match the basis size
            padding = (0, 0, target_len - current_len, 0)
            x = torch.nn.functional.pad(x, padding, "constant", 0)
        
        # Apply basis projection: [B, W, C] * [W_new, W] -> [B, W_new, C]
        out = torch.einsum('bwc, vw -> bvc', x, self.Q)
        
        # Return the relevant window transposed to [Batch, Channels, Window]
        return out[:, -current_len:, :].transpose(1, 2)

    def inverse(self, x_orth, disable_orth=False):
        disable_orth = False
        # x_orth: [Batch, Channels, Current_W]
        if disable_orth:
            return x_orth.transpose(1, 2)

        # --- ORTHOGONAL MODE ---
        current_w = x_orth.shape[2] 
        # Project back using the top coefficients
        Q_sliced = self.Q[:current_w, :current_w]
        
        # [B, C, W] * [W, W] -> [B, C, W]
        out = torch.einsum('bcw, wv -> bcv', x_orth, Q_sliced)
        
        return out.transpose(1, 2)
  

class vlinear(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=128, device="cpu", options=None):#128 for SMD
        super().__init__()
        self.num_vars = num_vars  
        self.order = order*1  -1      
        self.device = device
        
        self.orth_transformer = options.get('orth_transformer') 
        
        # 1. Delta Biases (Faithful to Model logic)
        # These act as "Learned Context" for the orthogonal domain
        # delta1: [1, Channels, 1, Lag]
        self.delta_latent1 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.delta_latent2 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))

        # Projection to match the output 'order'
        self.bias_proj = nn.Linear(hidden_dim, self.order)
        
        # 2. Updated Embeddings 
        # In the Model code, embeddings are often 1D and expanded
        #self.embeddings = nn.Parameter(torch.randn(1, hidden_dim))
        self.embeddings = nn.Parameter(torch.randn(1, num_vars, 1, hidden_dim))
        # 3. Projection matching the Model's logic
        #self.temporal_proj = nn.Linear(self.order, hidden_dim)
        self.temporal_proj = nn.Linear(1, hidden_dim)
        #self.temporal_proj = nn.Sequential(
        #    nn.Linear(1, hidden_dim // 2),
        #    nn.GELU(),
        #    nn.Linear(hidden_dim // 2, hidden_dim)
        #)
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, self.order) 
        )
        #self.revin = RevIN(num_vars)
        self.use_temporal_mixer = options.get('temporal_mixer', False)
        if self.use_temporal_mixer:
            self.temporal_mixer = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim
            )
        

    def forward(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape # [B, Window, Sensors] e.g., [B, 2, 51]
        
        # --- 1. Step into Orthogonal Domain ---
        if self.orth_transformer is None:
            x_orth = inputs.transpose(1, 2) # [B, 51, 2]
        else:
            x_orth = self.orth_transformer(inputs) # [B, 51, 2]
        
        # --- 2. Apply Delta1 Latent Bias ---
        # Project delta_latent1 [1, P, H] -> [1, P, Order]
        # Then unsqueeze to [1, P, 1, Order] for broadcasting
        d1 = self.bias_proj(self.delta_latent1).unsqueeze(-2) 
        x_orth_biased = x_orth.unsqueeze(-2) + d1 # [B, P, 1, Order]
        
        # --- 3. Truly Dynamic Latent Generation ---
        # [B, P, 1, Order] -> [B, Order, P, 1]
        x_t = x_orth_biased.squeeze(-2).transpose(1, 2).unsqueeze(-1)
        
        # Project each sensor at each time step into H-space
        # cond: [B, Order, P, H]
        cond = self.temporal_proj(x_t) * self.embeddings.transpose(1, 2) 
        # [B,T,P,H] -> [B,P,H,T]
        if self.use_temporal_mixer:
            cond_mix = cond.permute(0,2,3,1)
            H = cond_mix.shape[2]
            T = cond_mix.shape[3]
    #
            cond_mix = self.temporal_mixer(cond_mix.reshape(B*P, H, T))
            cond_mix = cond_mix.reshape(B, P, H, T).permute(0,3,1,2)

            cond = cond + cond_mix
        else:
            cond = cond
        # --- 4. Dynamic AERCA Coefficients ---
        # Creates a unique PxP matrix for every step in the window
        coeffs_time = torch.einsum('btph, btqh -> btpq', cond, cond)
        coeffs_time = torch.tanh(coeffs_time)

        # --- 5. Prediction (Forecasting) ---
        # Aggregate temporal info using max pooling (as per your best results)
        z_final, _ = torch.max(cond, dim=1) # [B, P, H]
        
        # Apply the second Latent Bias to the forecast
        # vf(z_final) -> [B, P, Order]
        # d2 -> [1, P, Order]
        d2 = self.bias_proj(self.delta_latent2)
        v_pred = self.vf(z_final) + d2 # [B, P, Order]

        if self.orth_transformer is None:
            preds_all_time = v_pred.transpose(1, 2) # [B, Order, P]
        else:
            preds_all_time = self.orth_transformer.inverse(v_pred)
        
        # Final forecast is the last step of the predicted window
        preds = preds_all_time[:, -1, :] 
        coeffs_freq = coeffs_time[:, 0, :, :] # First step coefficients

        return preds, coeffs_time, coeffs_freq
    


class MultiModalVLinear(nn.Module):
    def __init__(self, metric_dim, log_dim, trace_dim,
                 order, hidden_dim=128, device="cpu", options=None):

        super().__init__()
        self.md, self.ld, self.td = metric_dim, log_dim, trace_dim
        self.total = metric_dim + log_dim + trace_dim

        self.metric_orth = (options or {}).get("orth_transformer", None)

        # adapters (map -> metric space)
        self.ma = nn.Sequential(nn.Linear(metric_dim, metric_dim), nn.LayerNorm(metric_dim))
        self.la = nn.Sequential(nn.Linear(log_dim, metric_dim), nn.LayerNorm(metric_dim))
        self.ta = nn.Sequential(nn.Linear(trace_dim, metric_dim), nn.LayerNorm(metric_dim))

        # shared backbone
        opts = dict(options or {})
        opts["orth_transformer"] = None

        self.backbone = vlinear(
            num_vars=metric_dim,
            order=order,
            hidden_dim=hidden_dim,
            device=device,
            options=opts,
        )

        # fusion + output
        self.attn = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(metric_dim, self.total)

    # ---------------- utils ----------------

    def split(self, x):
        m, l = self.md, self.md + self.ld
        return x[..., :m], x[..., m:l], x[..., l:]

    def enc(self, x):
        return self.backbone(x)
          # [B,P,H]

    def fuse_linear_attn(self, zs):
        z = torch.stack(zs, dim=1)  # [B, M, P]

        q = torch.nn.functional.elu(z) + 1
        k = torch.nn.functional.elu(z) + 1

        scores = torch.einsum('bmp,bnp->bmpn', q, k)  # [B, M, P, M]
        attn = torch.softmax(scores, dim=-1)

        z_fused = torch.einsum('bmpn,bnp->bmp', attn, z)
        
        return z_fused.mean(dim=1), attn

    # ---------------- forward ----------------

    def forward(self, x):

        xm, xl, xt = self.split(x)

        if self.metric_orth is not None:
            xm = self.metric_orth(xm)

        xm, xl, xt = self.ma(xm), self.la(xl), self.ta(xt)

        pm, coeff_time_m, coeff_freq_m = self.enc(xm)
        pl, coeff_time_l, coeff_freq_l = self.enc(xl)
        pt, coeff_time_t, coeff_freq_t = self.enc(xt)

        zf, attn = self.fuse_linear_attn([pm, pl, pt])

        pred = self.out(zf)

        # --- aggregate coefficients ---
        coeff_time = (coeff_time_m + coeff_time_l + coeff_time_t) / 3
        coeff_freq = (coeff_freq_m + coeff_freq_l + coeff_freq_t) / 3

        return pred, coeff_time, coeff_freq