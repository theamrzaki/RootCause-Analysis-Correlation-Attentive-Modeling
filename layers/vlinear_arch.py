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

    def _compute_q_matrix_median(self, train_data, time_lag, save_path):
        """
        Computes a single Global Q matrix across all 51 variables using 
        Robust Median Aggregation and Temporal Decay.
        """
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        # train_data shape: [Samples, Window, Vars]
        S, W, V = train_data.shape
        
        sigma_list = []
        for feature_idx in range(V):
            # Extract feature across all samples: [Samples, Window]
            feat_windows = train_data[:, :, feature_idx]
            
            # 1. Zero-Mean the windows locally to focus on dynamics, not DC offset
            feat_windows = feat_windows - np.mean(feat_windows, axis=1, keepdims=True)
            
            # 2. Compute temporal covariance [W, W]
            cov = np.cov(feat_windows.T) 
            diag = np.diag(cov)
            
            # Skip features with zero variance to prevent NaNs
            if (diag < 1e-6).any(): 
                continue
                
            # 3. Standardize to Correlation Matrix (Pearson logic)
            corr = cov / (np.sqrt(np.outer(diag, diag)) + 1e-9) 
            sigma_list.append(corr)

        if not sigma_list:
            raise ValueError("No valid features found to compute OrthTransform. Check data variance.")

        # --- ENHANCEMENT 1: ROBUST MEDIAN AGGREGATION ---
        # Instead of np.mean, use median to prevent a single noisy sensor 
        # from 'smearing' the global temporal basis.
        sigma_stack = np.stack(sigma_list, axis=0)
        sigma_final = np.median(sigma_stack, axis=0)

        # --- ENHANCEMENT 2: TEMPORAL DECAY PRIOR (Optional but recommended) ---
        # Dampen correlations between time-steps that are very far apart
        # This sharpens the basis for local anomaly detection.
        mask = np.fromfunction(lambda i, j: np.exp(-np.abs(i - j) / (W / 2)), (W, W))
        sigma_final = sigma_final * mask

        # 4. Eigen-Decomposition
        eigenvalues, eigenvectors = eigh(sigma_final)
        
        # 5. Sort descending to get the Principal Temporal Components
        q_mat = np.flip(eigenvectors.T, axis=0)
        
        # 6. Save and Return
        np.save(self.matrix_path, q_mat)
        return q_mat

    def _compute_q_matrix_grouped_not_working(self, train_data, time_lag, save_path):
        """
        Computes a Nuanced Q matrix per sensor group (Stage-wise).
        """
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        S, W, V = train_data.shape
        
        # 1. Define SWaT Stages (Update these indices if your CSV columns differ)
        SWAT_STAGES = {
            "P1": list(range(0, 10)),   # Raw water
            "P2": list(range(10, 16)),  # Pre-treatment
            "P3": list(range(16, 26)),  # Ultrafiltration
            "P4": list(range(26, 35)),  # De-chlorination
            "P5": list(range(35, 46)),  # Reverse Osmosis
            "P6": list(range(46, 51))   # Effluent
        }

        # Final container: [Num_Sensors, Window, Window]
        full_q_tensor = np.zeros((V, W, W))

        for stage, indices in SWAT_STAGES.items():
            sigma_list = []
            
            # Calculate correlations only for this stage's sensors
            for idx in indices:
                feat_windows = train_data[:, :, idx]
                cov = np.cov(feat_windows.T)
                diag = np.diag(cov)
                
                if (diag < 1e-6).any(): continue
                
                corr = cov / (np.sqrt(np.outer(diag, diag)) + 1e-9)
                sigma_list.append(corr)

            if not sigma_list:
                print(f"Warning: No variance in Stage {stage}, using Identity.")
                q_stage = np.eye(W)
            else:
                # Average correlation for this specific stage
                sigma_mean = np.mean(sigma_list, axis=0)
                eigenvalues, eigenvectors = eigh(sigma_mean)
                q_stage = np.flip(eigenvectors.T, axis=0) # [W, W]

            # Assign this stage-specific Q to all sensors in this group
            for idx in indices:
                full_q_tensor[idx] = q_stage

        # Save the full [51, W, W] tensor
        np.save(self.matrix_path, full_q_tensor)
        return full_q_tensor

    def forward___(self, x):
        # x is [Batch, Window, Channels] -> (20, 36, 51)
        # self.Q is [1000, 1000]
        
        target_len = self.Q.shape[0] # 1000
        current_len = x.shape[1]    # 36
        
        if current_len < target_len:
            # Pad the temporal dimension (dim 1) with zeros at the beginning
            padding = (0, 0, target_len - current_len, 0) # (Left, Right, Top, Bottom) for the last two dims
            x = torch.nn.functional.pad(x, padding, "constant", 0)
        
        # Now x is [20, 1000, 51]
        # Apply transform: [B, W, C] * [W_new, W] -> [B, W_new, C]
        out = torch.einsum('bwc, vw -> bvc', x, self.Q)
        
        # If you only care about the original 36 steps, slice them back out
        return out[:, -current_len:, :].transpose(1, 2)

    #def inverse(self, x_orth):
    #    # [Batch, Channels, Window_Transformed] -> [Batch, Window, Channels]
    #    out = torch.einsum('bcw,wv->bcv', x_orth, self.Q)
    #    return out.transpose(1, 2)
    
    def inverse___(self, x_orth):
        # x_orth: [Batch, Channels, 10]
        # self.Q: [1000, 1000]
        
        current_w = x_orth.shape[2] # This is 10
        
        # We take the first 'current_w' basis vectors to project back
        # effectively projecting from the top-10 orthogonal components
        # back to a 10-step time series.
        Q_sliced = self.Q[:current_w, :current_w]
        
        # [B, C, 10] * [10, 10] -> [B, C, 10]
        out = torch.einsum('bcw, wv -> bcv', x_orth, Q_sliced)
        
        # Transpose to [Batch, Window, Channels] -> [131, 10, 51]
        return out.transpose(1, 2)
    
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# In __init__


# In forward


class RevIN(nn.Module):
    def __init__(self, num_vars, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_vars))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_vars))

    def forward(self, x, mode):
        # x: [B, O, P]
        if mode == 'norm':
            self.mu = x.mean(dim=1, keepdim=True)        # [B,1,P]
            self.sigma = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mu) / self.sigma
            return x * self.gamma + self.beta

        elif mode == 'denorm':
            x = (x - self.beta) / (self.gamma + self.eps)
            return x * self.sigma + self.mu
  

class vlinear_old(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=256, device="cpu", options=None):
        super().__init__()
        self.num_vars = num_vars  
        self.order = order*1  -1  
        self.device = device
        self.orth_transformer = options.get('orth_transformer') 
        
        # 1. Delta Biases
        self.delta_latent1 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.delta_latent2 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.bias_proj = nn.Linear(hidden_dim, self.order)
        
        # 2. Embeddings & Projections
        self.embeddings = nn.Parameter(torch.randn(1, num_vars, 1, hidden_dim))
        self.temporal_proj = nn.Linear(1, hidden_dim)
        
        # 3. Contrastive weighting layer 
        # This helps the model learn WHICH parts of the window indicate an anomaly
        self.temporal_weight = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.order) 
        )

    def forward(self, inputs: torch.Tensor):
        B, T, P = inputs.shape # [Batch, Window, Sensors]
        
        # --- 1. Step into Orthogonal Domain ---
        if self.orth_transformer is None:
            x_orth = inputs.transpose(1, 2) # [B, P, T]
        else:
            x_orth = self.orth_transformer(inputs)
        
        # --- 2. Apply Delta1 Latent Bias ---
        d1 = self.bias_proj(self.delta_latent1).unsqueeze(-2) 
        x_orth_biased = x_orth.unsqueeze(-2) + d1 
        
        # --- 3. Truly Dynamic Latent Generation ---
        x_t = x_orth_biased.squeeze(-2).transpose(1, 2).unsqueeze(-1) # [B, T, P, 1]
        cond = self.temporal_proj(x_t) * self.embeddings.transpose(1, 2) # [B, T, P, H]
        
        # --- 4. Contrastive Aggregation (The "BARO Killer" Logic) ---
        # Split window into History (Normal) and Current (Anomalous) like BARO
        split_idx = int(0.7 * T)
        cond_history = cond[:, :split_idx, :, :]
        cond_current = cond[:, split_idx:, :, :]
        
        # BARO logic: Mean of history vs Max of current
        # This highlights the DEVIATION rather than the raw value
        z_hist_mean = torch.mean(cond_history, dim=1) # [B, P, H]
        z_curr_max, _ = torch.max(cond_current, dim=1) # [B, P, H]
        
        # Feature Delta: This represents how much each sensor "jumped"
        z_final = z_curr_max - z_hist_mean 
        
        # --- 5. Dynamic AERCA Coefficients ---
        # We use the 'z_final' to build the correlation matrix
        coeffs_time = torch.einsum('bph, bqh -> bpq', z_final, z_final)
        coeffs_time = torch.tanh(coeffs_time).unsqueeze(1) # [B, 1, P, P]

        # --- 6. Prediction (Forecasting) ---
        d2 = self.bias_proj(self.delta_latent2)
        v_pred = self.vf(z_final) + d2 # [B, P, Order]

        if self.orth_transformer is None:
            preds_all_time = v_pred.transpose(1, 2)
        else:
            preds_all_time = self.orth_transformer.inverse(v_pred)
        
        preds = preds_all_time[:, -1, :] 
        coeffs_freq = coeffs_time[:, 0, :, :] 

        return preds, coeffs_time, coeffs_freq
    

import torch
import torch.nn as nn
class vlinear(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=256, device="cpu", options=None):
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
        self.a = nn.Parameter(torch.randn(num_vars))
        self.ln = nn.LayerNorm(hidden_dim)
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, self.order) 
        )
        #self.revin = RevIN(num_vars)
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
        #cond_mix = cond.permute(0,2,3,1)
        #H = cond_mix.shape[2]
        #T = cond_mix.shape[3]
#
        #cond_mix = self.temporal_mixer(cond_mix.reshape(B*P, H, T))
        #cond_mix = cond_mix.reshape(B, P, H, T).permute(0,3,1,2)

        cond = cond #+ cond_mix
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
    