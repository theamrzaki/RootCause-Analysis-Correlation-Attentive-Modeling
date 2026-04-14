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
            # IDENTITY MODE: Just return to time-major shape
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



class vlinear(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=256, device="cpu", options=None):
        super().__init__()
        self.num_vars = num_vars  
        self.order = order        
        self.device = device
        
        self.orth_transformer = options.get('orth_transformer') 
        
        # 1. Delta Biases (Faithful to Model logic)
        # These act as "Learned Context" for the orthogonal domain
        # delta1: [1, Channels, 1, Lag]
        self.delta1 = nn.Parameter(torch.zeros(1, num_vars, 1, self.order))
        self.delta2 = nn.Parameter(torch.zeros(1, num_vars, 1, self.order))

        # 2. Updated Embeddings 
        # In the Model code, embeddings are often 1D and expanded
        self.embeddings = nn.Parameter(torch.randn(1, hidden_dim))
        
        # 3. Projection matching the Model's logic
        self.temporal_proj = nn.Linear(self.order, hidden_dim)
        self.a = nn.Parameter(torch.randn(num_vars))
        self.ln = nn.LayerNorm(hidden_dim)
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.order) 
        )
        #self.spatial_weight = nn.Parameter(torch.ones(1, num_vars, 1))
        
        # Identity-preserving embedding
        #self.sensor_embeddings = nn.Parameter(torch.randn(num_vars, hidden_dim))
        # Instead of one hidden_dim, we use 3 heads 
        # (e.g., Immediate Lag, Medium Lag, Long Lag)
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        self.query_proj = nn.Linear(self.head_dim, self.head_dim)
        self.key_proj = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape
        # z: [B, P, H]

        
        # --- 1. Step into Orthogonal Domain ---
        x_orth = self.orth_transformer(inputs) # [B, P, 1000]
        
        # --- 2. Apply Delta1 Bias ---
        # We unsqueeze to [B, P, 1, 1000] to match Model's 4D logic
        # This bias helps overcome the "zero-padding" dilution
        x_orth_biased = x_orth.unsqueeze(-2) + self.delta1
        
        # --- 3. vecTrans Latent Generation ---
        # Flatten back to 3D for the Linear layer

        #z = self.temporal_proj(x_orth_biased.squeeze(-2)) # [B, P, H]


       
        z = self.temporal_proj(x_orth_biased.squeeze(-2))
        
        # Step 1: weights
        w = torch.sigmoid(self.a)
        w = w / (w.sum() + 1e-8)   # L1 normalize

        # Step 2: aggregation
        s = torch.einsum('p,bph->bh', w, z)   # [B, H]

        # Step 3: broadcast
        vec = s.unsqueeze(1).repeat(1, self.num_vars, 1)  # [B, P, H]

        # Residual-style combination (IMPORTANT)
        cond = self.ln(z + vec)
        
        # --- 4. Prediction with Delta2 ---
        v_pred = self.vf(cond).unsqueeze(-2) + self.delta2
        v_pred = v_pred.squeeze(-2) # [B, P, 1000]
        
        # --- 5. Return to Time Domain ---
        preds_all_time = self.orth_transformer.inverse(v_pred)
        preds = preds_all_time[:, -1, :] 

        # AERCA Coefficients
        coeffs_time = torch.einsum('bph,bqh->bpq', cond, cond)
        coeffs_time = coeffs_time.unsqueeze(1).repeat(1, O_curr, 1, 1)

        # Split into heads to capture different temporal "frequencies"
        #z_heads = z.view(B, P, self.num_heads, self.head_dim)

        ## Calculate causal coefficients per head
        ## This mimics the "ModuleList" of GVAR but stays in the Orthogonal domain
        #z_q = self.query_proj(z_heads)
        #z_k = self.key_proj(z_heads)
#
        ## Generate the A matrix using the heads
        ## 3. Generate the A matrix using the heads [B, P, P]
        #coeffs_spatial = torch.einsum('bphd, bqhd -> bpq', z_q, z_k)
        #coeffs_spatial = torch.tanh(coeffs_spatial) 
#
        ## 4. RESTORE THE SHAPE: [B, P, P] -> [B, O_curr, P, P]
        ## This makes it compatible with the AERCA/GVAR evaluation logic
        #coeffs_time = coeffs_spatial.unsqueeze(1).repeat(1, O_curr, 1, 1)

        # 5. Frequency representation (usually just a slice or the same matrix)
        coeffs_freq = coeffs_time[:, 0, :, :]

        return preds, coeffs_time, coeffs_freq
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class vlinear___(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=256, device="cpu", options=None):
        super().__init__()
        self.num_vars = num_vars
        self.order = order # window size for orth domain
        self.device = device
        self.orth_transformer = options.get('orth_transformer')
        
        # --- PATH 1: Orthogonal (Spectral) Domain ---
        self.spectral_proj = nn.Linear(self.order, hidden_dim)
        self.delta1 = nn.Parameter(torch.zeros(1, num_vars, 1, self.order))
        self.delta2 = nn.Parameter(torch.zeros(1, num_vars, 1, self.order))

        # --- PATH 2: Temporal (Point-wise) Domain ---
        # Captures the local "shocks" in the window
        self.temporal_path = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # --- BRIDGE: Linear Attention ---
        # Q: Spectral Context | K, V: Temporal Local Dynamics
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # --- Output / Forecasting ---
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.order)
        )
        
        self.sensor_embeddings = nn.Parameter(torch.randn(num_vars, hidden_dim))

    def forward(self, inputs: torch.Tensor):
        # inputs shape: [B, W, P]
        B, W, P = inputs.shape
        
        # --- 1. Path A: Spectral Extraction ---
        x_orth = self.orth_transformer(inputs) # [B, P, order]
        x_orth_biased = x_orth.unsqueeze(-2) + self.delta1
        z_spec = self.spectral_proj(x_orth_biased.squeeze(-2)) # [B, P, H]
        
        # --- 2. Path B: Temporal Extraction ---
        # Map each time-step per sensor to a latent feature
        # [B, W, P] -> [B, P, W, 1]
        x_temp = inputs.permute(0, 2, 1).unsqueeze(-1)
        z_temp_seq = self.temporal_path(x_temp) # [B, P, W, H]
        # Pool to match spectral dimensions [B, P, H]
        z_temp = z_temp_seq.mean(dim=2) 
        
        # --- 3. Linear Attention Interaction ---
        # Q: Spectral (Global Physics), K/V: Temporal (Local Dynamics)
        q = self.q_proj(z_spec).view(B, P, self.num_heads, self.head_dim)
        k = self.k_proj(z_temp).view(B, P, self.num_heads, self.head_dim)
        v = self.v_proj(z_temp).view(B, P, self.num_heads, self.head_dim)

        # Apply Feature Mapping for Linear Attention (Kernel Trick)
        # Using ELU+1 to ensure positivity
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Context computation (Linearized: O(P) instead of O(P^2))
        # This aggregates how the temporal shocks relate to the global physics
        context = torch.einsum('bphd, bphm -> bhdm', k, v) # [B, H, D, D]
        z_bridge = torch.einsum('bphd, bhdm -> bphm', q, context) # [B, P, H, D]
        z_bridge = z_bridge.reshape(B, P, -1)
        
        # --- 4. Causal & Prediction Heads ---
        # Combine bridge features with sensor identity
        cond = z_bridge * self.sensor_embeddings.unsqueeze(0)
        
        # Causal Matrix
        coeffs_spatial = torch.einsum('bph, bqh -> bpq', cond, cond)
        
        # Prediction (Residual spectral prediction)
        v_pred = self.vf(cond) + x_orth_biased.squeeze(-2)
        v_pred = v_pred.unsqueeze(-2) + self.delta2
        v_pred = v_pred.squeeze(-2)
        
        # Return to Time Domain
        preds_all_time = self.orth_transformer.inverse(v_pred)
        preds = preds_all_time[:, -1, :] 

        # AERCA/GVAR compatibility
        coeffs_time = coeffs_spatial.unsqueeze(1).repeat(1, W, 1, 1)
        coeffs_freq = coeffs_time[:, 0, :, :]

        return preds, coeffs_time, coeffs_freq