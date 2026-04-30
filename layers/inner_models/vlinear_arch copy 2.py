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
        """             
        # 1. Sort descending (top eigenvectors explain most variance)
        q_mat = np.flip(eigenvectors.T, axis=0)
        
        # 2. THE PCA STEP: Keep only the Top-K components
        # If W=10000, we take the top 200 to represent the window
        top_k = 1000 
        q_mat = q_mat[:top_k, :] # Shape becomes [200, 10000]
        
        # 3. Compression and Save
        q_mat = q_mat.astype(np.float32) # float16 might be too low-precision for eigh
        np.save(self.matrix_path, q_mat)
        
        return q_mat
        """
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
        # x at runtime might be [9999, 1, 10] 
        B, current_W, C = x.shape
        basis_W = self.Q.shape[1]  # 10,000
        K = self.Q.shape[0]        # 200

        if disable_orth:
            return x.transpose(1, 2)

        # RIGOROUS PADDING: If runtime window is 1, pad it to 10,000
        # This makes the "Pink" model look at the current point 
        # through the lens of the 10k-step basis.
        if current_W < basis_W:
            padding = (0, 0, basis_W - current_W, 0) 
            x = torch.nn.functional.pad(x, padding, "constant", 0)
        
        # Now x is [B, 10000, C], calculation is safe
        # But we MUST batch this if B is 9,999!
        
        # Batching logic to prevent the 3.73 GiB crash:
        if B > 512:
            outs = []
            for i in range(0, B, 512):
                chunk = x[i:i+512]
                outs.append(torch.einsum('bwc, kw -> bkc', chunk, self.Q))
            out = torch.cat(outs, dim=0)
        else:
            out = torch.einsum('bwc, kw -> bkc', x, self.Q)

        return out.transpose(1, 2) # [B, C, 200]

    def inverse(self, x_orth, disable_orth=False):
        # x_orth: [Batch, Channels, 200]
        B, C, K = x_orth.shape
        basis_W = self.Q.shape[1] # 10,000

        if disable_orth:
            return x_orth.transpose(1, 2)

        # Memory Guard: If the batch is too large, process in chunks
        if B > 512:
            reconstructed_list = []
            for i in range(0, B, 512):
                batch_orth = x_orth[i : i + 512] # [512, 10, 200]
                # [512, 10, 200] @ [200, 10000] -> [512, 10, 10000]
                recon_chunk = torch.einsum('bck, kw -> bcw', batch_orth, self.Q)
                reconstructed_list.append(recon_chunk)
            out = torch.cat(reconstructed_list, dim=0)
        else:
            # Standard fast path for small batches
            out = torch.einsum('bck, kw -> bcw', x_orth, self.Q)
        
        # Final Shape: [Batch, Window, Channels] -> [9999, 10000, 10]
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
        #self.revin = RevIN(num_vars)

    def forward(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape
        # z: [B, P, H]

        
        # --- 1. Step into Orthogonal Domain ---
        #inputs = self.revin(inputs, mode='norm')
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
        #w = torch.sigmoid(self.a)
        #τ < 1 → sharper distribution
        #try τ = 0.3 or 0.1
        ###tau = 0.1
        ###w = torch.softmax(self.a / tau, dim=0)
        ###w = w / (w.sum() + 1e-8)   # L1 normalize
####
        ##### Step 2: aggregation
        #####s = torch.einsum('p,bph->bh', w, z)   # [B, H]
        ###vec = w[None, :, None] * z
        ###vec = vec.sum(dim=1, keepdim=True).expand_as(z)
        ##### Step 3: broadcast
        #####vec = s.unsqueeze(1).repeat(1, self.num_vars, 1)  # [B, P, H]
####
        ##### Residual-style combination (IMPORTANT)
        ###cond = self.ln(z + vec)
        cond = z * self.embeddings # [B, P, H]
        # --- 4. Prediction with Delta2 ---
        v_pred = self.vf(cond).unsqueeze(-2) + self.delta2
        v_pred = v_pred.squeeze(-2) # [B, P, 1000]
        
        # --- 5. Return to Time Domain ---
        preds_all_time = self.orth_transformer.inverse(v_pred)
        preds = preds_all_time[:, -1, :] 
        #preds_all_time = self.revin(preds_all_time, mode='denorm')
        #preds = preds_all_time[:, -1, :]

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
    
class vlinear_new(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=256, device="cpu", options=None):
        super().__init__()

        self.num_vars = num_vars
        self.order = order
        self.device = device

        self.orth_transformer = options.get('orth_transformer')

        # ----------------------------
        # SAFE DELTAS (no 4D tensors)
        # ----------------------------
        self.delta1 = nn.Parameter(torch.zeros(1, num_vars, order))
        self.delta2 = nn.Parameter(torch.zeros(1, num_vars, order))

        # ----------------------------
        # embeddings (kept)
        # ----------------------------
        self.embeddings = nn.Parameter(torch.randn(1, hidden_dim))

        # ----------------------------
        # TSFlow-style projection block
        # replaces: temporal_proj + flatten logic
        # ----------------------------
        self.in_proj = nn.Linear(order, hidden_dim)
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out_proj = nn.Linear(hidden_dim, order)

        # ----------------------------
        # lightweight coefficient head (replaces P×P einsum)
        # ----------------------------
        self.q_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # output model
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, order)
        )

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------
    def forward(self, inputs: torch.Tensor):
        """
        inputs: [B, O_curr, P]
        """

        B, O_curr, P = inputs.shape

        # --------------------------------------------------------
        # 1. Orthogonal transform (UNCHANGED but treated safely)
        # --------------------------------------------------------
        x_orth = self.orth_transformer(inputs)  # [B, P, order]

        # --------------------------------------------------------
        # 2. Add delta WITHOUT 4D expansion
        # --------------------------------------------------------
        x_orth = x_orth + self.delta1  # broadcast-safe

        # --------------------------------------------------------
        # 3. TSFlow-style compression (NO flatten(P * order))
        # --------------------------------------------------------
        h = self.in_proj(x_orth)       # [B, P, H]
        h = self.mix(h)                # stabilizer

        # --------------------------------------------------------
        # 4. latent reconstruction (replaces temporal_proj path)
        # --------------------------------------------------------
        z = self.out_proj(h)           # [B, P, order]
        z = z + self.delta2            # safe bias

        # --------------------------------------------------------
        # 5. embeddings (kept but stable)
        # --------------------------------------------------------
        cond = z * self.embeddings     # [B, P, H]

        # --------------------------------------------------------
        # 6. velocity field prediction (unchanged idea)
        # --------------------------------------------------------
        v_pred = self.vf(cond)         # [B, P, order]

        # --------------------------------------------------------
        # 7. inverse transform (UNCHANGED dependency)
        # --------------------------------------------------------
        preds_all_time = self.orth_transformer.inverse(v_pred)
        preds = preds_all_time[:, -1, :]   # [B, P]

        # ========================================================
        # 8. REPLACED: NO MORE P×P EINSUM
        # ========================================================
        # OLD:
        # coeffs_time = torch.einsum('bph,bqh->bpq', cond, cond)

        # NEW: low-rank factorization (O(P²) memory removed)
        q = self.q_proj(cond)   # [B, P, d]
        k = self.k_proj(cond)   # [B, P, d]

        coeffs_time = torch.matmul(q, k.transpose(-1, -2))  # [B, P, P]

        # stabilize instead of tanh explosion
        coeffs_time = torch.tanh(coeffs_time)

        # --------------------------------------------------------
        # 9. temporal expansion (safe repeat)
        # --------------------------------------------------------
        coeffs_time = coeffs_time.unsqueeze(1).repeat(1, O_curr, 1, 1)

        # --------------------------------------------------------
        # 10. frequency view (cheap slice)
        # --------------------------------------------------------
        coeffs_freq = coeffs_time[:, 0, :, :]

        return preds, coeffs_time, coeffs_freq