import torch
import torch.nn as nn
import numpy as np
import os
from numpy.linalg import eigh
import pandas as pd
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import os
from numpy.linalg import eigh

def build_orthogonal_basis(num_vars, basis_type):
    t = np.linspace(-1, 1, num_vars)

    if basis_type == "legendre":
        from scipy.special import legendre
        basis = np.array([
            legendre(i)(t)
            for i in range(num_vars)
        ])

    elif basis_type == "chebyshev":
        from numpy.polynomial.chebyshev import chebvander
        basis = chebvander(t, num_vars - 1).T

    elif basis_type == "hermite":
        from numpy.polynomial.hermite import hermvander
        basis = hermvander(t, num_vars - 1).T

    elif basis_type == "laguerre":
        from numpy.polynomial.laguerre import lagvander
        basis = lagvander(t, num_vars - 1).T

    else:
        raise ValueError(f"Unknown basis: {basis_type}")

    basis = torch.tensor(basis, dtype=torch.float32)

    # Convert sampled basis to an orthonormal basis
    Q, R = torch.linalg.qr(basis.T)

    return Q

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
    def __init__(self, num_vars, order, hidden_dim=128, device="cpu", options=None):
        super().__init__()

        self.num_vars = num_vars
        self.order = order - 1
        self.options = options or {}

        # options
        self.latent_mode = self.options.get("latent_mode", "mul")
        self.temporal_mixer = self.options.get("temporal_mixer", False)
        self.coeff_mode = self.options.get("coeff_mode", "symmetric")
        self.pool = self.options.get("pool", "split_diff")
        self.context = self.options.get("context", "gate")
        self.predictor = self.options.get("predictor", "mlp")

        # orthogonal transform
        self.transformation = self.options.get("transformation", "orthogonal")
        self.orth_transformer = self.options.get("orth_transformer", None)
        if self.transformation == "learned":
            self.init_transform = nn.Parameter(
                torch.randn(num_vars, num_vars)
            )
        elif self.transformation in ["legendre", "laguerre", "chebyshev", "hermite"]:
            self.register_buffer("basis", build_orthogonal_basis(num_vars, self.transformation))


        # Latent construction
        self.embedding = nn.Parameter(
            torch.randn(1, num_vars, 1, hidden_dim)
        )
        self.proj = nn.Linear(1, hidden_dim)
        if self.latent_mode == "gate":
            self.value = nn.Linear(1, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Sigmoid()
            )

        # Coefficient model
        if self.coeff_mode == "bipartite":
            self.src = nn.Linear(hidden_dim, hidden_dim)
            self.tgt = nn.Linear(hidden_dim, hidden_dim)

        # Temporal mixer
        if self.temporal_mixer:
            self.mixer = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim
            )

        # Context sharpening
        if self.context == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)

        elif self.context == "residual":
            self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        elif self.context == "gate":
            self.context_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        elif self.context == "linear_attn":
            self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attn_scale = nn.Parameter(torch.tensor(-2.2))

        # Prediction head
        if self.predictor == "linear":
            self.head = nn.Linear(hidden_dim, self.order)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, self.order)
            )


        self.bias = nn.Parameter(
            torch.zeros(1, num_vars, self.order)
        )
        
        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



    def forward(self, x):
        B,T,P = x.shape

        # Normalization
        x = (x - x.mean(1, keepdim=True))
        x = x / (x.var(1, keepdim=True)+1e-5).sqrt()

        # Orthogonal space
        # Transformation
        if self.transformation == "orthogonal" and self.orth_transformer is not None:
            x = self.orth_transformer(x)
        elif self.transformation == "learned":
            x = torch.einsum("btp,pq->btq",x,self.init_transform)
        elif self.transformation in ["legendre", "laguerre", "chebyshev", "hermite"]:
            basis = getattr(self, f"basis")
            x = torch.einsum("btp,pq->btq",x,basis)

        # B,P,T -> B,T,P,1
        x = x.transpose(1,2).unsqueeze(-1)

        # Latent generation
        if self.latent_mode == "mul":
            cond = self.proj(x) * self.embedding.transpose(1, 2) 
        elif self.latent_mode == "add":
            cond = self.proj(x) + self.embedding.transpose(1, 2) 
        else:   # gate
            cond = (
                self.value(x)
                *
                self.gate(x)
                *
                self.embedding.transpose(1, 2) 
            )

        # Temporal mixer
        if self.temporal_mixer:
            z = cond.permute(0,2,3,1)
            z = self.mixer(
                z.reshape(B*P, z.size(2), T)
            )
            cond = cond + z.reshape(
                B,P,-1,T
            ).permute(0,3,1,2)

        # Context sharpening
        if self.context == "layernorm":
            cond = self.norm(cond)
        elif self.context == "residual":
            cond = cond + self.context_proj(cond)
        elif self.context == "gate":
            cond = cond * self.context_gate(cond)
        elif self.context == "linear_attn":
            q = self.q_proj(cond)
            k = self.k_proj(cond)
            v = self.v_proj(cond)

            # positive feature map
            q = F.elu(q) + 1
            k = F.elu(k) + 1

            kv = torch.einsum(
                "btph,btpd->bthd",
                k,
                v
            )

            z = torch.einsum(
                "btph,bthd->btpd",
                q,
                kv
            )

            norm = torch.einsum(
                "btph,bth->btp",
                q,
                k.sum(dim=2)
            ).unsqueeze(-1)

            
            attn = self.out_proj(
                z / (norm + 1e-6)
            )

            # with fixed residual connection
            #cond = cond + attn

            # with learnable scaling factor
            alpha = torch.sigmoid(self.attn_scale)
            cond = cond + alpha * attn

        # Coefficients
        if self.coeff_mode == "bipartite":
            coeff = torch.einsum(
                "btph,btqh->btpq",
                self.src(cond),
                self.tgt(cond)
            )
        else:
            c = (
                F.normalize(cond, dim=-1)
                if self.coeff_mode == "cosine"
                else cond
            )
            coeff = torch.einsum(
                "btph,btqh->btpq",
                c,c
            )
        coeff = torch.tanh(coeff)

        # Temporal aggregation
        if self.pool == "mean":
            z = cond.mean(1)
        elif self.pool == "max":
            z = cond.max(1).values
        else:
            h,r = torch.chunk(cond,2,dim=1)
            if self.pool == "split_mean":
                z = r.mean(1)-h.mean(1)
            elif self.pool == "split_max":
                z = r.max(1).values-h.max(1).values
            else: # split_diff
                z = r.mean(1)-h.max(1).values

        # Prediction
        pred = self.head(z) + self.bias
        if self.transformation == "orthogonal" and self.orth_transformer is not None:
            pred = self.orth_transformer.inverse(pred)
        elif self.transformation == "learned":
            pred = torch.einsum("btp,pq->btq",pred,self.init_transform.T)
        elif self.transformation in ["legendre", "laguerre", "chebyshev", "hermite"]:
            basis = getattr(self, f"basis")
            pred = torch.einsum("btp,pq->btq",pred,basis.T)
        else:
            pred = pred.transpose(1, 2)

        return pred[:,-1,:], coeff, coeff[:,0]
