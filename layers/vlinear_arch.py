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
  
class OrthTransform_multi_modal(nn.Module):
    def __init__(self,dataset_obj,save_path,time_lag,device):
        super().__init__()
        self.device=device
        self.time_lag=time_lag

        self.paths={"m":os.path.join(save_path,f"Q_m_lag{time_lag}.npy"),
                    "l":os.path.join(save_path,f"Q_l_lag{time_lag}.npy"),
                    "t":os.path.join(save_path,f"Q_t_lag{time_lag}.npy")}
        
        self.Qm=self._load_or_compute(dataset_obj.data_dict["x_n_list_m"],self.paths["m"],save_path)
        self.Ql=self._load_or_compute(dataset_obj.data_dict["x_n_list_l"],self.paths["l"],save_path)
        self.Qt=self._load_or_compute(dataset_obj.data_dict["x_n_list_t"],self.paths["t"],save_path)

        self.register_buffer("Qm_torch",torch.from_numpy(self.Qm.astype(np.float32)))
        self.register_buffer("Ql_torch",torch.from_numpy(self.Ql.astype(np.float32)))
        self.register_buffer("Qt_torch",torch.from_numpy(self.Qt.astype(np.float32)))

    def _load_or_compute(self,data,path,save_path):
        if os.path.isfile(path): return np.load(path)
        os.makedirs(save_path,exist_ok=True)
        S,W,V=data.shape
        sigma_list=[]
        for i in range(V):
            x=data[:,:,i]
            cov=np.cov(x.T)
            d=np.diag(cov)
            if np.any(d<1e-6): continue
            cov=cov/(np.sqrt(np.outer(d,d))+1e-9)
            sigma_list.append(cov)
        if len(sigma_list)==0: raise ValueError("OrthTransform failed")
        sigma=np.mean(sigma_list,axis=0)
        eigvals,eigvecs=np.linalg.eigh(sigma)
        Q=np.flip(eigvecs.T,axis=0)
        np.save(path,Q)
        return Q
    
    def forward(self,x,mode):
        Q={"m":self.Qm_torch,"l":self.Ql_torch,"t":self.Qt_torch}[mode]
        B,T,C=x.shape
        if T<Q.shape[0]:
            x=F.pad(x,(0,0,Q.shape[0]-T,0))
        out=torch.einsum("btc,vt->bvc",x,Q)
        return out[:,-T:,:].transpose(1,2)
    
    def inverse(self, x, mode):
        Q = {
            "m": self.Qm_torch,
            "l": self.Ql_torch,
            "t": self.Qt_torch
        }[mode]

        B, C, T = x.shape

        # DO NOT truncate Q blindly
        Qs = Q[:C, :C]

        out = torch.einsum("bct,tc->btc", x, Qs)

        return out.transpose(1, 2)


class TinyMoM(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_memories=4,
        top_k=2
    ):
        super().__init__()

        assert hidden_dim % num_memories == 0

        self.num_memories = num_memories
        self.sub_dim = hidden_dim // num_memories
        self.top_k = top_k

        self.router = nn.Linear(
            hidden_dim,
            num_memories,
            bias=False
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )


    def forward(self, x):

        # x:
        # [B,T,P,H]

        B,T,P,H = x.shape


        # --------------------------
        # Router
        # --------------------------

        state = x.mean(dim=1)
        # [B,P,H]


        scores = torch.softmax(
            self.router(state),
            dim=-1
        )
        # [B,P,N]


        if self.top_k < self.num_memories:

            vals, idx = torch.topk(
                scores,
                self.top_k,
                dim=-1
            )

            mask = torch.zeros_like(scores)

            mask.scatter_(
                -1,
                idx,
                vals
            )

            scores = mask / (
                mask.sum(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )


        # --------------------------
        # Memory split
        # --------------------------

        memories = x.reshape(
            B,
            T,
            P,
            self.num_memories,
            self.sub_dim
        )


        # temporal aggregation
        memories = memories.mean(dim=1)
        # [B,P,N,sub_dim]


        # --------------------------
        # Routing
        # --------------------------

        scores = scores.unsqueeze(-1)
        # [B,P,N,1]


        memories = memories * scores
        # [B,P,N,sub_dim]


        # merge memories back
        output = memories.reshape(
            B,
            P,
            H
        )


        return self.norm(output)

class vlinear_old(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=128, device="cpu", options=None):#128 for SMD
        super().__init__()
        self.num_vars = num_vars  
        self.order = order*1  -1      
        self.device = device
        self.options = options or {}
        if "orth_transformer_multi_modality" in options and options["orth_transformer_multi_modality"]:
            self.multi_modal = True
            self.orth_transformer = OrthTransform_multi_modal(
                dataset_obj=options['dataset_obj'],
                save_path=options['save_path'],
                time_lag=options['time_lag'],
                device=device
            )
        else:
            self.multi_modal = False
            self.orth_transformer = options.get('orth_transformer') 
        
        # 1. Delta Biases (Faithful to Model logic)
        # These act as "Learned Context" for the orthogonal domain
        # delta1: [1, Channels, 1, Lag]
        self.delta_latent1 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.delta_latent2 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))

        self.source_proj = nn.Linear(hidden_dim, hidden_dim)
        self.target_proj = nn.Linear(hidden_dim, hidden_dim)
        # Projection to match the output 'order'
        self.bias_proj = nn.Linear(hidden_dim, self.order)
        
        # 2. Updated Embeddings 
        # In the Model code, embeddings are often 1D and expanded
        #self.embeddings = nn.Parameter(torch.randn(1, hidden_dim))
        self.embeddings = nn.Parameter(torch.randn(1, num_vars, 1, hidden_dim))
        # 3. Projection matching the Model's logic
        #self.temporal_proj = nn.Linear(self.order, hidden_dim)
        self.temporal_proj = nn.Linear(1, hidden_dim)
        #self.mom = TinyMoM(
        #    hidden_dim,
        #    num_memories=4,
        #    top_k=2
        #)
        #self.use_MoM = options.get(
        #    "use_MoM",
        #    True
        #)
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
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all learnable layers."""

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():

                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)

                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)

                    elif "bias" in name:
                        nn.init.zeros_(param)


    def forward(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape # [B, Window, Sensors] e.g., [B, 2, 51]
        
        # RIN
        x_mean = torch.mean(inputs, dim=1, keepdim=True)
        inputs = inputs - x_mean
        x_var=torch.var(inputs, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        inputs = inputs / torch.sqrt(x_var)
        
        # --- 1. Step into Orthogonal Domain ---
        if self.orth_transformer is None:
            x_orth = inputs.transpose(1, 2) # [B, 51, 2]
        else:
            if self.multi_modal:
                md, ld, td = self.md, self.ld, self.td

                xm = inputs[:, :, :md]
                xl = inputs[:, :, md:md+ld]
                xt = inputs[:, :, md+ld:md+ld+td]

                xm, xl, xt = self.orth_transformer(xm, xl, xt)

                assert xm.shape[1] == xl.shape[1] == xt.shape[1], "Orth mismatch in time axis"

                x_orth = torch.cat([xm, xl, xt], dim=1)
            else:
                x_orth = self.orth_transformer(inputs)
        
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
        #coeffs_time = torch.einsum('btph, btqh -> btpq', cond, cond)
        #coeffs_time = torch.tanh(coeffs_time)
        src = self.source_proj(cond)
        tgt = self.target_proj(cond)

        coeffs_time = torch.einsum(
            'btph, btqh -> btpq',
            src,
            tgt
        )
        coeffs_time = torch.tanh(coeffs_time)
        # --- 5. Prediction (Forecasting) ---
        # Aggregate temporal info using max pooling (as per your best results)
        
        
        
        split = (cond.size(1) // 2) #+ 1
        history = cond[:, :split]      # first half
        recent  = cond[:, split:]      # second half
        #split_method = self.options.get("split_method")
        #if split_method == "mean":
        history = history.mean(dim=1)
        recent  = recent.mean(dim=1)
        #split_method == "max"
        #history = history.max(dim=1).values
        #recent  = recent.max(dim=1).values
        z_final = recent - history

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


class vlinear_MoM(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=128, device="cpu", options=None):
        super().__init__()
        self.num_vars = num_vars  
        self.order = order*1  -1        
        self.device = device
        self.options = options or {}
        self.gru = nn.GRU(
            input_size=num_vars,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.linear_time = nn.Linear(num_vars, hidden_dim)
        self.linear_freq = nn.Linear(2 * num_vars, hidden_dim)
        self.vf_gru = nn.Linear(hidden_dim, num_vars)
        self.vf_freq = nn.Linear(hidden_dim, num_vars)
        self.router = nn.Linear(num_vars, 2)   # GRU and Frequency experts


        
        self.num_vars = num_vars  
        self.order = order*1  -1      
        self.device = device
        self.options = options or {}
        if "orth_transformer_multi_modality" in options and options["orth_transformer_multi_modality"]:
            self.multi_modal = True
            self.orth_transformer = OrthTransform_multi_modal(
                dataset_obj=options['dataset_obj'],
                save_path=options['save_path'],
                time_lag=options['time_lag'],
                device=device
            )
        else:
            self.multi_modal = False
            self.orth_transformer = options.get('orth_transformer') 
        
        # 1. Delta Biases (Faithful to Model logic)
        # These act as "Learned Context" for the orthogonal domain
        # Projection to match the output 'order'
        self.bias_proj = nn.Linear(hidden_dim, self.order)
        self.bias_proj2 = nn.Linear(hidden_dim, self.order)
        # 2. Updated Embeddings 
        # In the Model code, embeddings are often 1D and expanded
        #self.embeddings = nn.Parameter(torch.randn(1, hidden_dim))


        # Learnable orthogonal-domain residual biases
        self.delta_latent1 = nn.Parameter(
            torch.empty(1, num_vars, hidden_dim)
        )

        self.delta_latent2 = nn.Parameter(
            torch.empty(1, num_vars, hidden_dim)
        )

        # Learnable feature scaling
        self.embeddings = nn.Parameter(
            torch.empty(1, num_vars, 1, hidden_dim)
        )

        
        self.hidden_dim = hidden_dim
        self.temporal_proj = nn.Linear(1, hidden_dim)
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, self.order) 
        )
        self.temporal_attention = nn.Linear(hidden_dim,1)
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
        self.recency_bias = nn.Parameter(torch.tensor(0.5))
        #self._init_weights()

    def _init_latent_parameters(self):
        # Residual biases: start almost inactive
        nn.init.normal_(
            self.delta_latent1,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.delta_latent2,
            mean=0.0,
            std=0.02
        )

        # Multiplicative embedding: start as identity scaling
        nn.init.normal_(
            self.embeddings,
            mean=1.0,
            std=0.02
        )
    def forward_gru(self, inputs: torch.Tensor):
        z_final  = self.gru(inputs)[0] # [B, T, H]
        z_final = z_final.mean(dim=1) # [B, H]    avg pool over time dimension    
        z_final = self.vf_gru(z_final) # [B, V]

        return z_final
    
    def forward_orth_testing(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape # [B, Window, Sensors] e.g., [B, 2, 51]
        
        # --- 1. Step into Orthogonal Domain ---
        x_orth = self.orth_transformer(inputs)
        
        # --- 2. Apply Delta1 Latent Bias ---
        d1 = self.bias_proj(self.delta_latent1).unsqueeze(-2) 
        x_orth_biased = x_orth.unsqueeze(-2) + d1 # [B, P, 1, Order]
        
        # --- 3. Truly Dynamic Latent Generation ---
        # [B, P, 1, Order] -> [B, Order, P, 1]
        x_t = x_orth_biased.squeeze(-2).transpose(1, 2).unsqueeze(-1)
        
        # Project each sensor at each time step into H-space
        cond = self.temporal_proj(x_t) * self.embeddings.transpose(1, 2) 
        # --- 5. Prediction (Forecasting) ---
        z_final, _ = torch.max(cond, dim=1) # [B, P, H]

        # Apply the second Latent Bias to the forecast
        d2 = self.bias_proj(self.delta_latent2)
        v_pred = self.vf_orth(z_final) + d2 # [B, P, Order]

        preds_all_time = v_pred.transpose(1, 2) # [B, Order, P]
        
        # Final forecast is the last step of the predicted window
        preds = preds_all_time[:, -1, :] 

        return preds,cond
    
    def forward_orth(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape # [B, Window, Sensors] e.g., [B, 2, 51]
        
        # --- 1. Step into Orthogonal Domain ---
        x_orth = self.orth_transformer(inputs)
        
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
        #if not self.use_MoM:
        #z_final, _ = torch.max(cond, dim=1) # [B, P, H]
        #score = self.temporal_attention(cond)
        #weight = torch.softmax(score, dim=1)
        #z_final = (cond * weight).sum(dim=1)
        B, T, P, H = cond.shape

        score = self.temporal_attention(cond)  # [B,T,P,1]

        pos = torch.arange(T, device=cond.device).float()
        pos = pos.view(1, T, 1, 1)

        score = score + self.recency_bias * pos

        weight = torch.softmax(score, dim=1)

        z_final = (cond * weight).sum(dim=1)

        #else:
        #    z_final = self.mom(cond).max(dim=1).values
        
        # Apply the second Latent Bias to the forecast
        # vf(z_final) -> [B, P, Order]
        # d2 -> [1, P, Order]
        d2 = self.bias_proj2(self.delta_latent2)
        #cond = cond.permute(0,2,3,1).reshape(B,P,-1)
        v_pred = self.vf(z_final) + d2 # [B, P, Order]

        preds_all_time = self.orth_transformer.inverse(v_pred)
        
        # Final forecast is the last step of the predicted window
        preds = preds_all_time[:, -1, :] 
        coeffs_freq = coeffs_time[:, 0, :, :] # First step coefficients

        return preds, coeffs_time, coeffs_freq

    def _init_weights(self):
        """Xavier initialization for all learnable layers."""

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():

                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)

                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)

                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward_freq(self, inputs: torch.Tensor):
        freq = torch.fft.rfft(inputs, dim=1, norm="ortho") # [B,F,V]
        freq = torch.cat([freq.real, freq.imag], dim=-1)   # [B, F, 2V]                                 # magnitude
        z_final = self.linear_freq(freq) # [B, T, H]
        z_final = z_final.mean(dim=1) # [B, H]    avg pool over time dimension    
        z_final = self.vf_freq(z_final) # [B, V]
        return z_final

    def forward(self, inputs: torch.Tensor):
        #x = inputs.transpose(1, 2)
        #coeffs_time = torch.einsum('btp, btq -> btpq', x, x)
        #coeffs_time = torch.tanh(coeffs_time)

        # inputs: [B, T, V]
        # Global representation of the window
        route = inputs.mean(dim=1)             # [B,V]
        scores = self.router(route)            # [B,2]
        weights = torch.softmax(scores, dim=-1)
        w_gru = weights[:, 0:1]
        w_freq = weights[:, 1:2]
        #
        #z_final_gru = self.forward_gru(inputs)
        z_final_gru = self.forward_freq(inputs)
        z_final, coeffs_time, coeffs_freq = self.forward_orth(inputs)
        
        #z_final_freq = self.forward_freq(inputs)
        ##weighted combination of the two paths
        z_final = w_gru * z_final_gru + w_freq * z_final

        return z_final, coeffs_time, coeffs_time

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
        self.orth_transformer = self.options.get("orth_transformer", None)


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
        x = (
            self.orth_transformer(x)
            if self.orth_transformer
            else x.transpose(1,2)
        )

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
        if self.orth_transformer:
            pred = self.orth_transformer.inverse(pred)


        return pred[:,-1,:], coeff, coeff[:,0]

class MultiModalVLinear(nn.Module):
    def __init__(self, md, ld, td, N, order, h=128, device="cpu", opt=None):
        super().__init__()

        self.md, self.ld, self.td = md, ld, td
        self.N = N
        self.h = h

        # -----------------------------
        # Encoders (node-wise)
        # -----------------------------
        self.enc_m = nn.Sequential(
            nn.Linear(md, h), nn.GELU(), nn.Linear(h, h)
        )
        self.enc_l = nn.Sequential(
            nn.Linear(ld, h), nn.ReLU(), nn.Linear(h, h)
        )
        self.enc_t = nn.Sequential(
            nn.Linear(td, h), nn.GELU(), nn.Linear(h, h)
        )

        # modality embedding
        self.type_emb = nn.Parameter(torch.randn(1, 1, 1, 3, h))

        # -----------------------------
        # temporal + modality attention
        # -----------------------------
        self.q = nn.Linear(h, h)
        self.k = nn.Linear(h, h)
        self.v = nn.Linear(h, h)

        self.fuse = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, h)
        )

        # node-level projection
        latent_per_pod = opt.get("latent_per_pod")
        self.node_out = nn.Linear(h, latent_per_pod)

        self.gate = nn.Parameter(torch.tensor(0.1))

        # backbone (keeps your V-linear structure)
        self.backbone = vlinear(N*latent_per_pod, order, h, device, opt or {})

        self.total_features = md + ld + td

        self.decoder = nn.Sequential(
            nn.Linear(latent_per_pod, h),
            nn.GELU(),
            nn.Linear(h, self.total_features)
        )

    # -----------------------------
    # split concatenated input
    # -----------------------------
    def split(self, x):
        m, l = self.md, self.md + self.ld
        return x[..., :m], x[..., m:l], x[..., l:]

    # -----------------------------
    # attention over modalities
    # -----------------------------
    def attn(self, z):
        """
        z: (B, T, N, M, H)
        M = modalities (3)
        """

        q = F.elu(self.q(z)) + 1
        k = F.elu(self.k(z)) + 1
        v = self.v(z)

        # pool over (T, M)
        kv = torch.einsum("btnmd,btnme->bnde", k, v)

        k_sum = k.sum(dim=(1, 3))  # (B, N, H)

        norm = 1 / (torch.einsum("btnmh,bnh->btnm", q, k_sum) + 1e-6)

        out = torch.einsum("btnmh,bnhh,btnm->btnmh", q, kv, norm)

        return self.fuse(out)

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, x):
        """
        x: (B, T, N, Fm+Fl+Ft)
        """

        xm, xl, xt = self.split(x)

        # -------------------------------------------------
        # modality encoders
        # -------------------------------------------------
        m = self.enc_m(xm)   # (B,T,N,H)
        l = self.enc_l(xl)   # (B,T,N,H)
        t = self.enc_t(xt)   # (B,T,N,H)

        # -------------------------------------------------
        # stack modalities
        # -------------------------------------------------
        z = torch.stack([m, l, t], dim=3)  # (B,T,N,3,H)

        z = z + self.type_emb

        # -------------------------------------------------
        # modality attention
        # -------------------------------------------------
        z = self.attn(z)  # (B,T,N,3,H)

        # -------------------------------------------------
        # collapse modality dimension
        # -------------------------------------------------
        z = z.mean(dim=3)  # (B,T,N,H)

        # -------------------------------------------------
        # richer pod latent representation
        # -------------------------------------------------
        x_nodes = self.node_out(z)  # (B,T,N,D)

        B, T, N, D = x_nodes.shape

        # optional residual stabilization
        residual = x.mean(dim=-1, keepdim=True)  # (B,T,N,1)

        x_nodes = x_nodes + self.gate * residual

        # -------------------------------------------------
        # flatten pod latent space
        # -------------------------------------------------
        x_nodes = x_nodes.reshape(B, T, N * D)

        # -------------------------------------------------
        # backbone causal modeling
        # -------------------------------------------------
        pred_latent, coeffs_time, coeffs_freq = self.backbone(x_nodes)
        pred_latent = pred_latent.view(B, N, D) #latent -> pod latent
        pred = self.decoder(pred_latent)  # (B,N,F)

        # -------------------------------------------------
        # reshape coeffs back to pod structure
        # -------------------------------------------------
        coeffs_time = coeffs_time.view(B, T, N, D, N, D)

        # aggregate latent interactions
        coeffs_time = coeffs_time.mean(dim=(3, 5))  # (B,T,N,N)

        coeffs_freq = coeffs_freq.view(B, N, D, N, D)
        coeffs_freq = coeffs_freq.mean(dim=(2, 4))  # (B,N,N)

        return pred, coeffs_time, coeffs_freq