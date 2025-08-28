import os
from models.senn import SENNGC
import torch.nn as nn
import torch
import math
from utils.utils import (compute_kl_divergence, sliding_window_view_torch,
                         eval_causal_structure, eval_causal_structure_binary,
                         pot, topk,topk_no_threshold, topk_at_step,write_results)
from utils.utils_current import compute_correlated_kl, compute_mmd
from numpy.lib.stride_tricks import sliding_window_view
import logging
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import torch.nn.functional as F
from models.scoring import scoring

from sklearn.cluster import KMeans
import numpy as np
import torch


class AERCA(nn.Module):
    def __init__(self, num_vars: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 window_size: int, stride: int = 1, encoder_alpha: float = 0.5, decoder_alpha: float = 0.5,
                 encoder_gamma: float = 0.5, decoder_gamma: float = 0.5,
                 encoder_lambda: float = 0.5, decoder_lambda: float = 0.5,
                 beta: float = 0.5, lr: float = 0.0001, epochs: int = 100,
                 recon_threshold: float = 0.95, data_name: str = 'ld',
                 causal_quantile: float = 0.80, root_cause_threshold_encoder: float = 0.95,
                 root_cause_threshold_decoder: float = 0.95, initial_z_score: float = 3.0,
                 risk: float = 1e-2, initial_level: float = 0.98, num_candidates: int = 100, options=None):
        super(AERCA, self).__init__()
        self.device = device
        self.options = options if options is not None else {}
        self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers,args=options, device=device)
        self.num_vars = num_vars
        self.num_modalities = 3
        self.num_vars_mod = num_vars // self.num_modalities  # integer division
        self.hidden_size = hidden_layer_size  # latent size from each encoder
        """
        # One encoder per modality
        self.encoders = nn.ModuleList([
            SENNGC(self.num_vars_mod, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device).to(device)
            for _ in range(self.num_modalities)
        ])
        """
        # Projection layers to merge modalities
        # Projection layers to merge modalities
        self.us_proj = nn.Linear(self.num_modalities * self.hidden_size, self.hidden_size).to(device)

        # For coeffs, keep the original shape (B, 1, num_vars, num_vars)
        self.coeff_proj = nn.Linear(self.num_modalities * self.num_vars_mod, self.num_vars).to(device)

        # For winds: (B, window_size, num_vars)
        self.winds_proj = nn.Linear(self.num_modalities * self.num_vars_mod, self.num_vars).to(device)

        # For nexts: (B, num_vars)
        self.nexts_proj = nn.Linear(self.num_modalities * self.num_vars_mod, self.num_vars).to(device)




        if(self.options["coeff_architecture"] == "deep_mlp"):
            self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device).to(device)
            self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device).to(device)
            self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self._log_and_print('Number of parameters in decoder: {}', self._count_parameters(self.decoder))
            self._log_and_print('Number of parameters in decoder_prev: {}', self._count_parameters(self.decoder_prev))
            self.total_params = (self._count_parameters(self.encoder) +
                                 self._count_parameters(self.decoder) +
                                 self._count_parameters(self.decoder_prev)  )
            
        elif(self.options["coeff_architecture"] == "TemporalGNN_Attention"):
            # --- Efficient attention-based decoder layers ---
            hidden_dim_small = min(hidden_layer_size, 64)  # smaller hidden dim to reduce parameters
            rank = 1                 # low-rank for coefficient matrices

            self.decoding_input_proj = nn.Linear(num_vars, hidden_dim_small).to(device)

            self.decoding_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim_small, num_heads=2, batch_first=True
            ).to(device)

            self.decoding_norm = nn.LayerNorm(hidden_dim_small).to(device)

            self.temporal_attn_decoder = nn.MultiheadAttention(
                embed_dim=hidden_dim_small, num_heads=1, batch_first=True
            ).to(device)

            self.decoding_output_proj = nn.Linear(hidden_dim_small, num_vars).to(device)

            self.decoding_coeff_proj = nn.Linear(hidden_dim_small, 2 * num_vars * rank).to(device)  
            # produces U and V for low-rank coeffs

            self.coeff_proj_decoder = nn.Linear(hidden_dim_small, 2 * num_vars * rank).to(device)   
            # for prev_coeffs

            #self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self._log_and_print('Number of parameters in decoding_input_proj: {}', self._count_parameters(self.decoding_input_proj))
            self._log_and_print('Number of parameters in decoding_attn: {}', self._count_parameters(self.decoding_attn))
            self._log_and_print('Number of parameters in decoding_output_proj: {}', self._count_parameters(self.decoding_output_proj))
            self._log_and_print('Number of parameters in decoding_coeff_proj: {}', self._count_parameters(self.decoding_coeff_proj))
            self._log_and_print('Number of parameters in decoding_norm: {}', self._count_parameters(self.decoding_norm))
            self._log_and_print('Number of parameters in temporal_attn_decoder: {}', self._count_parameters(self.temporal_attn_decoder))
            self._log_and_print('Number of parameters in coeff_proj_decoder: {}', self._count_parameters(self.coeff_proj_decoder))


            self.total_params = (self._count_parameters(self.encoder) +
                                self._count_parameters(self.decoding_input_proj) +
                                self._count_parameters(self.decoding_attn) +
                                self._count_parameters(self.decoding_output_proj) +
                                self._count_parameters(self.decoding_coeff_proj) +
                                self._count_parameters(self.decoding_norm)+
                                self._count_parameters(self.temporal_attn_decoder) +
                                self._count_parameters(self.coeff_proj_decoder))
        print('----------------------------------')
        print(f'Total number of parameters in AERCA: {self.total_params}')
        print('----------------------------------')
        
        
        
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.stride = stride
        self.encoder_alpha = encoder_alpha
        self.decoder_alpha = decoder_alpha
        self.encoder_gamma = encoder_gamma
        self.decoder_gamma = decoder_gamma
        self.encoder_lambda = encoder_lambda
        self.decoder_lambda = decoder_lambda
        self.current_epoch = 0
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.recon_threshold = recon_threshold
        self.root_cause_threshold_encoder = root_cause_threshold_encoder
        self.root_cause_threshold_decoder = root_cause_threshold_decoder
        self.initial_z_score = initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.log_lambda_indep = nn.Parameter(torch.tensor(0.0))  # log of lambda_indep
        self.log_lambda_corr = nn.Parameter(torch.tensor(0.0))   # log of lambda_corr
        self.log_lambda_mmd = nn.Parameter(torch.tensor(0.0))     # log of lambda_mmd    
        self.alpha_param = nn.Parameter(torch.tensor(0.0))    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.encoder.to(self.device)
        #self.decoder.to(self.device)
        #self.decoder_prev.to(self.device)
        self.model_name = 'AERCA_' + data_name + '_ws_' + str(window_size) + '_stride_' + str(stride) + \
                          '_encoder_alpha_' + str(encoder_alpha) + '_decoder_alpha_' + str(decoder_alpha) + \
                          '_encoder_gamma_' + str(encoder_gamma) + '_decoder_gamma_' + str(decoder_gamma) + \
                          '_encoder_lambda_' + str(encoder_lambda) + '_decoder_lambda_' + str(decoder_lambda) + \
                          '_beta_' + str(beta) + '_lr_' + str(lr) + '_epochs_' + str(epochs) + \
                          '_hidden_layer_size_' + str(hidden_layer_size) + '_num_hidden_layers_' + \
                          str(num_hidden_layers)
        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates

        # Create an absolute path for saving models and thresholds
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(self.save_dir, exist_ok=True)
        correlated_KL =  "correlated_&_normal" if self.options['correlated_KL'] == 1 else "normal_KL"
        family_of_exp = data_name + str(self.options["coeff_architecture"]) + '_(no mean)_' + correlated_KL
        from datetime import datetime
        now = datetime.now()
        datetime_str = now.strftime("%d_%H%M%S_")

        self.local_model_name =family_of_exp + datetime_str+ f"{str(window_size)}_{str(lr)}_{str(self.options['seed'])}_window_{str(self.window_size)}" 
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "runs", self.local_model_name))

    def _count_parameters(self, model):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # view it with commas
        return num_params#f"{num_params:,}"
    
    def _log_and_print(self, msg, *args):
        """Helper method to log and print testing results."""
        final_msg = msg.format(*args) if args else msg
        logging.info(final_msg)
        print(final_msg)

    def _sparsity_loss(self, coeffs, alpha):
        norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
        norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
        return (1 - alpha) * norm2 + alpha * norm1

    def _smoothness_loss(self, coeffs):
        return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()

    def encoding_batch(self, xs):  # xs shape: (batch, T, num_vars)
        batch_windows = []
        for x in xs:  # each x: (T, num_vars)
            windows = sliding_window_view(x, (self.window_size + 1, self.num_vars))[:, 0, :, :]
            batch_windows.append(windows)
        return np.stack(batch_windows)  
        # shape: (batch, T - window_size, window_size+1, num_vars)

    def encoding(self, xs):
        #
        try:
            windows = self.encoding_batch(xs.cpu().numpy())
            winds = windows[:, 0, :-1, :]   # (1000, 30, 10)
            nexts = windows[:, 0, -1, :]    # (1000, 10)
        except:
            #when testing
            windows = sliding_window_view(xs, (self.window_size + 1, self.num_vars))[:, 0, :, :]
            winds = windows[:, :-1, :]
            nexts = windows[:, -1, :]
        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        preds, coeffs, attn_weights = self.encoder(winds)
        us = preds - nexts                    # shape: (B, hidden_size)
        """
        us.shape
            torch.Size([999, 51])
        coeffs.shape
            torch.Size([999, 1, 51, 51])
        nexts.shape
            torch.Size([999, 51])
        nexts[self.window_size:].shape
            torch.Size([998, 51])
        winds.shape
            torch.Size([999, 1, 51])
        winds[:-self.window_size].shape
            torch.Size([998, 1, 51])
        """
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size], attn_weights

    def encoding_new(self, xs):
        # Split features into modalities
        modalities = self.split_by_clusters(xs)
        
        us_list, coeff_list = [], []
        winds_list, nexts_list = [], []

        for i, x_mod in enumerate(modalities):
            # Sliding window
            windows = sliding_window_view(x_mod.cpu().numpy(), (self.window_size + 1, x_mod.shape[1]))[:, 0, :, :]
            winds = torch.tensor(windows[:, :-1, :]).float().to(self.device)
            nexts = torch.tensor(windows[:, -1, :]).float().to(self.device)

            preds, coeffs = self.encoders[i](winds)
            us = preds - nexts

            us_list.append(us)
            coeff_list.append(coeffs)
            winds_list.append(winds)
            nexts_list.append(nexts)

        # --- Combine modalities ---
        us = torch.cat(us_list, dim=-1)
        # coeff_list: list of (B, 1, vars_mod, vars_mod), e.g. 3 × [999, 1, 17, 17]
        B, C, _, _ = coeff_list[0].shape
        num_blocks = len(coeff_list)
        vars_mod = coeff_list[0].shape[-1]
        total_vars = num_blocks * vars_mod

        # Initialize empty block matrix
        coeffs = coeff_list[0].new_zeros((B, C, total_vars, total_vars))

        # Fill diagonal blocks
        for i, block in enumerate(coeff_list):
            start = i * vars_mod
            end = (i + 1) * vars_mod
            coeffs[:, :, start:end, start:end] = block

        winds_flat = torch.cat(winds_list, dim=-1)
        winds = self.winds_proj(winds_flat)

        nexts_flat = torch.cat(nexts_list, dim=-1)
        nexts = self.nexts_proj(nexts_flat)

        # Return shapes compatible with previous code
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size]

    def decoding_1decoder_norm2(self, us, winds, add_u=True, attn_dropout=0.1, residual_alpha=0.9):
        """
        Attention-based decoding replacing dual decoders.
        us: latent states (B, T, num_vars)
        winds: original windows (B, T, num_vars)
        """
        batch_size, p = us.shape
        rank = self.decoding_coeff_proj.out_features // (2 * p)  # dynamically recover rank

        # --- Sliding windows ---
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]  # (B, window, p)
        u_next = u_windows[:, -1, :]    # (B, p)

        # --- Project and attend ---
        u_proj = self.decoding_input_proj(u_winds)   # (B, window, hidden_dim)
        attn_out, _ = self.decoding_attn(u_proj, u_proj, u_proj)  # (B, window, hidden_dim)
        attn_out = F.dropout(attn_out, p=attn_dropout, training=self.training)

        # Residual + norm
        u_proj_resid = attn_out + residual_alpha * u_proj
        attn_norm = self.decoding_norm(u_proj_resid)

        query = attn_norm[:, -1:, :]   # (B, 1, hidden_dim)
        temp_out, _ = self.temporal_attn_decoder(query, attn_norm, attn_norm)  # (B, 1, hidden_dim)

        # Residual + norm again
        temp_out = temp_out + residual_alpha * query
        temp_out = self.decoding_norm(temp_out)

        # --- Predictions ---
        preds = self.decoding_output_proj(temp_out).squeeze(1)  # (B, p)

        # --- Low-rank coefficient reconstruction ---
        coeff_flat = self.decoding_coeff_proj(temp_out)  # (B, 1, 2 * p * rank)
        U, V = torch.split(coeff_flat, p * rank, dim=-1)
        U = U.view(-1, 1, p, rank)
        V = V.view(-1, 1, p, rank)
        coeffs = torch.matmul(U, V.transpose(-2, -1))   # (B, 1, p, p)

        # --- Previous coefficients from winds ---
        winds_proj = self.decoding_input_proj(winds)
        winds_attn, _ = self.decoding_attn(winds_proj, winds_proj, winds_proj)
        winds_attn = F.dropout(winds_attn, p=attn_dropout, training=self.training)

        winds_resid = winds_attn + residual_alpha * winds_proj
        winds_norm = self.decoding_norm(winds_resid)

        winds_temp, _ = self.temporal_attn_decoder(winds_norm[:, -1:, :], winds_norm, winds_norm)
        winds_temp = winds_temp + residual_alpha * winds_norm[:, -1:, :]
        winds_temp = self.decoding_norm(winds_temp)

        prev_flat = self.coeff_proj_decoder(winds_temp)  # (B, 1, 2 * p * rank)
        U_prev, V_prev = torch.split(prev_flat, p * rank, dim=-1)
        U_prev = U_prev.view(-1, 1, p, rank)
        V_prev = V_prev.view(-1, 1, p, rank)
        prev_coeffs = torch.matmul(U_prev, V_prev.transpose(-2, -1))  # (B, 1, p, p)

        prev_preds = self.decoding_output_proj(winds_temp).squeeze(1)  # (B, p)

        # --- Final next-step prediction ---
        nexts_hat = preds + u_next + prev_preds if add_u else preds + prev_preds

        return nexts_hat, coeffs, prev_coeffs

    def decoding_1decoder(self, us, winds, add_u=True, attn_dropout=0.1, residual_alpha=0.9):
        """
        Attention-based decoding replacing dual decoders.
        us: latent states (B, T, num_vars)
        winds: original windows (B, T, num_vars)
        """
        batch_size, p = us.shape
        rank = self.decoding_coeff_proj.out_features // (2 * p)  # dynamically recover rank

        # --- Sliding windows ---
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]  # (B, window, p)
        u_next = u_windows[:, -1, :]    # (B, p)

        # --- Project and attend ---
        u_proj = self.decoding_input_proj(u_winds)                   # (B, window, hidden_dim)
        attn_out, _ = self.decoding_attn(u_proj, u_proj, u_proj)    # (B, window, hidden_dim)
        attn_norm = self.decoding_norm(attn_out)

        query = attn_norm[:, -1:, :]                                   # (B, 1, hidden_dim)
        temp_out, _ = self.temporal_attn_decoder(query, attn_norm, attn_norm)  # (B, 1, hidden_dim)

        # --- Predictions ---
        preds = self.decoding_output_proj(temp_out).squeeze(1)        # (B, p)

        # --- Low-rank coefficient reconstruction ---
        coeff_flat = self.decoding_coeff_proj(temp_out)               # (B, 1, 2 * p * rank)
        U, V = torch.split(coeff_flat, p * rank, dim=-1)
        U = U.view(-1, 1, p, rank)
        V = V.view(-1, 1, p, rank)
        coeffs = torch.matmul(U, V.transpose(-2, -1))

        # --- Previous coefficients from winds ---
        winds_proj = self.decoding_input_proj(winds)
        winds_attn, _ = self.decoding_attn(winds_proj, winds_proj, winds_proj)
        winds_norm = self.decoding_norm(winds_attn)
        winds_temp, _ = self.temporal_attn_decoder(winds_norm[:, -1:, :], winds_norm, winds_norm)

        prev_flat = self.coeff_proj_decoder(winds_temp)              # (B, 1, 2 * p * rank)
        U_prev, V_prev = torch.split(prev_flat, p * rank, dim=-1)
        U_prev = U_prev.view(-1, 1, p, rank)
        V_prev = V_prev.view(-1, 1, p, rank)
        prev_coeffs = torch.matmul(U_prev, V_prev.transpose(-2, -1)) # (B, 1, p, p)
        prev_preds = self.decoding_output_proj(winds_temp).squeeze(1) # (B, p)

        # --- Final next-step prediction ---
        nexts_hat = preds + u_next + prev_preds if add_u else preds + prev_preds

        return nexts_hat, coeffs, prev_coeffs
    
    def decoding_2decoders(self, us, winds, add_u=True):
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]

        preds, coeffs = self.decoder(u_winds)
        prev_preds, prev_coeffs = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
        return nexts_hat, coeffs, prev_coeffs


    def decoding_norm_residual(self, us, winds, add_u=True):
        """
        Attention-based decoding replacing dual decoders.
        us: latent states (B, T, num_vars)
        winds: original windows (B, T, num_vars)
        """
        _, p = us.shape
        attn_dropout = 0.1  # dropout for attention layers
        rank = self.decoding_coeff_proj.out_features // (2 * p)  # dynamically recover rank

        # --- Sliding windows ---
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]  # (B, window, p)
        u_next = u_windows[:, -1, :]    # (B, p)

        # --- Project and attend ---
        u_proj = self.decoding_input_proj(u_winds)                   # (B, window, hidden_dim)
        attn_out, _ = self.decoding_attn(u_proj, u_proj, u_proj)    
        attn_out = F.dropout(attn_out, p=attn_dropout, training=self.training)
        attn_norm = self.decoding_norm(attn_out)

        # --- Temporal attention with residual scaling ---
        query = attn_norm[:, -1:, :]
        temp_out, _ = self.temporal_attn_decoder(query, attn_norm, attn_norm)
        alpha = 0.9
        temp_out = temp_out + alpha * query  # residual connection
        temp_out = F.layer_norm(temp_out, temp_out.shape[-1:])

        # --- Predictions ---
        preds = self.decoding_output_proj(temp_out).squeeze(1)

        # --- Low-rank coefficient reconstruction ---
        coeff_flat = self.decoding_coeff_proj(temp_out)
        U, V = torch.split(coeff_flat, p * rank, dim=-1)
        U = U.view(-1, 1, p, rank)
        V = V.view(-1, 1, p, rank)
        coeffs = torch.matmul(U, V.transpose(-2, -1))

        # --- Previous coefficients from winds ---
        winds_proj = self.decoding_input_proj(winds)
        winds_attn, _ = self.decoding_attn(winds_proj, winds_proj, winds_proj)
        winds_attn = F.dropout(winds_attn, p=attn_dropout, training=self.training)
        winds_norm = self.decoding_norm(winds_attn)

        winds_temp, _ = self.temporal_attn_decoder(winds_norm[:, -1:, :], winds_norm, winds_norm)
        winds_temp = winds_temp + alpha * winds_norm[:, -1:, :]
        winds_temp = F.layer_norm(winds_temp, winds_temp.shape[-1:])

        prev_flat = self.coeff_proj_decoder(winds_temp)
        U_prev, V_prev = torch.split(prev_flat, p * rank, dim=-1)
        U_prev = U_prev.view(-1, 1, p, rank)
        V_prev = V_prev.view(-1, 1, p, rank)
        prev_coeffs = torch.matmul(U_prev, V_prev.transpose(-2, -1))
        prev_preds = self.decoding_output_proj(winds_temp).squeeze(1)

        # --- Final next-step prediction ---
        nexts_hat = preds + u_next + prev_preds if add_u else preds + prev_preds

        return nexts_hat, coeffs, prev_coeffs

    def decoding(self, us, winds, add_u=True):
        if self.options["coeff_architecture"] == "deep_mlp":
            return self.decoding_2decoders(us, winds, add_u=add_u)
        elif self.options["coeff_architecture"] == "TemporalGNN_Attention":
            return self.decoding_1decoder(us, winds, add_u=add_u)

    def decoding_batch(self, us, winds, add_u=True):
        # us: (B, P)
        us_seq = us.unsqueeze(1)  # shape -> (B, 1, P)

        preds, coeffs = self.decoder(us_seq)           # next-step prediction from latent
        prev_preds, prev_coeffs = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + us + prev_preds
        else:
            nexts_hat = preds + prev_preds

        return nexts_hat, coeffs, prev_coeffs

    def forward(self, x,add_u=True):
        us, encoder_coeffs, nexts, winds, attn_weights = self.encoding(x)
        if(self.options["correlated_KL"] == 1):
            kl_indep = compute_kl_divergence(us,self.device)  
            latent_dim = us.shape[1]
            split = latent_dim // 2

            lambda_indep = torch.exp(self.log_lambda_indep)
            lambda_corr = torch.exp(self.log_lambda_corr)
            lambda_mmd = torch.exp(self.log_lambda_mmd)
            shrinkage=self.options["shrinkage"]

            kl_corr = compute_correlated_kl(us, shrinkage=shrinkage)
            # Weighted combination
            s = (us[:, 0] > us[:, 0].median()).long()

            us_0 = us[s == 0]
            us_1 = us[s == 1]

            #fair_loss = compute_mmd(us_0, us_1)  # MMD loss between the two groups

            kl_div = lambda_indep * kl_indep + lambda_corr * kl_corr
            #kl_div = kl_div + lambda_mmd * fair_loss
        else:
            # --- KL divergence with independent prior ---
            kl_div = compute_kl_divergence(us, self.device)
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us
    
    def _training_step(self, x,add_u=True):
        # Forward pass
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)

        # === Full reconstruction loss ===
        loss_full_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss (full): %s', loss_full_recon.item())

        # === Mean/Std reconstruction loss (optional) ===
        if self.options.get("mean_std_recon_loss", False):
            mean_target = nexts.mean(dim=1, keepdim=True)
            std_target  = nexts.std(dim=1, keepdim=True)
            mean_hat = nexts_hat.mean(dim=1, keepdim=True)
            std_hat  = nexts_hat.std(dim=1, keepdim=True)
            loss_mean = self.mse_loss(mean_hat, mean_target)
            loss_std  = self.mse_loss(std_hat, std_target)
            loss_stats_recon = loss_mean + loss_std
            logging.info('Reconstruction loss (mean+std): %s', loss_stats_recon.item())
            alpha = torch.sigmoid(self.alpha_param)
            loss_recon = alpha * loss_full_recon + (1 - alpha) * loss_stats_recon
            logging.info('Blended reconstruction loss: %s (alpha=%.4f)' % (loss_recon.item(), alpha.item()))
        else:
            loss_recon = loss_full_recon
            loss_stats_recon = torch.tensor(0.0)
            alpha = torch.tensor(0.0)

        # === Sparsity losses ===
        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        loss_prev_coeffs    = self._sparsity_loss(prev_coeffs, self.decoder_alpha)

        # === Smoothness losses ===
        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        loss_prev_smooth    = self._smoothness_loss(prev_coeffs)

        # === KL divergence loss ===
        loss_kl = kl_div

        # === Regularization ===
        reg_lambda = 0.01 * (self.log_lambda_indep ** 2 + self.log_lambda_corr ** 2)

        # === Latent AMOC loss (optional) ===
        if self.options.get("AMOC_Loss", False):
            diffs = (us[1:, :] - us[:-1, :]).pow(2).mean(dim=-1)
            latent_disc_loss = (diffs.sum() - diffs.max()) / diffs.shape[0]
            lambda_amoc = self.options.get("lambda_amoc", 0.1)
        else:
            latent_disc_loss = torch.tensor(0.0)
            lambda_amoc = 0.0

        # === Total loss ===
        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl +
                reg_lambda +
                lambda_amoc * latent_disc_loss)

        # === Logging all losses ===
        losses_dict = {
            "loss_full_recon": loss_full_recon.item(),
            "loss_stats_recon": loss_stats_recon.item(),
            "alpha": alpha.item(),
            "loss_encoder_coeffs": loss_encoder_coeffs.item(),
            "loss_decoder_coeffs": loss_decoder_coeffs.item(),
            "loss_prev_coeffs": loss_prev_coeffs.item(),
            "loss_encoder_smooth": loss_encoder_smooth.item(),
            "loss_decoder_smooth": loss_decoder_smooth.item(),
            "loss_prev_smooth": loss_prev_smooth.item(),
            "loss_kl": loss_kl.item(),
            "reg_lambda": reg_lambda.item(),
            "latent_disc_loss": latent_disc_loss.item()
        }

        return loss, losses_dict

    def _training(self, xs):
        if len(xs) == 1:
            xs_train = xs[:, :int(0.8 * len(xs[0]))]
            xs_val = xs[:, int(0.8 * len(xs[0])):]
        else:
            xs_train = xs[:int(0.8 * len(xs))]
            xs_val = xs[int(0.8 * len(xs)):]

        #xs_array = np.concatenate([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in xs_train], axis=0)
        #self.cluster_assignments = self.cluster_modalities(xs_array, n_clusters=self.num_modalities)  # fixed split

        best_val_loss = np.inf
        count = 0
        for epoch in tqdm(range(self.epochs), desc=f'Epoch'):
            count += 1
            epoch_loss = 0
            self.current_epoch = epoch
            self.train()
            for x in xs_train:
                self.optimizer.zero_grad()
                loss,_ = self._training_step(x)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            logging.info('Epoch %s/%s', epoch + 1, self.epochs)
            logging.info('Epoch training loss: %s', epoch_loss)
            logging.info('-------------------')
            epoch_val_loss = 0
            losses_dict_validation = defaultdict(list)
            self.eval()
            with torch.no_grad():
                for x in xs_val:
                    loss, losses_dict = self._training_step(x)
                    for key, value in losses_dict.items():
                        if key not in losses_dict_validation:
                            losses_dict_validation[key] = 0
                        losses_dict_validation[key] += value
                    epoch_val_loss += loss.item()
            self.writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            for key, value in losses_dict_validation.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            logging.info('Epoch val loss: %s', epoch_val_loss)
            logging.info('-------------------')
            if epoch_val_loss < best_val_loss:
                count = 0
                logging.info(f'Saving model at epoch {epoch + 1}')
                if self.options["early_stopping"]: #AERCA paper style early stopping
                    best_val_loss = epoch_val_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            if count >= 20:
                print('Early stopping')
                break
            if epoch % 5 == 0:
                self.writer.flush()
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device))
        logging.info('Training complete')
        #self._get_recon_threshold(xs_val)
        #self._get_root_cause_threshold_encoder(xs_val)
        #self._get_root_cause_threshold_decoder(xs_val)

    def cluster_modalities(self, xs, n_clusters=4, random_state=42):
        """
        Cluster metrics (columns) into modalities using KMeans.
        
        Args:
            xs: np.ndarray of shape (num_samples, num_vars)
            n_clusters: int, number of clusters/modalities
            random_state: int, for reproducibility
        
        Returns:
            cluster_assignments: np.ndarray of shape (num_vars,), mapping each metric to a cluster
        """
        # Transpose so that columns are "samples" for clustering
        X_cols = xs.T  # shape: (num_vars, num_samples)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(X_cols)
        
        cluster_assignments = kmeans.labels_  # shape: (num_vars,)
        return cluster_assignments

    def split_by_clusters(self, x):
        """
        Split features into fixed-size modalities.
        x: (num_samples, num_vars) tensor or ndarray
        Returns: list of tensors, one per modality
        """
        modalities = []
        start = 0
        for i in range(self.num_modalities):
            end = start + self.num_vars_mod
            modalities.append(x[:, start:end])
            start = end
        # Convert to tensor if needed
        modalities = [m if isinstance(m, torch.Tensor) else torch.tensor(m).float().to(self.device)
                    for m in modalities]
        return modalities



    def _training_batches(self, xs,batch_size=200):
        """
        xs: list of windows, each of shape (window_size+1, num_vars)
        batch_size: number of windows per batch
        """
        # Split into train and validation
        split_idx = int(0.8 * len(xs))
        xs_train = xs[:split_idx]
        xs_val = xs[split_idx:]

        best_val_loss = np.inf
        count = 0

        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            count += 1
            self.current_epoch = epoch
            self.train()
            epoch_loss = 0

            # Shuffle training windows
            np.random.shuffle(xs_train)

            # --- Training loop with batching ---
            for i in tqdm(range(0, len(xs_train), batch_size)):
                batch_windows = xs_train[i:i+batch_size]
                x_batch = torch.tensor(batch_windows, dtype=torch.float32, device=self.device)  # (B, W, P)

                self.optimizer.zero_grad()
                loss, _ = self._training_step(x_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            logging.info('Epoch %s/%s', epoch + 1, self.epochs)
            logging.info('Epoch training loss: %s', epoch_loss)

            # --- Validation loop ---
            self.eval()
            val_loss = 0
            losses_dict_validation = defaultdict(float)
            with torch.no_grad():
                for i in range(0, len(xs_val), batch_size):
                    batch_windows = xs_val[i:i+batch_size]
                    x_batch = torch.tensor(batch_windows, dtype=torch.float32, device=self.device)
                    loss, losses_dict = self._training_step(x_batch)
                    val_loss += loss.item()
                    for k, v in losses_dict.items():
                        losses_dict_validation[k] += v

            self.writer.add_scalar('Loss/val', val_loss, epoch)
            for k, v in losses_dict_validation.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)

            logging.info('Epoch val loss: %s', val_loss)

            # --- Early stopping ---
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    early_stop_count = 0
            #    logging.info(f'Saving model at epoch {epoch + 1}')
            #    torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            #else:
            #    early_stop_count += 1
            #    if early_stop_count >= 20:
            #        print('Early stopping')
            #        break
            if val_loss < best_val_loss:
                count = 0
                logging.info(f'Saving model at epoch {epoch + 1}')
                if self.options["early_stopping"]: #AERCA paper style early stopping
                    best_val_loss = val_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            if count >= 20:
                print('Early stopping')
                break
            if epoch % 5 == 0:
                self.writer.flush()

        # --- Load best model ---
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device))
        logging.info('Training complete')

        # --- Compute thresholds ---
        #self._get_recon_threshold(xs_val)
        #self._get_root_cause_threshold_encoder(xs_val)
        #self._get_root_cause_threshold_decoder(xs_val)



    def _testing_step(self, x, label=None, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)

        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        if (self.options["coeff_architecture"] == "deep_mlp"):
            loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
            logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

            loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
            logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

            loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
            logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

            loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
            logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

            loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
            logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

            loss_prev_smooth = self._smoothness_loss(prev_coeffs)
            logging.info('Prev smooth loss: %s', loss_prev_smooth.item())
            loss = (loss_recon +
                    self.encoder_lambda * loss_encoder_coeffs +
                    self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                    self.encoder_gamma * loss_encoder_smooth +
                    self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                    self.beta * loss_kl)
        else:
            loss = (loss_recon +
                    self.beta * loss_kl)
            logging.info('Total loss: %s', loss.item())

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us


    def _testing_step_(self, x, label=None, add_u=True):
        # Forward pass
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us, attn_weights = self.forward(x, add_u=add_u)
        # Compute mean and std targets for anomaly detection
        mean_target = nexts.mean(dim=1, keepdim=True)
        std_target = nexts.std(dim=1, keepdim=True)

        # Predict mean and std from the decoder output (assuming nexts_hat has same shape)
        mean_hat = nexts_hat.mean(dim=1, keepdim=True)
        std_hat = nexts_hat.std(dim=1, keepdim=True)

        # Reconstruction loss on mean and std
        loss_mean = self.mse_loss(mean_hat, mean_target)
        loss_std = self.mse_loss(std_hat, std_target)
        loss_recon = loss_mean + loss_std
        logging.info('Reconstruction loss (mean+std): %s', loss_recon.item())

        # KL divergence loss
        #loss_kl = kl_div
        #logging.info('KL loss: %s', loss_kl.item())

        # Encoder/decoder coefficient losses and smoothness (for deep_mlp)
        if self.options["coeff_architecture"] == "deep_mlp":
            loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
            logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

            loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
            logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

            loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
            logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

            loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
            logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

            loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
            logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

            loss_prev_smooth = self._smoothness_loss(prev_coeffs)
            logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

            loss = (loss_recon +
                    self.encoder_lambda * loss_encoder_coeffs +
                    self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                    self.encoder_gamma * loss_encoder_smooth +
                    self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) 
                    )
        else:
            # Simple reconstruction + KL loss
            loss = loss_recon #+ self.beta * loss_kl
            logging.info('Total loss: %s', loss.item())

        # Keep preprocessed label for evaluation if needed
        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us



    def _get_recon_threshold(self, xs):
        self.eval()#(1,10000,10)
        losses_list = []
        with torch.no_grad():
            for x in xs:
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us = self._testing_step(x, add_u=False)
                loss_arr = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy().ravel()
                losses_list.append(loss_arr)
        recon_losses = np.concatenate(losses_list)
        self.recon_threshold_value = np.quantile(recon_losses, self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_threshold.npy'), self.recon_threshold_value)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_mean.npy'), self.recon_mean)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_std.npy'), self.recon_std)

    def _get_root_cause_threshold_encoder(self, xs):
        self.eval()
        us_list = []
        with torch.no_grad():
            for x in xs:
                us = self._testing_step(x)[-1]
                us_list.append(us.cpu().numpy())
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self.lower_encoder = np.quantile(us_all, (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.upper_encoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_encoder.npy'), self.lower_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_encoder.npy'), self.upper_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'), self.us_mean_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'), self.us_std_encoder)

    def _get_root_cause_threshold_decoder(self, xs):
        self.eval()
        diff_list = []
        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x, add_u=False)
                diff = (nexts - nexts_hat).cpu().numpy().ravel()
                diff_list.append(diff)
        us_all = np.concatenate(diff_list, axis=0).reshape(-1, self.num_vars)
        self.lower_decoder = np.quantile(us_all, (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.upper_decoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.us_mean_decoder = np.mean(us_all, axis=0)
        self.us_std_decoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_decoder.npy'), self.lower_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_decoder.npy'), self.upper_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_decoder.npy'), self.us_mean_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_decoder.npy'), self.us_std_decoder)


    def _get_recon_threshold_batch(self, xs):
        self.eval()
        losses_list = []
        with torch.no_grad():
            for x in xs:
                # x is now (window_size, P), expand to batch of 1
                x_batch = x.unsqueeze(0) if torch.is_tensor(x) else torch.tensor(x).unsqueeze(0).float().to(self.device)
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x_batch, add_u=False)
                loss_arr = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy().ravel()
                losses_list.append(loss_arr)
        recon_losses = np.concatenate(losses_list)
        self.recon_threshold_value = np.quantile(recon_losses, self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_threshold.npy'), self.recon_threshold_value)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_mean.npy'), self.recon_mean)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_std.npy'), self.recon_std)


    def _get_root_cause_threshold_encoder_batch(self, xs):
        self.eval()
        us_list = []
        with torch.no_grad():
            for x in xs:
                x_batch = x.unsqueeze(0) if torch.is_tensor(x) else torch.tensor(x).unsqueeze(0).float().to(self.device)
                us = self._testing_step(x_batch)[-1]  # latent residuals
                us_list.append(us.cpu().numpy())
        us_all = np.concatenate(us_list, axis=0)  # shape: (total_samples, P)
        self.lower_encoder = np.quantile(us_all, (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.upper_encoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_encoder.npy'), self.lower_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_encoder.npy'), self.upper_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'), self.us_mean_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'), self.us_std_encoder)


    def _get_root_cause_threshold_decoder_batch(self, xs):
        self.eval()
        diff_list = []
        with torch.no_grad():
            for x in xs:
                x_batch = x.unsqueeze(0) if torch.is_tensor(x) else torch.tensor(x).unsqueeze(0).float().to(self.device)
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x_batch, add_u=False)
                diff = (nexts - nexts_hat).cpu().numpy().ravel()
                diff_list.append(diff)
        us_all = np.concatenate(diff_list, axis=0)
        self.lower_decoder = np.quantile(us_all, (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.upper_decoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.us_mean_decoder = np.mean(us_all, axis=0)
        self.us_std_decoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_decoder.npy'), self.lower_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_decoder.npy'), self.upper_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_decoder.npy'), self.us_mean_decoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_decoder.npy'), self.us_std_decoder)


    def _testing_root_cause(self, xs, labels,alpha: float = 0.5, use_attention_fusion: bool = False):
        # Load model and only the encoder-related parameters required for the POT computations.
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                        map_location=self.device))
        self.eval()
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        # Collect the latent representations from each sample.
        us_list = []
        us_sample_list = []
        attn_list = []
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us = self._testing_step(x, label, add_u=False)
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())
                if use_attention_fusion:
                    # aggregate attention over time (mean across timesteps)
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()  # shape [num_vars]
                    attn_list.append(attn_mean)

        # Combine all latent representations for POT threshold computation.
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self._log_and_print('=' * 50)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        us_all_z_score_pot = []
        for i in range(self.num_vars):
            pot_val, _ = pot(us_all_z_score[:, i], self.risk, self.initial_level, self.num_candidates)
            us_all_z_score_pot.append(pot_val)
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        # Compute top-k statistics for each sample using the computed POT thresholds.
        k_all = []
        k_at_step_all = []
        for i in range(len(xs)):
            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            if use_attention_fusion:
                # broadcast attn to match z_scores shape [T, num_vars]
                #attn_importance = attn_list[i].mean(axis=(0, 1))  # mean over lags and “from” vars → shape (P,)
                #attn_importance = np.expand_dims(attn_importance, axis=0).repeat(z_scores.shape[0], axis=0)  # (T, P)
                # attn_seq: shape (O, P, P) -> (lags, to_vars, from_vars)
                # mean over the "from" dimension → importance per "to" variable per lag
                attn_per_lag = attn_list[i].mean(axis=2)  # shape (O, P)

                # Then mean over lags
                attn_importance = attn_per_lag.mean(axis=0)  # shape (P,)

                # Broadcast to match z_scores (T, P)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(z_scores.shape[0], axis=0)
                """
                # Assume z_scores[0] and attn_importance[0] are lists of length P
                z_scores_list = z_scores[0].tolist()
                attn_list = attn_importance[0].tolist()

                # Sort descending and keep track of indices
                z_scores_sorted = sorted(enumerate(z_scores_list), key=lambda x: x[1], reverse=True)
                attn_sorted = sorted(enumerate(attn_list), key=lambda x: x[1], reverse=True)

                print("Top variables by z_scores:")
                for idx, val in z_scores_sorted[:10]:  # top 10
                    print(f"Var {idx}: {val:.4f}")

                print("\nTop variables by attention:")
                for idx, val in attn_sorted[:10]:  # top 10
                    print(f"Var {idx}: {val:.4f}")
                """
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance
            else:
                z_scores = z_scores
            k_lst = topk(z_scores, labels[i][self.window_size * 2:], us_all_z_score_pot)
            k_at_step = topk_at_step(z_scores, labels[i][self.window_size * 2:])
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)
        k_all = np.array(k_all).mean(axis=0)
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)
        ac_at = [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]]
        self._log_and_print('Root cause analysis AC@1: {:.5f}', ac_at[0])
        self._log_and_print('Root cause analysis AC@3: {:.5f}', ac_at[1])
        self._log_and_print('Root cause analysis AC@5: {:.5f}', ac_at[2])
        self._log_and_print('Root cause analysis AC@10: {:.5f}', ac_at[3])
        self._log_and_print('Root cause analysis Avg@10: {:.5f}', np.mean(k_at_step_all))

        ac_star_at = [k_all[0], k_all[9], k_all[99], k_all[499]]
        self._log_and_print('Root cause analysis AC*@1: {:.5f}', ac_star_at[0])
        self._log_and_print('Root cause analysis AC*@10: {:.5f}', ac_star_at[1])
        self._log_and_print('Root cause analysis AC*@100: {:.5f}', ac_star_at[2])
        self._log_and_print('Root cause analysis AC*@500: {:.5f}', ac_star_at[3])
        self._log_and_print('Root cause analysis Avg*@500: {:.5f}', np.mean(k_all))
        write_results(self.options,self.local_model_name,ac_at,k_at_step_all,self.total_params,'RQ_swat_windows.csv')

    def _testing_root_cause_new(self, xs, labels, alphas=np.arange(0, 1.1, 0.1), use_attention_fusion=True, sample_idx_for_plot=0):
        # Load model and encoder stats
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                        map_location=self.device))
        self.eval()
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        # Collect latent representations and attention weights
        us_list = []
        us_sample_list = []
        attn_list = []

        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us, attn_weights = self._testing_step(x, label, add_u=False)
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())
                if use_attention_fusion:
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()  # shape [num_vars]
                    attn_list.append(attn_mean)

        # POT threshold computation
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        us_all_z_score_pot = [pot(us_all_z_score[:, i], self.risk, self.initial_level, self.num_candidates)[0] for i in range(self.num_vars)]
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        # Sweep over alphas
        ac1_list, ac3_list, ac5_list, ac10_list = [], [], [], []

        for alpha in alphas:
            k_all = []
            k_at_step_all = []
            for i in range(len(xs)):
                us_sample = us_sample_list[i]
                z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)

                if use_attention_fusion:
                    attn_importance = attn_list[i]
                    attn_importance = np.expand_dims(attn_importance, axis=0).repeat(z_scores.shape[0], axis=0)
                    attn_importance = attn_importance.reshape(1, -1)  # (1, 51)
                    z_scores = alpha * z_scores + (1 - alpha) * attn_importance

                k_lst = topk(z_scores, labels[i][self.window_size*2:], us_all_z_score_pot)
                k_at_step = topk_at_step(z_scores, labels[i][self.window_size*2:])
                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)

            k_all_mean = np.array(k_all).mean(axis=0)
            k_at_step_mean = np.array(k_at_step_all).mean(axis=0)
            ac1_list.append(k_at_step_mean[0])
            ac3_list.append(k_at_step_mean[2])
            ac5_list.append(k_at_step_mean[4])
            ac10_list.append(k_at_step_mean[9])

        # Plot AC@K vs alpha
        plt.figure(figsize=(8,5))
        plt.plot(alphas, ac1_list, '-o', label='AC@1')
        plt.plot(alphas, ac3_list, '-o', label='AC@3')
        plt.plot(alphas, ac5_list, '-o', label='AC@5')
        plt.plot(alphas, ac10_list, '-o', label='AC@10')
        plt.xlabel('Alpha (weight for z-score)')
        plt.ylabel('AC@K')
        plt.title(f'AC@K vs Alpha for {self.model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Visualize variable-level fusion for a sample
        latent_sample = (-(us_sample_list[sample_idx_for_plot] - self.us_mean_encoder) / self.us_std_encoder)
        if use_attention_fusion:
            attn_sample = attn_list[sample_idx_for_plot]
            fused_sample = alpha * latent_sample + (1 - alpha) * np.expand_dims(attn_sample, axis=0).repeat(latent_sample.shape[0], axis=0)
        else:
            fused_sample = latent_sample
            attn_sample = np.zeros_like(latent_sample[0])

        # Plot per-variable scores
        plt.figure(figsize=(12,4))
        plt.plot(normalize(latent_sample.mean(axis=0)), label='Latent z-score')
        plt.plot(normalize(attn_sample), label='Attention importance')
        plt.plot(normalize(fused_sample.mean(axis=0)), label='Fused score', linewidth=2)
        plt.scatter(np.where(labels[sample_idx_for_plot][self.window_size*2:]==1)[0],
                    normalize(fused_sample.mean(axis=0))[labels[sample_idx_for_plot][self.window_size*2:]==1],
                    color='red', label='True anomalies')
        plt.xlabel('Variable index')
        plt.ylabel('Score')
        plt.title(f'Variable-level latent vs attention vs fused for sample {sample_idx_for_plot}')
        plt.legend()
        plt.show()

    def _testing_root_cause_(self, xs, labels):
        # Load model and encoder stats
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                        map_location=self.device))
        self.eval()
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        # Collect latent representations
        us_list = []
        us_sample_list = []
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                us = self._testing_step(x, label, add_u=False)[-1]
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())

        # Combine latent representations for RCA
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        import pandas as pd
        import time
        data_for_rca = pd.DataFrame(us_all, columns=[f"var_{i}" for i in range(self.num_vars)])

        # Determine anomaly index (example: first anomalous step)
        anomaly_idx = 0  

        # Run RCA
        sorted_scores = self.run_rca(anomaly_idx, data_for_rca, data_for_rca)

        # Convert sorted_scores to a dict for easier indexing
        score_dict = dict(sorted_scores)

        # Compute top-k metrics for each sample
        k_all = []
        k_at_step_all = []
        for i in range(len(xs)):
            # Use the same columns as the latent space
            z_scores = np.array([score_dict[f"var_{j}"] for j in range(self.num_vars)])
            k_lst = topk_no_threshold(z_scores, labels[i][self.window_size * 2:])
            k_at_step = topk_at_step(z_scores, labels[i][self.window_size * 2:])
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)

        k_all = np.array(k_all).mean(axis=0)
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)

        # Log metrics
        ac_at = [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]]
        self._log_and_print('Root cause analysis AC@1: {:.5f}', ac_at[0])
        self._log_and_print('Root cause analysis AC@3: {:.5f}', ac_at[1])
        self._log_and_print('Root cause analysis AC@5: {:.5f}', ac_at[2])
        self._log_and_print('Root cause analysis AC@10: {:.5f}', ac_at[3])
        self._log_and_print('Root cause analysis Avg@10: {:.5f}', np.mean(k_at_step_all))

        ac_star_at = [k_all[0], k_all[9], k_all[99], k_all[499]]
        self._log_and_print('Root cause analysis AC*@1: {:.5f}', ac_star_at[0])
        self._log_and_print('Root cause analysis AC*@10: {:.5f}', ac_star_at[1])
        self._log_and_print('Root cause analysis AC*@100: {:.5f}', ac_star_at[2])
        self._log_and_print('Root cause analysis AC*@500: {:.5f}', ac_star_at[3])
        self._log_and_print('Root cause analysis Avg*@500: {:.5f}', np.mean(k_all))

        write_results(self.options, self.local_model_name, ac_at, k_at_step_all, self.total_params, 'RQ_swat_windows.csv')


    def run_rca(self, anomaly, data, data_scaled):
        scores = scoring(data=data, data_scaled=data_scaled, anomaly=anomaly)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_scores


    def run_for_datapath(datapath, args):
        args.datapath = datapath

        data, data_scaled, inject_time = prepare_data(datapath=datapath)
            
        if args.ad is None or args.ad == "inject":
            anomaly = inject_time
        else:
            dataset = datapath.strip(os.sep).split(os.sep)[3]

            complexity = "simple" if "simple" in datapath else "full"
            anomalies_path = f"./evaluation_ad/{args.ad}_{dataset}_{complexity}.txt"

            anomalies = None
            with open(anomalies_path, "r") as file:
                for line in file:
                    if args.datapath in line.lower():
                        anomalies = line.strip()
                        break
            
            anomalies = re.search(r'\[(.*?)\]', anomalies).groups()[0]
            anomaly = anomalies.split(",")[0]
            anomaly = int(anomaly)    
        
        rca_start = time()
        sorted_scores = run_rca(args, anomaly, data, data_scaled)
        rca_end = time()    

        return datapath, rca_end-rca_start, sorted_scores


    def _testing_causal_discover(self, xs, causal_struct_value):
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                        map_location=self.device))
        self.eval()
        encoder_causal_list = []
        with torch.no_grad():
            for x in xs:
                # Only the encoder coefficients are used for causal discovery
                _, _, _, encoder_coeffs, _, _, _, _ = self._testing_step(x)
                encoder_estimate = torch.max(torch.median(torch.abs(encoder_coeffs), dim=0)[0],
                                             dim=0).values.cpu().numpy()
                encoder_causal_list.append(encoder_estimate)
        encoder_causal_struct_estimate_lst = np.stack(encoder_causal_list, axis=0)

        encoder_auroc = []
        encoder_auprc = []
        encoder_hamming = []
        encoder_f1 = []
        for i in range(len(encoder_causal_struct_estimate_lst)):
            encoder_auroc_temp, encoder_auprc_temp = eval_causal_structure(
                a_true=causal_struct_value, a_pred=encoder_causal_struct_estimate_lst[i])
            encoder_auroc.append(encoder_auroc_temp)
            encoder_auprc.append(encoder_auprc_temp)
            encoder_q = np.quantile(encoder_causal_struct_estimate_lst[i], q=self.causal_quantile)
            encoder_a_hat_binary = (encoder_causal_struct_estimate_lst[i] >= encoder_q).astype(float)
            _, _, _, _, ham_e = eval_causal_structure_binary(a_true=causal_struct_value,
                                                             a_pred=encoder_a_hat_binary)
            encoder_hamming.append(ham_e)
            encoder_f1.append(f1_score(causal_struct_value.flatten(), encoder_a_hat_binary.flatten()))
        self._log_and_print('Causal discovery F1: {:.5f} std: {:.5f}',
                            np.mean(encoder_f1), np.std(encoder_f1))
        self._log_and_print('Causal discovery AUROC: {:.5f} std: {:.5f}',
                            np.mean(encoder_auroc), np.std(encoder_auroc))
        self._log_and_print('Causal discovery AUPRC: {:.5f} std: {:.5f}',
                            np.mean(encoder_auprc), np.std(encoder_auprc))
        self._log_and_print('Causal discovery Hamming Distance: {:.5f} std: {:.5f}',
                            np.mean(encoder_hamming), np.std(encoder_hamming))