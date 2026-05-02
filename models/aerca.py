import json
import os
from models.senn import SENNGC
import torch.nn as nn
import torch
from utils.utils import (compute_kl_divergence, 
                         pot, topk, topk_at_step,write_results)
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import torch.nn.functional as F
from models.scoring import scoring
from models.statical_rca import StatisticalRCA

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

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
        self.total_params =0


        # Initialize log variances to 0 (which means initial weight is 1.0)
        self.log_var_recon = nn.Parameter(torch.zeros(1))
        self.log_var_sparse = nn.Parameter(torch.zeros(1))
        self.log_var_sdi = nn.Parameter(torch.zeros(1))
        self.log_var_smooth = nn.Parameter(torch.zeros(1))

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

        self.models_encoder_only = ["GVAR","vlinear"] 
        if(self.options["coeff_architecture"] in self.models_encoder_only):
            self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self.total_params = (self._count_parameters(self.encoder)  )

        if(self.options["coeff_architecture"] in ["deep_mlp"]):
            self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device).to(device)
            self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device).to(device)
            self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self._log_and_print('Number of parameters in decoder: {}', self._count_parameters(self.decoder))
            self._log_and_print('Number of parameters in decoder_prev: {}', self._count_parameters(self.decoder_prev))
            self.total_params = (self._count_parameters(self.encoder) +
                                 self._count_parameters(self.decoder) +
                                 self._count_parameters(self.decoder_prev)  )
            
        elif(self.options["coeff_architecture"] in ["TemporalGNN_Attention", "TemporalGNN_Attention_fourier", "TemporalGNN_Attention_crossattn","TemporalGNN_Attention_crossattn_Legendre","TemporalGNN_Attention_crossattn_enhanced","cuts_mlp","cuts_lstm"]):
            # --- Efficient attention-based decoder layers ---
            hidden_dim_small = 256
            # INCREASE RANK: 8 is too low for 30-50 vars. Try 16 to allow more complex interactions.
            self.rank = 16                 

            self.decoding_norm = nn.LayerNorm(hidden_dim_small).to(device)

            # FIX 1: Map hidden state back to EVERY sensor (num_vars), not just 1.
            self.decoding_output_proj = nn.Linear(hidden_dim_small, num_vars).to(device)

            # Coeff Projections
            self.decoding_coeff_proj = nn.Linear(hidden_dim_small, 2 * num_vars * self.rank).to(device)  
            self.coeff_proj_decoder = nn.Linear(hidden_dim_small, 2 * num_vars * self.rank).to(device)   

            # Input Projection: Needs to handle the dual residuals (2 * num_vars)
            # or just num_vars depending on your 'us' shape
            self.decoding_input_proj = nn.Linear(num_vars, hidden_dim_small).to(device)

            # FIX 2: Use the Orthogonal Basis (vf) correctly
            # If window is 25, order should be 25 or 1 depending on basis type
            order = window_size 
            self.vf = nn.Sequential(
                nn.Linear(hidden_dim_small, hidden_dim_small * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim_small * 2, order) 
            ).to(device)

            #self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self._log_and_print('Number of parameters in decoding_input_proj: {}', self._count_parameters(self.decoding_input_proj))
            #self._log_and_print('Number of parameters in decoding_attn: {}', self._count_parameters(self.decoding_attn))
            self._log_and_print('Number of parameters in decoding_output_proj: {}', self._count_parameters(self.decoding_output_proj))
            self._log_and_print('Number of parameters in decoding_coeff_proj: {}', self._count_parameters(self.decoding_coeff_proj))
            self._log_and_print('Number of parameters in decoding_norm: {}', self._count_parameters(self.decoding_norm))
            #self._log_and_print('Number of parameters in temporal_attn_decoder: {}', self._count_parameters(self.temporal_attn_decoder))
            self._log_and_print('Number of parameters in coeff_proj_decoder: {}', self._count_parameters(self.coeff_proj_decoder))


            self.total_params = (self._count_parameters(self.encoder) +
                                self._count_parameters(self.decoding_input_proj) +
                                #self._count_parameters(self.decoding_attn) +
                                self._count_parameters(self.decoding_output_proj) +
                                self._count_parameters(self.decoding_coeff_proj) +
                                self._count_parameters(self.decoding_norm)+
                                #self._count_parameters(self.temporal_attn_decoder) +
                                self._count_parameters(self.coeff_proj_decoder) +
                                self._count_parameters(self.vf)  )
        
        elif(self.options["coeff_architecture"] == "causalrca"):
            from models.causalrca import MLPDecoder
            self.decoder = MLPDecoder(
                n_in_node=None,
                n_in_z=1,
                n_out=1,
                data_variable_size=options.get("num_vars"),
                n_hid=options.get("outer_hidden_dim", 64),
            ).to(device)
            
            self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self._log_and_print('Number of parameters in decoder: {}', self._count_parameters(self.decoder))
            self.total_params = (self._count_parameters(self.encoder) +
                                 self._count_parameters(self.decoder)  )
            
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
        

        # Initialize
        self.log_var_recon = nn.Parameter(torch.zeros(1))
        self.log_var_sparse = nn.Parameter(torch.zeros(1))
        self.log_var_sdi = nn.Parameter(torch.zeros(1))
        self.log_var_smooth = nn.Parameter(torch.zeros(1))

        # If your class has self.device defined:
        self.log_var_recon.to(self.device)
        self.log_var_sparse.to(self.device)
        self.log_var_sdi.to(self.device)
        self.log_var_smooth.to(self.device)
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
        # coeffs shape: [Batch, Sensors, Sensors] -> [64, 30, 30]
        if coeffs.dim() == 3:
            # Subtract the coefficient matrix of sample 'j' from sample 'j+1'
            # This represents the change in sensor relationships over 1 timestep
            diff = coeffs[1:, :, :] - coeffs[:-1, :, :]
            
            # Calculate the norm and mean
            return torch.norm(diff, dim=(1, 2)).mean()
        
        # Fallback for 4D if you ever switch back
        return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()

    def encoding(self, xs):
        if isinstance(xs, np.ndarray):
            xs = torch.tensor(xs).float().to(self.device)
        if xs.dim() == 2: # for testing, where we test with single sample, we need to add batch dimension
            xs = xs.unsqueeze(0) 

        winds = xs[:, :-1, :] # input is all but last time step
        nexts = xs[:, -1, :] # target is the last time step

        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        preds, coeffs, attn_weights = self.encoder(winds)
        us = preds - nexts                    # shape: (B, hidden_size)

        if self.options["coeff_architecture"] in self.models_encoder_only:
            return us, coeffs, nexts, winds[:-self.window_size], attn_weights, preds
        else:
            return us, coeffs, nexts, winds, attn_weights, preds

    def decoding_1decoder(self, us, winds, add_u=True):
        # us shape: (B, p) -> [64, 30]
        # winds shape: (B, T, p) -> [64, 25, 30]
        batch_size, p = us.shape
        rank = self.rank

        # 1. Project the residual directly (No temporal dimension here)
        # us is (B, p), so u_proj becomes (B, hidden)
        u_proj = self.decoding_input_proj(us) 
        current_state = self.decoding_norm(u_proj)
        
        # 2. Sensor-wise predictions (B, p)
        # This matches your observed torch.Size([64, 30])
        preds = self.decoding_output_proj(current_state) 

        # 3. Low-rank Coefficients (The VLinear interaction)
        coeff_flat = self.decoding_coeff_proj(current_state)
        U, V = torch.split(coeff_flat, p * rank, dim=-1)
        
        # Reshape for matrix multiplication
        U = U.view(batch_size, p, rank)
        V = V.view(batch_size, rank, p)
        coeffs = torch.matmul(U, V) # (B, p, p)

        # 4. Final next-step prediction
        # nexts_hat = Linear Prediction + The Surprise (us)
        nexts_hat = preds + us if add_u else preds

        # Return None for prev_coeffs if you aren't using them in this mode
        return nexts_hat, coeffs, torch.zeros_like(coeffs)

    def decoding_causalrca(self, us, winds, add_u=True, aux_vars=None):
        """
        MLP-based CausalRCA decoding.

        Args:
            us:    latent states from encoder (B, T, p)
            winds: original sliding windows (B, T, p) — not used here
            add_u: residual addition flag
            aux_vars: dict containing encoder graph outputs:
                    {
                        "adj_A1",
                        "adj_A",
                        "adj_A_tilt",
                        "logits",
                        "enc_x",
                        "Wa",
                        "z",
                        "z_positive"
                    }

        Returns:
            nexts_hat:      (B_windowed, p)
            decoder_coeffs: mat_z from causal RCA (B_windowed, p)
            prev_coeffs:    zeros placeholder to match 1decoder signature (B_windowed, 1, p, p)
        """
        if aux_vars is None:
            raise ValueError("decoding_causalrca requires aux_vars from causalrca encoder.")

        B, p = us.shape  # latent includes temporal dimension

        # ----------------------
        # Extract encoder auxiliary variables
        # ----------------------
        origin_A   = aux_vars["adj_A1"]       # sinh(3A)
        adj_A_tilt = aux_vars["adj_A_tilt"]   # I - A^T
        Wa         = aux_vars["Wa"]           # learnable param
        input_z    = aux_vars["logits"]       # graph-weighted latent (B, T, p)

        # In decoding_causalrca
        # u_next comes from us (after sliding windows)
        input_z_windows = input_z  # NO slicing

        # Call MLPDecoder with aligned batch
        mat_z, preds, _ = self.decoder(
            inputs=None,
            input_z=input_z_windows,
            n_in_node=p,
            origin_A=origin_A,
            adj_A_tilt=adj_A_tilt,
            Wa=Wa
        )

        # Ensure preds has shape (B_windowed, p)
        if preds.dim() == 3:
            preds = preds.squeeze(-1)

        u_next = us   
        # Final prediction
        nexts_hat = preds + u_next if add_u else preds

        # prev_coeffs placeholder
        # Outer product to get full p x p matrix per batch
        mat_z_flat = mat_z.squeeze(-1) if mat_z.dim() == 3 else mat_z  # (B, p)
        decoder_coeffs = torch.einsum('bi,bj->bij', mat_z_flat, mat_z_flat)  # (B,p,p)
        decoder_coeffs = decoder_coeffs.unsqueeze(1)  # (B,1,p,p)
        prev_coeffs = torch.zeros(B, 1, p, p, device=us.device)


        return nexts_hat, decoder_coeffs, prev_coeffs

    
    def decoding_2decoders(self, us, winds, add_u=True):
        #u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = us[:, :-1, :]
        u_next = us[:, -1, :]

        preds, coeffs,_ = self.decoder(u_winds)
        prev_preds, prev_coeffs,_ = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
        return nexts_hat, coeffs, prev_coeffs

    def decoding(self, us, winds, add_u=True,aux_vars=None):
        if self.options["coeff_architecture"] in ["deep_mlp"]:
            return self.decoding_2decoders(us, winds, add_u=add_u)
        elif self.options["coeff_architecture"] in ["TemporalGNN_Attention", "TemporalGNN_Attention_fourier", "TemporalGNN_Attention_crossattn","TemporalGNN_Attention_crossattn_Legendre","TemporalGNN_Attention_crossattn_enhanced","cuts_mlp","cuts_lstm"]:
            return self.decoding_1decoder(us, winds, add_u=add_u)
        elif self.options["coeff_architecture"] == "causalrca":
            return self.decoding_causalrca(us, winds, add_u=add_u, aux_vars=aux_vars)

    def forward(self, x,add_u=True):
        us, encoder_coeffs, nexts, winds, attn_weights, preds = self.encoding(x)
        try:
            kl_div = compute_kl_divergence(us, self.device)
        except Exception as e:
            # In case of error, like when KL cannot be computed due to numerical issues, 
            # sometimes happens when lr is high (0.0005 for SWAT) instead of 0.0001
            print(f"Error computing KL divergence: {e}")
            kl_div = torch.tensor(0.0, device=self.device)

        if self.options["coeff_architecture"] == "causalrca":
            # as attnn_weights contains both the attn_weights and aux vars used by the decoder
            nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u,aux_vars=attn_weights[1])
            attn_weights = attn_weights[1]
        elif self.options["coeff_architecture"] in self.models_encoder_only:
            #no decoder, so return empty tensors
            nexts_hat = preds 
            decoder_coeffs = torch.tensor([])
            prev_coeffs = torch.tensor([])
        else:
            nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us, attn_weights
    
    def _training_step(self, x, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us, attns = self.forward(x, add_u=add_u)
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha) 
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        loss_prev_smooth = self._smoothness_loss(prev_coeffs) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        loss_kl = kl_div# if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        logging.info('KL loss: %s', loss_kl.item())

        # constraint DAG if causla RCA
        if self.options["coeff_architecture"] == "causalrca":
            from models import causalrca
            loss_DAG = causalrca._h_A(attns["adj_A1"])
        else:
            loss_DAG = torch.tensor(0.0)
        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl +  
                loss_DAG #only for causal RCA
                )
        logging.info('Total loss: %s', loss.item())

        losses_to_log = {
            "loss_recon": loss_recon.item(),
            "loss_encoder_coeffs": loss_encoder_coeffs.item(),
            "loss_decoder_coeffs": loss_decoder_coeffs.item(),
            "loss_prev_coeffs": loss_prev_coeffs.item(),
            "loss_encoder_smooth": loss_encoder_smooth.item(),
            "loss_decoder_smooth": loss_decoder_smooth.item(),
            "loss_prev_smooth": loss_prev_smooth.item(),
            "loss_kl": loss_kl.item(),
            "loss_DAG": loss_DAG.item() if self.options["coeff_architecture"] == "causalrca" else 0.0
        }
        tensorboard_log = {f'training_step/{key}': value for key, value in losses_to_log.items()}
        for key, value in tensorboard_log.items():
            self.writer.add_scalar(key, value, self.current_epoch)

        return loss, losses_to_log
    
    def _training(self, xs):
        if self.options["dataset_name"] in ["swat","smap","smd","wadi","msds","aiops","gaia"]:
            self._training_batches_swat(xs)
        else:
            raise ValueError(f"Unknown dataset {self.options['dataset']} for training")
        
    def _training_batches_swat(self, xs,batch_size=512):
        """
        xs: list of windows, each of shape (window_size+1, num_vars)
        batch_size: number of windows per batch
        """

        #if len(xs.shape) == 3:
        #    xs = np.concatenate(xs, axis=0)
        #    xs = torch.tensor(xs, dtype=torch.float32, device=self.device)
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
            for i in range(0, len(xs_train), batch_size):
                # xs_train shape
                # swat = (131, 1000, 51)
                # smd = (56672, 10, 38)
                batch_windows = xs_train[i:i+batch_size]
                x_batch = torch.tensor(batch_windows, dtype=torch.float32, device=self.device)  # (B, W, P)

                self.optimizer.zero_grad()
                #SWaT = torch.Size([131, 1000, 51])
                #SMD = torch.Size([1000, 10, 38])
                loss, _ = self._training_step(x_batch)
                loss.backward()
                try:
                    self.optimizer.step()
                # check if exception due to out of memory error 
                except Exception as e:
                    print(e)
                    if 'CUDA out of memory' in str(e):
                        self._log_and_print('Total parameters exceed 100 million, stopping training.')
                        ac_at = [0, 0, 0, 0]
                        k_at_step_all = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        write_results(self.options, self.local_model_name, ac_at, k_at_step_all, self.total_params, self.options.get("results_csv", 'RQ_swat_windows.csv'))
                        #stop the whole python program
                        os._exit(1)
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
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us, attn_weights = self.forward(x, add_u=add_u)

        #if label is not None:
        #    preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        #else:
        #    preprocessed_label = None
        # 2. LABEL ALIGNMENT FIX:
        # If x is [25, 30], label is also [25, 30].
        # We don't need a sliding window. We just need the label for the target step.
        if label is not None:
            # If label is [25, 30], we take the last timestamp to match 'nexts'
            if torch.is_tensor(label):
                preprocessed_label = label[-1:, :] 
            else:
                preprocessed_label = label[-1:]
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

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us, attn_weights

    def _get_recon_threshold(self, xs):
        self.eval()#(1,10000,10)
        losses_list = []
        with torch.no_grad():
            for x in xs:
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us,_ = self._testing_step(x, add_u=False)
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
                us = self._testing_step(x)[-2]
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
                _, nexts_hat, nexts, _, _, _, _, _, _ = self._testing_step(x, add_u=False)
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

    def _evaluate_rcd(self, xs, labels, bins=None, gamma=5, agg="mean"):
        """
        RCD baseline for root cause analysis with temporal windows preserved.
        - xs: ndarray of shape [N, T, P]  (N windows, T timesteps, P variables)
        - labels: ndarray of shape [N, T, P] (0=normal, 1=anomalous)
        - agg: str, aggregation method across time ("mean", "median", "last")
        """
        import pandas as pd
        from models.baselines.rcd import rca_with_rcd
        import numpy as np

        N, T, P = xs.shape

        # --- Aggregate across the time dimension ---
        if agg == "mean":
            X_all = xs.mean(axis=1)       # (N, P)
            y_all = labels.max(axis=1)    # (N, P), mark anomalous if anomaly in any timestep
        elif agg == "median":
            X_all = np.median(xs, axis=1) # (N, P)
            y_all = labels.max(axis=1)
        elif agg == "last":
            X_all = xs[:, -1, :]          # take last timestep per window
            y_all = labels[:, -1, :]
        else:
            raise ValueError(f"Unknown agg={agg}")

        # --- Masks at window level ---
        mask_normal = (y_all == 0).all(axis=-1)   # window normal if all vars=0
        mask_anom   = (y_all == 1).any(axis=-1)   # window anomalous if any var=1

        # --- Apply masks ---
        normal_X = X_all[mask_normal, :]
        anomalous_X = X_all[mask_anom, :]

        # --- Convert to DataFrame ---
        cols = [f"var{i}" for i in range(P)]
        normal_df = pd.DataFrame(normal_X, columns=cols)
        anomalous_df = pd.DataFrame(anomalous_X, columns=cols)

        # --- Run RCD ---
        result = rca_with_rcd(
            normal_df,
            anomalous_df,
            bins=bins,
            gamma=gamma,
            localized=False,
            verbose=False
        )

        return {
            "root_cause": result['root_cause'],
            "num_tests": result['tests'],
            "time": result['time']
        }


    def plot_case_study(self, z_scores, labels=None, attn_importance=None, mlp_scores=None, num_vars=None, threshold=0.1):
        """
        Plots variable importance for a single sample and overlays true root causes.

        Args:
            z_scores: array of model's latent variable importance (T, P)
            labels: array of ground truth (T, P)
            attn_importance: optional array of attention importance (P,)
            mlp_scores: optional array of baseline MLP importance (P,)
            num_vars: number of variables
            threshold: value above which a label is considered a root cause
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if num_vars is None:
            num_vars = z_scores.shape[1]

        # Aggregate z_scores over time (mean)
        mean_z = z_scores.mean(axis=0)

        x = np.arange(num_vars)
        width = 0.25
        plt.figure(figsize=(12, 5))

        plt.bar(x - width, mean_z, width, label='Summary Causal Graph')
        if attn_importance is not None:
            attn_per_var = attn_importance.mean(axis=0).mean(axis=0).mean(axis=-1) # mean over first 2 axes → shape (10,)
            plt.bar(x, attn_per_var, width, label='Attention')
        if mlp_scores is not None:
            plt.bar(x + width, mlp_scores, width, label='MLP per lag')

        # Highlight true root causes
        # print if label is not None
        print("Plotting case study with ground truth labels:", labels)
        if labels is not None:
            # aggregate labels over time
            mean_labels = labels.mean(axis=0)   # shape (40,)
            attn_arr = attn_per_var if attn_importance is not None else np.zeros_like(mean_z)
            mlp_arr = mlp_scores if mlp_scores is not None else np.zeros_like(mean_z)

            max_vals = np.maximum.reduce([mean_z, attn_arr, mlp_arr])
            root_causes = mean_labels > threshold

            plt.scatter(x[root_causes], max_vals[root_causes] + 0.05,
                        color='red', label='Ground truth')

            root_df = pd.DataFrame({
                "RootCauseX": x[root_causes],
                "RootCauseY": max_vals[root_causes] + 0.05
            })           

        plt.xlabel('Variable')
        plt.ylabel('Importance / Score')
        #plt.title('Case Study: Variable Importance Comparison')
        #save the plt as pdf
        plt.legend()
        coeff_architecture = self.options.get("coeff_architecture")
        dataset_name = self.options.get("dataset_name")
        plt.savefig("results/case_study_variable_importance("+dataset_name+")("+coeff_architecture+").pdf")
        plt.show()

        # Save data to CSV
        df = pd.DataFrame({
            "Variable": x,
            "SummaryCausalGraph": mean_z,
            "Attention": attn_arr,
            "MLP": mlp_arr,
        })

        df.to_csv("results/case_study_variable_importance_data("+dataset_name+")("+coeff_architecture+").csv", index=False)
        root_df.to_csv("results/case_study_root_causes("+dataset_name+")("+coeff_architecture+").csv", index=False)

    def plot_case_study_heatmap(self, z_scores, labels=None, attn_importance=None, num_vars=None):
        """
        Heatmap case study: shows variable importance over time + ground truth overlay.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if num_vars is None:
            num_vars = z_scores.shape[1]
        
        # Normalize scores for visualization
        norm_z = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min() + 1e-8)
        
        plt.figure(figsize=(14, 6))
        plt.imshow(norm_z.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label="Normalized z-score")
        plt.ylabel("Variable")
        plt.xlabel("Time step")
        
        # Overlay ground truth anomalies in red
        if labels is not None:
            anomaly_indices = np.where(labels > 0)
            plt.scatter(anomaly_indices[0], anomaly_indices[1], color="red", s=10, label="Ground Truth")
            plt.legend()
        
        coeff_architecture = self.options.get("coeff_architecture")
        dataset_name = self.options.get("dataset_name")
        plt.title(f"Case Study Heatmap ({dataset_name}, {coeff_architecture})")
        plt.savefig(f"results/case_study_heatmap({dataset_name})({coeff_architecture}).pdf")
        plt.show()

    def _testing_root_cause(self, xs, labels, alpha: float = 0.5, use_attention_fusion: bool = False):
        coeff_architecture = self.options["coeff_architecture"]
        
        # 1. Baseline check
        if coeff_architecture in ["rcd", "baro", "nsigma"]:
            if coeff_architecture == "rcd":
                res = StatisticalRCA.evaluate_rcd(xs, labels)
            elif coeff_architecture == "baro":
                res = StatisticalRCA.evaluate_baro(xs, labels)
            elif coeff_architecture == "nsigma":
                res = StatisticalRCA.evaluate_nsigma(xs, labels)

            if res:
                k_at_step_all = res["avg_k_at_step"]
                self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
                self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
                self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
                
                # Write results for the RQ tables
                write_results(self.options, self.local_model_name, 
                              [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]], 
                              k_at_step_all, 0, self.options.get("results_csv"))
            return res

        # 2. Model Loading & Setup
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device))
        self.eval()
        
        # Load normalization stats
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        us_list = []        # For global POT threshold
        us_sample_list = [] # For individual sample evaluation
        attn_list = []
        
        # 3. Inference Loop
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x = xs[i] # Current window: [25, 30]
                label = labels[i]
                
                # Forward pass - encoding handles unsqueeze and slicing
                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(x, label, add_u=False)
                
                # PRECISION FIX: us is [1, 30], we take the whole thing
                # No more [self.window_size:] slicing which resulted in empty arrays
                u_numpy = us.cpu().numpy() # [1, 30]
                us_sample_list.append(u_numpy)
                us_list.append(u_numpy)
                
                if use_attention_fusion:
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()
                    attn_list.append(attn_mean)

        # 4. Global POT Threshold Calculation
        us_all = np.concatenate(us_list, axis=0) # [Total_Windows, num_vars]
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        
        us_all_z_score_pot = []
        for i in range(self.num_vars):
            col_data = us_all_z_score[:, i]
            col_data = col_data[np.isfinite(col_data)]
            
            if col_data.size == 0:
                us_all_z_score_pot.append(0.0)
                continue
                
            try:
                pot_val, _ = pot(col_data, self.risk, self.initial_level, self.num_candidates)
            except:
                # Robust fallback: 3-Sigma
                pot_val = np.mean(col_data) + 3 * np.std(col_data)
            us_all_z_score_pot.append(pot_val)
        
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        # 5. Top-K Evaluation
        k_all = []
        k_at_step_all = []
        
        for i in tqdm(range(len(xs)), desc="Top-K Evaluation"):
            us_sample = us_sample_list[i] # [1, 30]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            
            if use_attention_fusion:
                attn_per_lag = attn_list[i].mean(axis=2)
                attn_importance = attn_per_lag.mean(axis=0)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(z_scores.shape[0], axis=0)
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance

            # LABEL ALIGNMENT FIX:
            # We are predicting the very last step of the input window i.
            # Therefore, we compare z_scores [1, 30] with the LAST row of labels[i].
            #current_labels = labels[i][-1:] # [1, 30]
            # Change this: current_labels = labels[i][-1:]
            # To this:
            # This takes the max across the window. If ANY sensor is an anomaly 
            # at ANY point in the 25-step window, we evaluate it.
            current_labels = np.max(labels[i], axis=0, keepdims=True)
            try:
                # TopK requires (Time, Vars) shapes. Both are [1, 30] here.
                k_lst = topk(z_scores, current_labels, us_all_z_score_pot)
                k_at_step = topk_at_step(z_scores, current_labels)
                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)
            except Exception as e:
                self._log_and_print("Error computing top-k for sample {}: {}", i, str(e))
                continue

        # 6. Result Aggregation
        valid_samples = len(k_all)
        total_samples = len(xs)
        
        self._log_and_print("RCA Coverage: {}/{} ({:.2f}%)", valid_samples, total_samples, (valid_samples/total_samples)*100)
        
        if valid_samples > 0:
            k_all = np.array(k_all).mean(axis=0)
            k_at_step_all = np.array(k_at_step_all).mean(axis=0)
            
            self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
            self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
            self._log_and_print('Root cause analysis AC@5: {:.5f}', k_at_step_all[4])
            self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
            
            # Save results
            write_results(self.options, self.local_model_name, [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]], 
                          k_at_step_all, self.total_params, self.options.get("results_csv"))
        else:
            self._log_and_print("Zero valid samples found. Check if labels[i][-1] contains any anomalies.")

    def _testing_root_cause_services_metrics(self, xs, labels, alpha: float = 0.5, use_attention_fusion: bool = False):
        # 0. Feature Mapping Setup
        mapping_path = '/home/db2003/Desktop/Amr/Tests/Medicine/dataset/aiops22-pre/初赛评分数据/idx_to_feature.json'
        with open(mapping_path, 'r') as f:
            self.idx_to_feature = json.load(f)
        feature_names = [self.idx_to_feature[str(i)] for i in range(self.num_vars)]

        coeff_architecture = self.options["coeff_architecture"]
        
        # 1. Baseline check
        if coeff_architecture == "rcd":
            rcd_result = self._evaluate_rcd(xs, labels, bins=None, gamma=5)
            return rcd_result

        # 2. Model Loading & Setup
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device))
        self.eval()
        
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        us_list = []        
        us_sample_list = [] 
        attn_list = []
        
        # 3. Inference Loop
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x = xs[i]
                label = labels[i]
                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(x, label, add_u=False)
                u_numpy = us.cpu().numpy() 
                us_sample_list.append(u_numpy)
                us_list.append(u_numpy)
                if use_attention_fusion:
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()
                    attn_list.append(attn_mean)

        # 4. Global POT Threshold Calculation
        us_all = np.concatenate(us_list, axis=0) 
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        
        us_all_z_score_pot = []
        for i in range(self.num_vars):
            col_data = us_all_z_score[:, i]
            col_data = col_data[np.isfinite(col_data)]
            if col_data.size == 0:
                us_all_z_score_pot.append(0.0)
                continue
            try:
                pot_val, _ = pot(col_data, self.risk, self.initial_level, self.num_candidates)
            except:
                pot_val = np.mean(col_data) + 3 * np.std(col_data)
            us_all_z_score_pot.append(pot_val)
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        # 5. Top-K Evaluation (Faithful to Original Loop)
        k_all = []
        k_at_step_all = []
        
        # Sub-level tracking
        results = {
            "service": {"top1": 0, "top3": 0, "top5": 0, "top10": 0},
            "metric": {"top1": 0, "top3": 0, "top5": 0, "top10": 0},
            "node": {"top1": 0, "top3": 0, "top5": 0, "top10": 0}
        }
        
        valid_samples = 0
        for i in tqdm(range(len(xs)), desc="Top-K Evaluation"):
            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            
            if use_attention_fusion:
                attn_per_lag = attn_list[i].mean(axis=2)
                attn_importance = attn_per_lag.mean(axis=0)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(z_scores.shape[0], axis=0)
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance

            current_labels = np.max(labels[i], axis=0, keepdims=True)
            
            # Ground Truth Check for valid_samples count
            if np.sum(current_labels) == 0: continue
            valid_samples += 1

            try:
                # Original Top-K Logic (Faithful)
                k_lst = topk(z_scores, current_labels, us_all_z_score_pot)
                k_at_step = topk_at_step(z_scores, current_labels)
                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)

                # --- Faithfully Integrated Multi-Level Logic ---
                gt_indices = np.where(current_labels[0] > 0)[0]
                gt_completes = [feature_names[idx] for idx in gt_indices]

                # Parsing helper based on: node.service-id-metric
                def parse(name):
                    node = name.split('.')[0]
                    service = name.split('.')[1].split("-")[0]
                    metric = name.split('-')[-1]
                    return node, service, metric

                gt_nodes = set(parse(m)[0] for m in gt_completes)
                gt_services = set(parse(m)[1] for m in gt_completes)
                gt_metrics = set(parse(m)[2] for m in gt_completes)

                sorted_indices = np.argsort(z_scores[0])[::-1]
                ranked_completes = [feature_names[idx] for idx in sorted_indices]

                # Ranked Sub-lists
                seen_n, r_nodes = set(), []
                seen_s, r_services = set(), []
                seen_m, r_metrics = set(), []

                for m in ranked_completes:
                    n, s, met = parse(m)
                    if n not in seen_n: r_nodes.append(n); seen_n.add(n)
                    if s not in seen_s: r_services.append(s); seen_s.add(s)
                    if met not in seen_m: r_metrics.append(met); seen_m.add(met)

                for k in [1, 3, 5, 10]:
                    if any(n in gt_nodes for n in r_nodes[:k]): results["node"][f"top{k}"] += 1
                    if any(s in gt_services for s in r_services[:k]): results["service"][f"top{k}"] += 1
                    if any(m in gt_metrics for m in r_metrics[:k]): results["metric"][f"top{k}"] += 1

            except Exception as e:
                self._log_and_print(f"Error for sample {i}: {str(e)}")
                continue

        # 6. Result Aggregation (Faithful Output)
        self._log_and_print("RCA Coverage: {}/{} ({:.2f}%)", valid_samples, len(xs), (valid_samples/len(xs))*100)
        
        if valid_samples > 0:
            k_at_step_all = np.array(k_at_step_all).mean(axis=0)
            
            # 6a. Original Logs
            self._log_and_print('--- COMPLETE LEVEL RCA ---')
            self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
            self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
            self._log_and_print('Root cause analysis AC@5: {:.5f}', k_at_step_all[4])
            self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])

            # 6b. New Sub-Level Logs
            for track in ["node", "service", "metric"]:
                self._log_and_print(f'\n--- {track.upper()} LEVEL RCA ---')
                for k in [1, 3, 5, 10]:
                    acc = results[track][f"top{k}"] / valid_samples
                    self._log_and_print(f'AC@{k}: {acc:.5f}')
            
            write_results(self.options, self.local_model_name, [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]], 
                          k_at_step_all, self.total_params, self.options.get("results_csv"))
        else:
            self._log_and_print("Zero valid samples found.")

    def plot_case(self,z_scores, labels, t_idx=None):
        """
        z_scores: shape (T, P)
        labels: shape (T, P)
        t_idx: time index to visualize (default: the most anomalous)
        """

        # Pick the time with max anomaly if none specified
        if t_idx is None:
            t_idx = np.argmax(z_scores.max(axis=1))

        scores = z_scores[t_idx]
        true_causes = np.where(labels[t_idx] == 1)[0]

        plt.figure(figsize=(14, 4))
        plt.bar(np.arange(len(scores)), scores)

        # draw red rectangles on true causes
        for c in true_causes:
            plt.gca().add_patch(
                plt.Rectangle(
                    (c - 0.4, 0), 0.8, scores[c],
                    fill=False, edgecolor='red', linewidth=2.5
                )
            )

        plt.xlabel("Variable Index")
        plt.ylabel("Fused z-score")
        plt.title("Case Study: Variable-level Root Cause Signal")
        os.makedirs("results/case_csv", exist_ok=True)
        plt.savefig(f"results/case_csv/case_study_{self.model_name}.pdf")
        
        #plt.show()

        # ----- SAVE TO JSON -----
        import json

        data_to_save = {
            "variable_idx": np.arange(len(scores)).tolist(),       # convert to Python list
            "z_score": scores.astype(float).tolist(),             # convert np.float to float
            "is_root_cause": labels[t_idx].astype(int).tolist()   # convert np.int64 to int
        }

        os.makedirs("results/case_json", exist_ok=True)
        json_file = f"results/case_json/case_study_{self.model_name}_t{t_idx}.json"
        with open(json_file, "w") as f:
            json.dump(data_to_save, f, indent=2)

        print(f"Case-study data saved to: {json_file}")


    def plot_case_heatmap(self, z_scores, labels, t_start, window=3):
        """
        z_scores: (T, P) array of all z-scores
        labels: (T, P) variable-level ground truth
        t_start: starting timestep of the window
        window: number of timesteps plotted
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import json, os

        T, P = z_scores.shape

        # Bound the window
        t_end = min(t_start + window, T)

        # Extract (window, P)
        z_win = z_scores[t_start:t_end]      # shape (W, P)
        labels_win = labels[t_start:t_end]   # shape (W, P)
        
        W = z_win.shape[0]

        # Normalize for better color visibility
        z_norm = (z_win - z_win.min()) / (z_win.max() - z_win.min() + 1e-8)

        plt.figure(figsize=(12, 6))

        # Note: transpose so:
        #   rows = vars (P)
        #   columns = window steps (W)
        plt.imshow(z_norm.T, aspect='auto', cmap='viridis_r', origin='upper')

        plt.colorbar(label="Normalized z-score")
        plt.xlabel("Window step (0 → W-1)")
        plt.ylabel("Variable index")

        # Overlay ground-truth anomalies
        anom = np.where(labels_win > 0)
        # anom: (time_idx, var_idx)
        # BUT heatmap is transposed → need to invert coordinates
        if len(anom[0]) > 0:
            t_coords = anom[0]      # x-axis (window steps)
            v_coords = anom[1]      # y-axis (variables)
            plt.scatter(t_coords, v_coords, color='red', s=15, label="Root cause")

        plt.title(f"RCA Heatmap Window={window}, t_start={t_start}")
        plt.legend(loc='upper right')

        os.makedirs("results/case_heatmap", exist_ok=True)
        fname = f"results/case_heatmap/{self.model_name}_t{t_start}_win{window}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

        # -------- SAVE JSON FOR REPRODUCTION ----------
        data_json = {
            "window_start": int(t_start),
            "window_end": int(t_end),
            "z_score_window": z_win.astype(float).tolist(),     # W×P values
            "root_cause_window": labels_win.astype(int).tolist() # W×P labels
        }

        os.makedirs("results/case_json", exist_ok=True)
        json_file = f"results/case_json/{self.model_name}_t{t_start}_win{window}.json"
        with open(json_file, "w") as f:
            json.dump(data_json, f, indent=2)

        print(f"[✓] Heatmap + JSON saved for window {t_start}:{t_end}")

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
