import json
import time
import os
from models.senn import SENNGC
import torch.nn as nn
import torch
from utils.utils import (compute_kl_divergence, 
                         pot, topk, topk_at_step,topk_at_step_multi_modality_new,write_results)
from utils.utils import XSDataset
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import torch.nn.functional as F
from models.statical_rca import StatisticalRCA

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import threading    
class AERCA(nn.Module):
    def __init__(self, num_vars: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 window_size: int, stride: int = 1, encoder_alpha: float = 0.5, decoder_alpha: float = 0.5,
                 encoder_gamma: float = 0.5, decoder_gamma: float = 0.5,
                 encoder_lambda: float = 0.5, decoder_lambda: float = 0.5,
                 beta: float = 0.5, lr: float = 0.0001, epochs: int = 100,
                 recon_threshold: float = 0.95, data_name: str = 'ld',
                 causal_quantile: float = 0.80, root_cause_threshold_encoder: float = 0.95,
                 root_cause_threshold_decoder: float = 0.95, initial_z_score: float = 3.0,
                 risk: float = 1e-2, initial_level: float = 0.98, num_candidates: int = 100, graph_structure=None, options=None):
        super(AERCA, self).__init__()
        self.device = device
        self.options = options if options is not None else {}
        self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers,args=options, graph_structure=graph_structure, device=device)
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

        self.models_encoder_only = ["GVAR","vlinear","cLSTM","cMLP","CUTS_PLUS"] 
        self.models_simple_next_step = ["cLSTM","cMLP"]
        if(self.options["coeff_architecture"] in self.models_encoder_only):
            self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
            self.total_params = (self._count_parameters(self.encoder)  )

        if(self.options["coeff_architecture"] in ["deep_mlp"]):
            self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, graph_structure=graph_structure, device=device).to(device)
            self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, graph_structure=graph_structure,   device=device).to(device)
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
        self.model_name =  self.options["coeff_architecture"] + '_' + data_name + '_ws_' + str(window_size) + '_seed_' + str(self.options['seed']) + '_numvars_' + str(num_vars) 
        
        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates

        # Create an absolute path for saving models and thresholds
        if self.options["exp_name"] != "":
            self.save_dir = os.path.join(os.getcwd(), 'saved_models', self.options["exp_name"])
        else:
            self.save_dir = os.path.join(os.getcwd(), 'saved_models','')
        os.makedirs(self.save_dir, exist_ok=True)
        correlated_KL =  "correlated_&_normal" if self.options['correlated_KL'] == 1 else "normal_KL"
        family_of_exp = data_name + str(self.options["coeff_architecture"]) + '_(no mean)_' + correlated_KL
        from datetime import datetime
        now = datetime.now()
        datetime_str = now.strftime("%d_%H%M%S_")

        self.local_model_name =self.model_name + "_" + datetime_str

        # --- SETUP LOGGING TO FILE ---
        # 1. Ensure the directory results_journal/logs exists
        log_dir = os.path.join("results_journal", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 2. Construct full log file path using self.local_model_name
        log_filepath = os.path.join(log_dir, f"{self.local_model_name}.log")

        # 3. Configure file handler for the python logging module
        file_handler = logging.FileHandler(log_filepath, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )

        # 4. Attach handler to the root logger and set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove any existing file handlers to prevent writing to multiple files across runs
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)

        root_logger.addHandler(file_handler)
        # -----------------------------

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "runs", self.local_model_name))
        self.init_weights()

    def init_weights(self):
        print("Initializing weights...")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
        
        # CRITICAL for vlinear: Initialize raw nn.Parameters
        for name, param in self.named_parameters():
            if 'delta_latent' in name or 'embeddings' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.normal_(param.data)


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

        # -------------------------
        # MULTIMODAL MODE
        # -------------------------
        if isinstance(coeffs, dict):
            loss = 0.0
            n = 0

            for k, v in coeffs.items():
                if v is None:
                    continue

                norm2 = torch.mean(torch.norm(v, dim=1, p=2))
                norm1 = torch.mean(torch.norm(v, dim=1, p=1))

                loss = loss + (1 - alpha) * norm2 + alpha * norm1
                n += 1

            return loss / max(n, 1)

        # -------------------------
        # SINGLE MODE (metrics only)
        # -------------------------
        norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
        norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))

        return (1 - alpha) * norm2 + alpha * norm1

    def _sparsity_loss_cLSTM(self, W, alpha):
        group_norm = torch.norm(W, dim=0, p=2)
        return torch.sum(group_norm)

    def _sparsity_loss_cMLP(self, W, alpha):
        group_norm = torch.norm(W, dim=(0, 2), p=2)
        return torch.sum(group_norm)
    
    def _smoothness_loss(self, coeffs):

        # -------------------------
        # MULTIMODAL MODE
        # -------------------------
        if isinstance(coeffs, dict):
            loss = 0.0
            n = 0

            for k, v in coeffs.items():
                if v is None:
                    continue

                if v.dim() == 3:
                    diff = v[1:] - v[:-1]
                    loss = loss + torch.norm(diff, dim=(1, 2)).mean()
                else:
                    loss = loss + torch.norm(
                        v[:, 1:] - v[:, :-1], dim=1
                    ).mean()

                n += 1

            return loss / max(n, 1)

        # -------------------------
        # SINGLE MODE (metrics only)
        # -------------------------
        if coeffs.dim() == 3:
            diff = coeffs[1:] - coeffs[:-1]
            return torch.norm(diff, dim=(1, 2)).mean()

        return torch.norm(coeffs[:, 1:] - coeffs[:, :-1], dim=1).mean()

    def encoding(self, xs):
        
        if "include_logs_and_traces" in self.options and self.options["include_logs_and_traces"]:
                if isinstance(xs, np.ndarray):
                    xs = torch.tensor(xs).float()

                if xs.dim() == 3:
                    # (T, Modalities, Num_vars) → add batch
                    xs = xs.unsqueeze(0)

                elif xs.dim() == 4:
                    # already (B, T, Modalities, Num_vars)
                    pass

                else:
                    raise ValueError(f"Invalid multimodal shape: {xs.shape}")
        else:
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

    def decoding_2decoders(self, nexts, winds, add_u=True):
        #u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = winds
        u_next = nexts

        preds, coeffs,_ = self.decoder(u_winds)
        prev_preds, prev_coeffs,_ = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
        return nexts_hat, coeffs, prev_coeffs

    def decoding(self, us, nexts, winds, add_u=True,aux_vars=None):
        if self.options["coeff_architecture"] in ["deep_mlp"]:
            return self.decoding_2decoders(nexts, winds, add_u=add_u)

    def forward(self, x,add_u=True):
        us, encoder_coeffs, nexts, winds, attn_weights, preds = self.encoding(x)
        try:
            if "include_logs_and_traces" in self.options and self.options["include_logs_and_traces"]:
                if self.options["coeff_architecture"] in self.models_simple_next_step:
                    kl_div = torch.tensor(0.0, device=self.device)
                else:
                    us_kl = us.mean(dim=1)
                    kl_div = compute_kl_divergence(us_kl, self.device)
            else:
                kl_div = compute_kl_divergence(us, self.device)
        except Exception as e:
            # In case of error, like when KL cannot be computed due to numerical issues, 
            # sometimes happens when lr is high (0.0005 for SWAT) instead of 0.0001
            print(f"Error computing KL divergence: {e}")
            kl_div = torch.tensor(0.0, device=self.device)

        if self.options["coeff_architecture"] in self.models_encoder_only:
            #no decoder, so return empty tensors
            nexts_hat = preds 
            decoder_coeffs = torch.tensor([])
            prev_coeffs = torch.tensor([])
        else:
            nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, nexts,winds, add_u=add_u)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us, attn_weights
    
    def _training_step(self, x, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us, attns = self.forward(x, add_u=add_u)
        loss_recon = self.mse_loss(nexts_hat, nexts)
        #logging.info('Reconstruction loss: %s', loss_recon.item())

        if self.options["coeff_architecture"] in self.models_simple_next_step:
            loss_encoder_coeffs = torch.zeros((), device=self.device)
        else:
            loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        #logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())
        if self.options["coeff_architecture"] == "cLSTM":
            for net in self.encoder.coeff_net.networks:
                loss_encoder_coeffs += self._sparsity_loss_cLSTM(
                    net.lstm.weight_ih_l0,
                    torch.tensor(self.encoder_alpha, device=net.lstm.weight_ih_l0.device)
                )
        if self.options["coeff_architecture"] == "cMLP":
            for net in self.encoder.coeff_net.networks:
                loss_encoder_coeffs += self._sparsity_loss_cMLP(
                    net.layers[0].weight,
                    torch.tensor(
                        self.encoder_alpha,
                        device=net.layers[0].weight.device
                    )
                )
        loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        #logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        #logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        loss_encoder_smooth = self._smoothness_loss(encoder_coeffs) if self.options["coeff_architecture"] not in self.models_simple_next_step else torch.tensor(0.0)
        #logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        loss_decoder_smooth = self._smoothness_loss(decoder_coeffs) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        #logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        loss_prev_smooth = self._smoothness_loss(prev_coeffs) if self.options["coeff_architecture"] not in self.models_encoder_only else torch.tensor(0.0)
        #logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        loss_kl = kl_div if self.options["coeff_architecture"] not in self.models_simple_next_step else torch.tensor(0.0)
        #logging.info('KL loss: %s', loss_kl.item())


        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl
                )
        #logging.info('Total loss: %s', loss.item())

        losses_to_log = {
            "loss_recon": loss_recon.item(),
            "loss_encoder_coeffs": loss_encoder_coeffs.item(),
            "loss_decoder_coeffs": loss_decoder_coeffs.item(),
            "loss_prev_coeffs": loss_prev_coeffs.item(),
            "loss_encoder_smooth": loss_encoder_smooth.item(),
            "loss_decoder_smooth": loss_decoder_smooth.item(),
            "loss_prev_smooth": loss_prev_smooth.item(),
            "loss_kl": loss_kl.item()
        }
        tensorboard_log = {f'training_step/{key}': value for key, value in losses_to_log.items()}
        for key, value in tensorboard_log.items():
            self.writer.add_scalar(key, value, self.current_epoch)

        return loss, losses_to_log
    
    def _training(self, xs):
        if self.options["dataset_name"] in ["swat","wadi","batadal"]:
            self._training_batches_swat(xs, self.options.get("batch_size"))
        else:
            raise ValueError(f"Unknown dataset {self.options['dataset']} for training")
        
    def _training_batches_swat(self, xs, batch_size=256):
        import time
        import numpy as np
        import psutil
        import threading
        import os
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from collections import defaultdict
        from tqdm import tqdm

        print(f"batch_size: {batch_size}, dataset length: {len(xs)}")
        # =========================================================
        # SPLIT DATA (KEEP ON CPU)
        # =========================================================
        split_idx = int(0.8 * len(xs))
        xs_train = xs[:split_idx]
        xs_val = xs[split_idx:]

        # Convert once (CPU tensors ONLY)
        xs_train = torch.tensor(xs_train, dtype=torch.float32)
        xs_val = torch.tensor(xs_val, dtype=torch.float32)

        # =========================================================
        # DATALOADERS (correct + fast)
        # =========================================================
        if "include_logs_and_traces" in self.options and self.options["include_logs_and_traces"]:
            # to help with stability of torch coversion for multimodality
            train_loader = DataLoader(
                XSDataset(xs_train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,   # keep small for now
                pin_memory=True,
                persistent_workers=False  # IMPORTANT for debugging
            )
            val_loader = DataLoader(
                XSDataset(xs_val),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False
            )
        else:
            print("preparing the dataloader")
            train_loader = DataLoader(
                TensorDataset(xs_train),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
                persistent_workers=True
            )

            val_loader = DataLoader(
                TensorDataset(xs_val),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )

        # =========================================================
        # METRICS
        # =========================================================
        best_val_loss = float("inf")
        early_stop_counter = 0

        epoch_times = []
        train_losses = []

        process = psutil.Process(os.getpid())

        peak_mem_bytes = {"value": 0}
        stop_event = threading.Event()

        def memory_poller():
            while not stop_event.is_set():
                mem = process.memory_info().rss
                peak_mem_bytes["value"] = max(peak_mem_bytes["value"], mem)
                time.sleep(0.1)

        monitor_thread = None
        if not torch.cuda.is_available():
            monitor_thread = threading.Thread(target=memory_poller)
            monitor_thread.start()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # =========================================================
        # TRAIN LOOP
        # =========================================================
        print("Starting training loop...")
        for epoch in tqdm(range(self.epochs), desc="Epoch"):

            self.current_epoch = epoch
            self.train()

            epoch_start = time.time()
            epoch_loss = 0.0

            # -----------------------------
            # TRAIN STEP
            # -----------------------------
            for (x_batch,) in tqdm(train_loader, desc="Batch"):

                x_batch = x_batch.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                loss, _ = self._training_step(x_batch)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.detach()

            epoch_loss = epoch_loss.item() / len(train_loader)

            # =========================================================
            # LOGGING
            # =========================================================
            epoch_time = time.time() - epoch_start
            # save time in seconds 
            epoch_times.append(epoch_time)
            train_losses.append(epoch_loss)

            cpu_mem_mb = process.memory_info().rss / (1024 ** 2)

            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Time/epoch", epoch_time, epoch)



            # =========================================================
            # VALIDATION
            # =========================================================
            self.eval()
            val_loss = 0.0
            losses_dict_validation = defaultdict(float)

            with torch.no_grad():
                for (x_batch,) in val_loader:

                    x_batch = x_batch.to(self.device, non_blocking=True)

                    loss, losses_dict = self._training_step(x_batch)

                    val_loss += loss.item()

                    for k, v in losses_dict.items():
                        losses_dict_validation[k] += v
            val_loss /= len(val_loader)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            for k, v in losses_dict_validation.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            logging.info(
                "Epoch %d/%d | Loss: %.6f | Time: %.3fs | CPU Mem: %.2f MB | Val Loss: %.6f",
                epoch + 1, self.epochs, epoch_loss, epoch_time, cpu_mem_mb, val_loss
            )
            # =========================================================
            # EARLY STOPPING
            # =========================================================
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0

                torch.save(
                    self.state_dict(),
                    os.path.join(self.save_dir, f"{self.model_name}.pt"),
                )
            else:
                early_stop_counter += 1

            if early_stop_counter >= 20:
                print("Early stopping triggered.")
                logging.info("Early stopping triggered at epoch %d", epoch + 1)
                break

            if epoch % 5 == 0:
                self.writer.flush()

        # =========================================================
        # STOP MONITORING
        # =========================================================
        if monitor_thread is not None:
            stop_event.set()
            monitor_thread.join()

        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mem_mb = peak_mem_bytes["value"] / (1024 ** 2)

        # =========================================================
        # FINAL METRICS
        # =========================================================
        total_train_time = sum(epoch_times)
        avg_epoch_time = np.mean(epoch_times)
        train_throughput = len(xs_train) / avg_epoch_time if avg_epoch_time > 0 else 0

        self.training_metrics = {
            "total_train_time": total_train_time,
            "avg_epoch_time": avg_epoch_time,
            "train_throughput": train_throughput,
            "peak_mem_mb": peak_mem_mb,
        }

        self._log_and_print("Training complete")
        self._log_and_print("Avg Epoch Time: {:.4f}s", avg_epoch_time)
        self._log_and_print("Throughput: {:.2f} samples/s", train_throughput)
        self._log_and_print("Peak Memory: {:.2f} MB", peak_mem_mb)

        # =========================================================
        # LOAD BEST MODEL
        # =========================================================
        self.load_state_dict(
            torch.load(
                os.path.join(self.save_dir, f"{self.model_name}.pt"),
                map_location=self.device,
            )
        )

        logging.info("Training complete")



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
        if "include_logs_and_traces" in self.options and self.options["include_logs_and_traces"] == 1:
            self.num_vars = self.options["num_metrics"] + self.options["num_log_features"] + self.options["num_trace_features"]
        else:
            self.num_vars = self.options["num_vars"]
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self.lower_encoder = np.quantile(us_all, (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.upper_encoder = np.quantile(us_all, 1 - (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_lower_encoder.npy'), self.lower_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_upper_encoder.npy'), self.upper_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'), self.us_mean_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'), self.us_std_encoder)

    def _get_root_cause_threshold_encoder_multi_modality(self, xs):
        self.eval()
        us_list = []
        with torch.no_grad():
            for x in xs:
                us = self._testing_step(x)[-2]
                us_list.append(us.cpu().numpy())
                # us shape: (1, num_vars, latent_dim)

        # (N, num_vars, latent_dim)
        us_all = np.concatenate(us_list, axis=0)

        # Reduce latent dim → (N, num_vars) score per var per sample
        us_all_scores = np.linalg.norm(us_all, axis=-1)

        # Stats per variable over N samples → shape (num_vars,)
        self.us_mean_encoder = np.median(us_all_scores, axis=0)
        self.us_std_encoder  = np.std(us_all_scores, axis=0)

        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'), self.us_mean_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'),  self.us_std_encoder)


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

    def _estimate_model_memory_mb(self):
        """
        Dynamically estimate model memory usage based on actual parameter + buffer dtype.
        More accurate than fixed FP32 assumption.
        """

        total_bytes = 0

        # Parameters
        for p in self.parameters():
            total_bytes += p.numel() * p.element_size()

        # Buffers (BatchNorm, running stats, etc.)
        for b in self.buffers():
            total_bytes += b.numel() * b.element_size()

        return total_bytes / (1024 ** 2)
    

    def _testing_root_cause(self, xs, labels, alpha: float = 0.5, use_attention_fusion: bool = False):
        coeff_architecture = self.options["coeff_architecture"]

        # 1. Baseline check
        if coeff_architecture in ["rcd", "baro", "nsigma", "torai", "e_diagnosis"]:
            if coeff_architecture == "rcd":
                res = StatisticalRCA.evaluate_rcd(xs, labels)
            elif coeff_architecture == "baro":
                res = StatisticalRCA.evaluate_baro(xs, labels)
            elif coeff_architecture == "nsigma":
                res = StatisticalRCA.evaluate_baro(xs, labels, scalar_type="Standard")
            elif coeff_architecture == "torai":
                res = StatisticalRCA.evaluate_torai(xs, labels)
            elif coeff_architecture == "e_diagnosis":
                res = StatisticalRCA.evaluate_e_diagnosis(xs, labels)

            if res:
                k_at_step_all = res["avg_k_at_step"]
                scores_list = res["scores"]
                labels_list = res["labels"]

                mrr_list = []
                hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []

                for z_scores, current_labels in zip(scores_list, labels_list):

                    ranking = np.argsort(-z_scores[0])
                    true_idx = np.where(current_labels[0] == 1)[0]

                    # MRR
                    rr = 0.0
                    for rank, idx in enumerate(ranking, start=1):
                        if idx in true_idx:
                            rr = 1.0 / rank
                            break

                    mrr_list.append(rr)

                    # HR@K
                    def hit(k):
                        return int(any(idx in ranking[:k] for idx in true_idx))

                    hr1_list.append(hit(1))
                    hr3_list.append(hit(3))
                    hr5_list.append(hit(5))
                    hr10_list.append(hit(10))

                mrr = np.mean(mrr_list)
                hr1, hr3, hr5, hr10 = map(
                    np.mean,
                    [hr1_list, hr3_list, hr5_list, hr10_list]
                )

                auc_k = np.mean(k_at_step_all[:10])
                std_ac = np.std(np.array(k_at_step_all))
                self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
                self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
                self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
                self._log_and_print("MRR: {:.5f}", mrr)

                self._log_and_print(
                    "HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                    hr1, hr3, hr5, hr10
                )
                valid_samples = len(scores_list)
                total_samples = len(xs)
                coverage = valid_samples / total_samples if total_samples > 0 else 0.0
                write_results(
                    self.options,
                    self.local_model_name,
                    [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                    k_at_step_all,
                    0,
                    self.options.get("results_csv"),
                    extra_metrics={
                        "mrr": mrr,
                        "hr@1": hr1,
                        "hr@3": hr3,
                        "hr@5": hr5,
                        "hr@10": hr10,
                        "auc@10": auc_k,
                        "std_ac": std_ac,
                        "coverage": coverage,
                        "avg_time": 0,
                        "throughput": 0,
                        "model_mem_mb": 0,
                        "peak_mem_mb": 0,
                    },
                )
            return res

        # 2. Model Loading & Setup
        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                    map_location=self.device)
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy')
        )#(30,)
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy')
        )#(30,)

        # =========================================================
        # MEMORY TRACKING (UNIFIED CPU + GPU)
        # =========================================================
        use_cuda = torch.cuda.is_available() and self.device != "cpu"

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process(os.getpid())

        peak_mem_bytes = {"value": 0}
        stop_event = threading.Event()

        def memory_poller():
            """CPU memory polling for true peak tracking."""
            while not stop_event.is_set():
                mem = process.memory_info().rss
                if mem > peak_mem_bytes["value"]:
                    peak_mem_bytes["value"] = mem
                time.sleep(0.01)  # 10ms resolution

        monitor_thread = None
        if not use_cuda:
            monitor_thread = threading.Thread(target=memory_poller)
            monitor_thread.start()

        model_mem_mb = self._estimate_model_memory_mb()

        us_list = []
        us_sample_list = []
        attn_list = []

        # ============================
        # NEW METRIC STORAGE (ADDED ONLY)
        # ============================
        inference_times = []
        mrr_list = []
        hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []

        # 3. Inference Loop
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x = xs[i]# x = (10,30)   x[i]=(30,) 
                label = labels[i]# label = (10,30)   label[i]=(30,) 

                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(
                    x, label, add_u=False
                )#us.shape = (1,30)

                u_numpy = us.cpu().numpy()
                us_sample_list.append(u_numpy)
                us_list.append(u_numpy)

                if use_attention_fusion:
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()
                    attn_list.append(attn_mean)

        # 4. Global POT Threshold Calculation
        us_all = np.concatenate(us_list, axis=0) #(1430,30)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)# (1430,30)

        # as statistical models can't compute pot, we use a threshold based topk
        #us_all_z_score_pot = []
        #for i in tqdm(range(self.num_vars), desc="Calculating POT thresholds"):
        #    col_data = us_all_z_score[:, i]
        #    col_data = col_data[np.isfinite(col_data)]
#
        #    if col_data.size == 0:
        #        us_all_z_score_pot.append(0.0)
        #        continue
#
        #    try:
        #        pot_val, _ = pot(col_data, self.risk, self.initial_level, self.num_candidates)
        #    except:
        #        pot_val = np.mean(col_data) + 3 * np.std(col_data)
#
        #    us_all_z_score_pot.append(pot_val)
#
        #us_all_z_score_pot = np.array(us_all_z_score_pot)#(30,)


        # =========================================================
        # 🔹 STOP MEMORY TRACKING
        # =========================================================
        if not use_cuda:
            stop_event.set()
            monitor_thread.join()

            # final correction sample
            final_mem = process.memory_info().rss
            peak_mem_mb = max(peak_mem_bytes["value"], final_mem) / (1024 ** 2)
        else:
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 5. Top-K Evaluation
        k_all = []
        k_at_step_all = []

        for i in tqdm(range(len(xs)), desc="Top-K Evaluation"):
            start_time = time.time()

            us_sample = us_sample_list[i] #(1,30)
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)#(1,30)


            ## =========================================================
            ## Robust statistical RCA score
            ## =========================================================
            #from sklearn.preprocessing import RobustScaler
#
            #window_data = xs[i]          # (window, num_vars)
#
            #split_idx = max(1, window_data.shape[0] // 2)
#
            #normal_part = window_data[:split_idx]
            #anomal_part = window_data[split_idx:]
#
            #robust_scores = []
#
            #for var in range(self.num_vars):
#
            #    a = normal_part[:, var]
            #    b = anomal_part[:, var]
#
            #    scaler = RobustScaler().fit(a.reshape(-1, 1))
#
            #    zscores = scaler.transform(
            #        b.reshape(-1, 1)
            #    )[:, 0]
#
            #    # BARO score
            #    robust_scores.append(np.max(np.abs(zscores)))
#
            #robust_scores = np.asarray(robust_scores)
            #
            #
            ## =========================================================
            ## Rank-based fusion
            ## =========================================================
            #ed_scores = z_scores[0]
            #baro_scores = np.array(robust_scores)
            ## rankings
            #ed_rank = np.argsort(np.argsort(-ed_scores)) + 1
            #baro_rank = np.argsort(np.argsort(-baro_scores)) + 1
            #
            #agreement = np.abs(ed_rank - baro_rank)
            #agreement = agreement / (agreement.max() + 1e-8)
            #
            #alpha_dyn = 1.0 - agreement
            #
            #hybrid_scores = (
            #    alpha_dyn * ed_scores +
            #    (1.0 - alpha_dyn) * robust_scores
            #)
            #
            #z_scores = hybrid_scores.reshape(1, -1)
            
            if use_attention_fusion:
                attn_per_lag = attn_list[i].mean(axis=2)
                attn_importance = attn_per_lag.mean(axis=0)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(
                    z_scores.shape[0], axis=0
                )
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance

            current_labels = np.max(labels[i], axis=0, keepdims=True)# labels = (1430, 10, 30)

            try:
                k_lst = topk(z_scores, current_labels, threshold=0.5)
                k_at_step = topk_at_step(z_scores, current_labels)

                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)

                # ============================
                # NEW METRICS (ADDED ONLY)
                # ============================

                ranking = np.argsort(-z_scores[0])
                lab = np.asarray(current_labels)
                if lab.ndim == 0:
                    lab = np.array([lab])

                true_idx = np.where(lab == 1)[0]

                # MRR
                rr = 0.0
                for rank, idx in enumerate(ranking, start=1):
                    if idx in true_idx:
                        rr = 1.0 / rank
                        break
                mrr_list.append(rr)

                # HR@K
                def hit(k):
                    return int(any(idx in ranking[:k] for idx in true_idx))

                hr1_list.append(hit(1))
                hr3_list.append(hit(3))
                hr5_list.append(hit(5))
                hr10_list.append(hit(10))
                
                inference_times.append(time.time() - start_time)

            except Exception as e:
                self._log_and_print("Error computing top-k for sample {}: {}", i, str(e))
                continue

        # 6. Result Aggregation
        valid_samples = len(k_all)
        total_samples = len(xs)
        coverage = valid_samples / total_samples if total_samples > 0 else 0.0

        self._log_and_print(
            "RCA Coverage: {}/{} ({:.2f}%)",
            valid_samples,
            total_samples,
            coverage * 100,
        )

        if valid_samples > 0:
            k_at_step_all = np.array(k_at_step_all).mean(axis=0)

            # ============================
            # NEW METRIC AGGREGATION
            # ============================
            mrr = np.mean(mrr_list)
            hr1, hr3, hr5, hr10 = map(
                np.mean, [hr1_list, hr3_list, hr5_list, hr10_list]
            )
            auc_k = np.mean(k_at_step_all[:10])
            std_ac = np.std(np.array(k_at_step_all))

            avg_time = np.mean(inference_times)
            throughput = 1.0 / avg_time if avg_time > 0 else 0.0

            self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
            self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
            self._log_and_print('Root cause analysis AC@5: {:.5f}', k_at_step_all[4])
            self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])

            # NEW LOGS
            self._log_and_print("MRR: {:.5f}", mrr)
            self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                                hr1, hr3, hr5, hr10)
            self._log_and_print("Avg time: {:.6f}s | Throughput: {:.2f} samples/s",
                                avg_time, throughput)

            write_results(
                self.options,
                self.local_model_name,
                [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                k_at_step_all,
                self.total_params,
                self.options.get("results_csv"),
                extra_metrics={
                    "mrr": mrr,
                    "hr@1": hr1,
                    "hr@3": hr3,
                    "hr@5": hr5,
                    "hr@10": hr10,
                    "auc@10": auc_k,
                    "std_ac": std_ac,
                    "coverage": coverage,
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "model_mem_mb": model_mem_mb,
                    "peak_mem_mb": peak_mem_mb,


                    # -------------------------
                    # TRAINING efficiency (NEW)
                    # -------------------------
                    "train_total_time": self.training_metrics["total_train_time"],
                    "train_avg_epoch_time": self.training_metrics["avg_epoch_time"],
                    "train_throughput": self.training_metrics["train_throughput"],
                    "train_peak_mem_mb": self.training_metrics["peak_mem_mb"],
                },
            )
        else:
            self._log_and_print(
                "Zero valid samples found. Check if labels[i][-1] contains any anomalies."
            )
   
   
    def case_study_rca_pipeline(
        self,
        x,
        label,
        alpha=0.5,
        use_attention_fusion=False
    ):
        """
        Executes the complete RCA inference pipeline for qualitative analysis.

        Pipeline:
            input window
                -> residual generation
                -> z-score normalization
                -> optional attention fusion
                -> POT filtering
                -> ranking

        Returns all intermediate representations for visualization and analysis.
        """

        # =========================================================
        # 1. Forward inference
        # =========================================================
        with torch.no_grad():

            _, _, _, _, _, _, _, us, attn_weights = self._testing_step(
                x,
                label,
                add_u=False
            )

        residual = us.detach().cpu().numpy()  # (1, P)

        # =========================================================
        # 2. Z-score normalization
        # =========================================================
        z_scores = -(
            (residual - self.us_mean_encoder)
            / (self.us_std_encoder + 1e-8)
        )

        # =========================================================
        # 3. Optional attention fusion
        # =========================================================
        attention_scores = None

        if use_attention_fusion and attn_weights is not None:

            if torch.is_tensor(attn_weights):
                attn_np = attn_weights.detach().cpu().numpy()
            else:
                attn_np = attn_weights

            # expected shape:
            # (B, T, P, P)
            attn_per_lag = attn_np.mean(axis=2)
            attention_scores = attn_per_lag.mean(axis=1)

            z_scores = (
                alpha * z_scores
                + (1 - alpha) * attention_scores
            )

        # =========================================================
        # 4. POT thresholding
        # =========================================================
        z_pot = np.zeros_like(z_scores)

        for var_idx in range(self.num_vars):

            col = z_scores[:, var_idx]
            col = col[np.isfinite(col)]

            if len(col) == 0:
                continue

            try:
                threshold, _ = pot(
                    col,
                    self.risk,
                    self.initial_level,
                    self.num_candidates
                )

            except Exception:
                threshold = np.mean(col) + 3 * np.std(col)

            z_pot[:, var_idx] = np.where(
                z_scores[:, var_idx] >= threshold,
                z_scores[:, var_idx],
                0.0
            )

        # =========================================================
        # 5. Final ranking
        # =========================================================
        ranking = np.argsort(-z_pot[0])

        # =========================================================
        # 6. Label aggregation
        # =========================================================
        lab = np.asarray(label)

        current_labels = np.max(label, axis=0, keepdims=True)
        true_idx = np.where(current_labels[0] == 1)[0]
        # =========================================================
        # 7. Ranking metrics
        # =========================================================
        def compute_mrr(rank_list, gt_idx):

            for rank, idx in enumerate(rank_list, start=1):
                if idx in gt_idx:
                    return 1.0 / rank

            return 0.0

        def compute_hr(rank_list, gt_idx, k):

            return int(
                any(idx in rank_list[:k] for idx in gt_idx)
            )

        metrics = {
            "MRR": compute_mrr(ranking, true_idx),
            "HR@1": compute_hr(ranking, true_idx, 1),
            "HR@3": compute_hr(ranking, true_idx, 3),
            "HR@5": compute_hr(ranking, true_idx, 5),
            "HR@10": compute_hr(ranking, true_idx, 10),
        }

        # =========================================================
        # 8. Return full RCA interpretability bundle
        # =========================================================
        return {

            # raw residual anomaly signal
            "residual": residual,

            # normalized anomaly scores
            "z_scores": z_scores,

            # EVT/POT-filtered anomaly scores
            "z_pot": z_pot,

            # final RCA ranking
            "ranking": ranking,

            # GT root causes
            "true_idx": true_idx,
            "labels": current_labels,

            # evaluation metrics
            "metrics": metrics,

            # optional interpretability signal
            "attention": attention_scores,

            # labels used internally
            "labels": current_labels,
        }
   
   
    def _testing_root_cause_multi_modality_old(self, xs, labels, alpha: float = 0.5, use_attention_fusion: bool = False):
        coeff_architecture = self.options["coeff_architecture"]

        # 1. Baseline check
        if coeff_architecture in ["torai"]:
            inference_times = []
            if coeff_architecture == "torai":
                res = StatisticalRCA.evaluate_torai_multi_modality(xs, labels)
                print(f"TORAI multi-modality evaluation results: {res}")
            if res:
                k_at_step_all = res["avg_k_at_step"]
                scores_list = res["scores"]
                labels_list = res["labels"]

                mrr_list = []
                hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []

                for z_scores, current_labels in zip(scores_list, labels_list):
                    start_time = time.time()
                    ranking = np.argsort(-z_scores[0])
                    true_idx = np.where(current_labels[0] == 1)[0]

                    # MRR
                    rr = 0.0
                    for rank, idx in enumerate(ranking, start=1):
                        if idx in true_idx:
                            rr = 1.0 / rank
                            break

                    mrr_list.append(rr)

                    # HR@K
                    def hit(k):
                        return int(any(idx in ranking[:k] for idx in true_idx))

                    hr1_list.append(hit(1))
                    hr3_list.append(hit(3))
                    hr5_list.append(hit(5))
                    hr10_list.append(hit(10))
                    inference_times.append(time.time() - start_time)
                mrr = np.mean(mrr_list)
                hr1, hr3, hr5, hr10 = map(
                    np.mean,
                    [hr1_list, hr3_list, hr5_list, hr10_list]
                )
                
                auc_k = np.mean(k_at_step_all[:10])
                std_ac = np.std(np.array(k_at_step_all))
                self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
                self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
                self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
                self._log_and_print("MRR: {:.5f}", mrr)

                self._log_and_print(
                    "HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                    hr1, hr3, hr5, hr10
                )
                valid_samples = len(scores_list)
                total_samples = len(xs)
                coverage = valid_samples / total_samples if total_samples > 0 else 0.0
                avg_time = np.mean(inference_times)
                throughput = 1.0 / avg_time if avg_time > 0 else 0.0
                write_results(
                    self.options,
                    self.local_model_name,
                    [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                    k_at_step_all,
                    0,
                    self.options.get("results_csv"),
                    extra_metrics={
                        "mrr": mrr,
                        "hr@1": hr1,
                        "hr@3": hr3,
                        "hr@5": hr5,
                        "hr@10": hr10,
                        "auc@10": auc_k,
                        "std_ac": std_ac,
                        "coverage": coverage,
                        "avg_time": avg_time,
                        "throughput": throughput,
                        "model_mem_mb": 0,
                        "peak_mem_mb": 0,
                    },
                )
            return res

        # 2. Model Loading & Setup
        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                    map_location=self.device)
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy')
        )
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy')
        )

        # =========================================================
        # MEMORY TRACKING (UNIFIED CPU + GPU)
        # =========================================================
        use_cuda = torch.cuda.is_available() and self.device != "cpu"

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process(os.getpid())

        peak_mem_bytes = {"value": 0}
        stop_event = threading.Event()

        def memory_poller():
            """CPU memory polling for true peak tracking."""
            while not stop_event.is_set():
                mem = process.memory_info().rss
                if mem > peak_mem_bytes["value"]:
                    peak_mem_bytes["value"] = mem
                time.sleep(0.01)  # 10ms resolution

        monitor_thread = None
        if not use_cuda:
            monitor_thread = threading.Thread(target=memory_poller)
            monitor_thread.start()

        model_mem_mb = self._estimate_model_memory_mb()

        us_list = []
        us_sample_list = []
        attn_list = []

        # ============================
        # NEW METRIC STORAGE (ADDED ONLY)
        # ============================
        inference_times = []
        mrr_list = []
        hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []

        # 3. Inference Loop
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x = xs[i]
                label = labels[i]

                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(
                    x, label, add_u=False
                )

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
        num_pods = us_all_z_score.shape[1]
        pod_scores_all = np.linalg.norm(us_all_z_score, axis=-1)
        
        for i in tqdm(range(num_pods), desc="Calculating POT thresholds"):
            col_data = pod_scores_all[:, i]
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
        # SANITY CHECK 1
        print(f"[SANITY] us_all shape: {us_all.shape}")
        print(f"[SANITY] us_all_z_score shape: {us_all_z_score.shape}")
        print(f"[SANITY] pod_scores_all shape: {pod_scores_all.shape}")
        print(f"[SANITY] us_all_z_score_pot shape: {us_all_z_score_pot.shape}")
        print(f"[SANITY] us_all_z_score_pot values: {us_all_z_score_pot}")

        # =========================================================
        # 🔹 STOP MEMORY TRACKING
        # =========================================================
        if not use_cuda:
            stop_event.set()
            monitor_thread.join()

            # final correction sample
            final_mem = process.memory_info().rss
            peak_mem_mb = max(peak_mem_bytes["value"], final_mem) / (1024 ** 2)
        else:
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 5. Top-K Evaluation
        k_all = []
        k_at_step_all = []

        for i in tqdm(range(len(xs)), desc="Top-K Evaluation"):
            start_time = time.time()

            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)

            if use_attention_fusion:
                attn_per_lag = attn_list[i].mean(axis=2)
                attn_importance = attn_per_lag.mean(axis=0)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(
                    z_scores.shape[0], axis=0
                )
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance

            #pod_scores = np.linalg.norm(z_scores, axis=-1)
            # Norm over modalities and latent dim → shape: (num_vars,) = (5,)
            pod_scores = np.linalg.norm(z_scores, axis=(0, -1))  # shape: (5,)
            #current_labels = (np.asarray(labels[i]) > 0).astype(np.int32)
            #if np.sum(current_labels) == 0:
            #    continue
            #try:
            #    k_lst = topk(pod_scores, current_labels, us_all_z_score_pot)
            #    k_at_step = topk_at_step_multi_modality_new(pod_scores, current_labels)
#
            #    k_all.append(k_lst)
            #    k_at_step_all.append(k_at_step)
            current_labels = (np.asarray(labels[i]) > 0).astype(np.int32)  # shape: (5,)
            # SANITY CHECK 2 — first anomalous sample only
            if len(k_all) == 0:
                print(f"\n[SANITY] First anomalous sample {i}:")
                print(f"  z_scores shape: {z_scores.shape}")
                print(f"  z_scores min/max per var: {z_scores.min(axis=(0,-1))} / {z_scores.max(axis=(0,-1))}")
                print(f"  pod_scores (after norm): {pod_scores}")
                print(f"  pod_scores all same? std={pod_scores.std():.6f}")
                print(f"  current_labels: {current_labels}")
                print(f"  true_idx: {np.where(current_labels == 1)[0]}")
                print(f"  ranking: {np.argsort(-pod_scores)}")
                print(f"  us_all_z_score_pot len: {len(us_all_z_score_pot)}")
            if np.sum(current_labels) == 0:
                continue

            try:
                k_lst = topk(pod_scores[np.newaxis, :], current_labels[np.newaxis, :], us_all_z_score_pot)
                k_at_step = topk_at_step_multi_modality_new(pod_scores[np.newaxis, :], current_labels[np.newaxis, :])

                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)

                ranking = np.argsort(-pod_scores)       # directly on (5,)
                true_idx = np.where(current_labels == 1)[0]
                            # ============================
                # NEW METRICS (ADDED ONLY)
                # ============================

                #ranking = np.argsort(-pod_scores[0]) <--this
                #lab = np.asarray(current_labels)
                #if lab.ndim == 0:
                #    lab = np.array([lab])
#
                #true_idx = np.where(lab == 1)[0]
                #true_idx = np.where(current_labels[0] == 1)[0]<--this

                #ranking = np.argsort(-pod_scores.max(axis=0))          # ← was pod_scores[0]
                #true_idx = np.where(current_labels.max(axis=0) == 1)[0] # ← was current_labels[0]

                # MRR
                rr = 0.0
                for rank, idx in enumerate(ranking, start=1):
                    if idx in true_idx:
                        rr = 1.0 / rank
                        break
                mrr_list.append(rr)

                # HR@K
                def hit(k):
                    return int(any(idx in ranking[:k] for idx in true_idx))

                hr1_list.append(hit(1))
                hr3_list.append(hit(3))
                hr5_list.append(hit(5))
                hr10_list.append(hit(10))
                
                inference_times.append(time.time() - start_time)

            except Exception as e:
                self._log_and_print("Error computing top-k for sample {}: {}", i, str(e))
                continue

        # 6. Result Aggregation
        valid_samples = len(k_all)
        total_samples = len(xs)
        coverage = valid_samples / total_samples if total_samples > 0 else 0.0

        self._log_and_print(
            "RCA Coverage: {}/{} ({:.2f}%)",
            valid_samples,
            total_samples,
            coverage * 100,
        )

        if valid_samples > 0:
            k_at_step_all = np.array(k_at_step_all).mean(axis=0)
            # SANITY CHECK 3
            print(f"\n[SANITY] valid_samples: {valid_samples}/{total_samples}")
            print(f"[SANITY] mrr_list[:10]: {mrr_list[:10]}")
            print(f"[SANITY] hr1_list[:10]: {hr1_list[:10]}")
            print(f"[SANITY] k_at_step_all[:5]: {k_at_step_all[:5]}")
            # ============================
            # NEW METRIC AGGREGATION
            # ============================
            mrr = np.mean(mrr_list)
            hr1, hr3, hr5, hr10 = map(
                np.mean, [hr1_list, hr3_list, hr5_list, hr10_list]
            )
            auc_k = np.mean(k_at_step_all[:10])
            std_ac = np.std(np.array(k_at_step_all))

            avg_time = np.mean(inference_times)
            throughput = 1.0 / avg_time if avg_time > 0 else 0.0

            self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
            self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
            self._log_and_print('Root cause analysis AC@5: {:.5f}', k_at_step_all[4])
            self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])

            # NEW LOGS
            self._log_and_print("MRR: {:.5f}", mrr)
            self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                                hr1, hr3, hr5, hr10)
            self._log_and_print("Avg time: {:.6f}s | Throughput: {:.2f} samples/s",
                                avg_time, throughput)

            write_results(
                self.options,
                self.local_model_name,
                [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                k_at_step_all,
                self.total_params,
                self.options.get("results_csv"),
                extra_metrics={
                    "mrr": mrr,
                    "hr@1": hr1,
                    "hr@3": hr3,
                    "hr@5": hr5,
                    "hr@10": hr10,
                    "auc@10": auc_k,
                    "std_ac": std_ac,
                    "coverage": coverage,
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "model_mem_mb": model_mem_mb,
                    "peak_mem_mb": peak_mem_mb,


                    # -------------------------
                    # TRAINING efficiency (NEW)
                    # -------------------------
                    "train_total_time": self.training_metrics["total_train_time"],
                    "train_avg_epoch_time": self.training_metrics["avg_epoch_time"],
                    "train_throughput": self.training_metrics["train_throughput"],
                    "train_peak_mem_mb": self.training_metrics["peak_mem_mb"],
                },
            )
        else:
            self._log_and_print(
                "Zero valid samples found. Check if labels[i][-1] contains any anomalies."
            )
   
    
    def _testing_root_cause_multi_modality(self, xs, labels, alpha=0.5, use_attention_fusion=False):
        coeff_architecture = self.options["coeff_architecture"]

        # 1. Baseline check
        if coeff_architecture in ["torai"]:
            inference_times = []
            res = StatisticalRCA.evaluate_torai_multi_modality(xs, labels)
            print(f"TORAI multi-modality evaluation results: {res}")
            if res:
                k_at_step_all = res["avg_k_at_step"]
                scores_list = res["scores"]
                labels_list = res["labels"]

                mrr_list = []
                hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []

                for z_scores, current_labels in zip(scores_list, labels_list):
                    start_time = time.time()
                    # z_scores shape from torai: assume (1, num_vars) same as single
                    ranking  = np.argsort(-z_scores[0])
                    true_idx = np.where(current_labels == 1)[0]  # labels[i] is (num_vars,)

                    rr = 0.0
                    for rank, idx in enumerate(ranking, start=1):
                        if idx in true_idx:
                            rr = 1.0 / rank
                            break
                    mrr_list.append(rr)

                    def hit(k):
                        return int(any(idx in ranking[:k] for idx in true_idx))

                    hr1_list.append(hit(1))
                    hr3_list.append(hit(3))
                    hr5_list.append(hit(5))
                    hr10_list.append(hit(10))
                    inference_times.append(time.time() - start_time)

                mrr = np.mean(mrr_list)
                hr1, hr3, hr5, hr10 = map(np.mean, [hr1_list, hr3_list, hr5_list, hr10_list])
                auc_k  = np.mean(k_at_step_all[:10])
                std_ac = np.std(k_at_step_all)
                avg_time   = np.mean(inference_times)
                throughput = 1.0 / avg_time if avg_time > 0 else 0.0
                valid_samples = len(scores_list)
                coverage = valid_samples / len(xs) if len(xs) > 0 else 0.0

                self._log_and_print('Root cause analysis AC@1: {:.5f}',  k_at_step_all[0])
                self._log_and_print('Root cause analysis AC@3: {:.5f}',  k_at_step_all[2])
                self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
                self._log_and_print("MRR: {:.5f}", mrr)
                self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}", hr1, hr3, hr5, hr10)

                write_results(
                    self.options, self.local_model_name,
                    [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                    k_at_step_all, 0, self.options.get("results_csv"),
                    extra_metrics={
                        "mrr": mrr, "hr@1": hr1, "hr@3": hr3, "hr@5": hr5, "hr@10": hr10,
                        "auc@10": auc_k, "std_ac": std_ac, "coverage": coverage,
                        "avg_time": avg_time, "throughput": throughput,
                        "model_mem_mb": 0, "peak_mem_mb": 0,
                    },
                )
            return res

        # 2. Model Loading & Setup
        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                    map_location=self.device)
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy')
        )  # shape: (num_vars, latent_dim) = (5,)
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy')
        )  # shape: (num_vars, latent_dim) = (5,)

        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process(os.getpid())
        peak_mem_bytes = {"value": 0}
        stop_event = threading.Event()

        def memory_poller():
            while not stop_event.is_set():
                mem = process.memory_info().rss
                if mem > peak_mem_bytes["value"]:
                    peak_mem_bytes["value"] = mem
                time.sleep(0.01)

        monitor_thread = None
        if not use_cuda:
            monitor_thread = threading.Thread(target=memory_poller)
            monitor_thread.start()

        model_mem_mb = self._estimate_model_memory_mb()

        us_list        = []
        us_sample_list = []
        attn_list      = []
        inference_times = []
        mrr_list = []
        hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []

        # 3. Inference Loop
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x     = xs[i] # x =  (2,5,275)
                label = labels[i] # labels = (11088,5)  

                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(
                    x, label, add_u=False
                )
                # us shape: (1, num_vars, latent_dim) = (1, 5, 275) <-- correct

                u_numpy = us.cpu().numpy()
                us_sample_list.append(u_numpy)
                us_list.append(u_numpy)

                if use_attention_fusion:
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()
                    attn_list.append(attn_mean)

        # 4. Global POT Threshold Calculation
        # us_all: (N, num_vars, latent_dim) = (N, 5, 275) <--(11088,5,275)
        us_all = np.concatenate(us_list, axis=0)
        us_all_scores = np.linalg.norm(us_all, axis=-1)   # (N, 5) — reduce latent dim first <--(11088,5)

        # z_score per latent dim per var: (N, num_vars, latent_dim)
        us_all_z_score = -(us_all_scores - self.us_mean_encoder) / self.us_std_encoder  # us_all_z_score = (N, num_vars) = (11088,5)
        num_pods = us_all_scores.shape[1]  # 5
        # One POT threshold per variable, matching single modality's (num_vars,)
        us_all_z_score_pot = []
        for i in tqdm(range(num_pods), desc="Calculating POT thresholds"):
            col_data = us_all_z_score[:, i] # 
            col_data = col_data[np.isfinite(col_data)] # col_data = (N,)

            if col_data.size == 0:
                us_all_z_score_pot.append(0.0)
                continue
            try:
                pot_val, _ = pot(col_data, self.risk, self.initial_level, self.num_candidates)
            except:
                pot_val = np.mean(col_data) + 3 * np.std(col_data)

            us_all_z_score_pot.append(pot_val)

        us_all_z_score_pot = np.array(us_all_z_score_pot)  # (num_vars,) = (5,) <--correct

        if not use_cuda:
            stop_event.set()
            monitor_thread.join()
            final_mem  = process.memory_info().rss
            peak_mem_mb = max(peak_mem_bytes["value"], final_mem) / (1024 ** 2)
        else:
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 5. Top-K Evaluation
        k_all        = []
        k_at_step_all = []

        for i in tqdm(range(len(xs)), desc="Top-K Evaluation"):

            start_time = time.time()

            us_sample = us_sample_list[i]  # (1, num_vars, latent_dim) <-- (1,5,275)
            # Reduce to (1, num_vars) — matching single modality's z_scores shape
            scores_per_var = np.linalg.norm(us_sample, axis=-1)            # (1, 5)

            # z_scores: (1, num_vars, latent_dim)
            z_scores = -(scores_per_var - self.us_mean_encoder) / self.us_std_encoder

            if len(k_all) == 0:
                print(f"\n[FINAL SANITY] First anomalous sample {i}:")
                print(f"  scores_per_var: {scores_per_var}")
                print(f"  scores_per_var std: {scores_per_var.std():.6f}")  # should NOT be ~0 anymore
                print(f"  z_scores: {z_scores}")
                print(f"  z_scores std: {z_scores.std():.6f}")
                print(f"  ranking: {np.argsort(-scores_per_var[0])}")

            if use_attention_fusion:
                attn_per_lag    = attn_list[i].mean(axis=2)
                attn_importance = attn_per_lag.mean(axis=0)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(
                    z_scores.shape[0], axis=0
                )
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance


            # Labels: single modality uses np.max(labels[i], axis=0, keepdims=True)
            # labels[i] is (num_vars,) here — already flat, just add dim to match (1, num_vars)
            current_labels = (np.asarray(labels[i]) > 0).astype(np.int32)  # (num_vars,)<--(5,)
            current_labels_2d = current_labels[np.newaxis, :]               # (1, num_vars)<--(1,5)

            if np.sum(current_labels) == 0:
                continue

            try:
                # Now matches single modality: topk(z_scores(1,num_vars), labels(1,num_vars), pot(num_vars,))
                k_lst    = topk(scores_per_var, current_labels_2d, us_all_z_score_pot) #(500,)
                k_at_step = topk_at_step_multi_modality_new(scores_per_var, current_labels_2d)#(10,)

                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)

                # Ranking over num_vars — matches single modality's argsort(-z_scores[0])
                ranking  = np.argsort(-scores_per_var[0])       # (num_vars,) <--(5,)
                true_idx = np.where(current_labels == 1)[0]     # flat (num_vars,)<--(1,) claude says fine
                
                rr = 0.0
                for rank, idx in enumerate(ranking, start=1):
                    if idx in true_idx:
                        rr = 1.0 / rank
                        break
                mrr_list.append(rr)

                def hit(k):
                    return int(any(idx in ranking[:k] for idx in true_idx))

                hr1_list.append(hit(1))
                hr3_list.append(hit(3))
                hr5_list.append(hit(5))
                hr10_list.append(hit(10))

                inference_times.append(time.time() - start_time)

            except Exception as e:
                self._log_and_print("Error computing top-k for sample {}: {}", i, str(e))
                continue

        # 6. Result Aggregation
        valid_samples = len(k_all)
        total_samples = len(xs)
        coverage = valid_samples / total_samples if total_samples > 0 else 0.0

        self._log_and_print(
            "RCA Coverage: {}/{} ({:.2f}%)", valid_samples, total_samples, coverage * 100
        )

        if valid_samples > 0:
            k_at_step_all = np.array(k_at_step_all).mean(axis=0)

            mrr = np.mean(mrr_list)
            hr1, hr3, hr5, hr10 = map(np.mean, [hr1_list, hr3_list, hr5_list, hr10_list])
            auc_k  = np.mean(k_at_step_all[:10])
            std_ac = np.std(k_at_step_all)
            avg_time   = np.mean(inference_times)
            throughput = 1.0 / avg_time if avg_time > 0 else 0.0

            self._log_and_print('Root cause analysis AC@1: {:.5f}',  k_at_step_all[0])
            self._log_and_print('Root cause analysis AC@3: {:.5f}',  k_at_step_all[2])
            self._log_and_print('Root cause analysis AC@5: {:.5f}',  k_at_step_all[4])
            self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
            self._log_and_print("MRR: {:.5f}", mrr)
            self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}", hr1, hr3, hr5, hr10)
            self._log_and_print("Avg time: {:.6f}s | Throughput: {:.2f} samples/s", avg_time, throughput)

            write_results(
                self.options, self.local_model_name,
                [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                k_at_step_all, self.total_params, self.options.get("results_csv"),
                extra_metrics={
                    "mrr": mrr, "hr@1": hr1, "hr@3": hr3, "hr@5": hr5, "hr@10": hr10,
                    "auc@10": auc_k, "std_ac": std_ac, "coverage": coverage,
                    "avg_time": avg_time, "throughput": throughput,
                    "model_mem_mb": model_mem_mb, "peak_mem_mb": peak_mem_mb,
                    "train_total_time":    self.training_metrics["total_train_time"],
                    "train_avg_epoch_time": self.training_metrics["avg_epoch_time"],
                    "train_throughput":    self.training_metrics["train_throughput"],
                    "train_peak_mem_mb":   self.training_metrics["peak_mem_mb"],
                },
            )
        else:
            self._log_and_print(
                "Zero valid samples found. Check if labels[i][-1] contains any anomalies."
            )
   
    def _testing_root_cause_services_metrics(self, xs, labels, alpha: float = 0.5, use_attention_fusion: bool = False):
        # 0. Feature Mapping Setup
        mapping_path = '/home/db2003/Desktop/Amr/Tests/Medicine/dataset/aiops22-pre/初赛评分数据/idx_to_feature.json'
        with open(mapping_path, 'r') as f:
            self.idx_to_feature = json.load(f)
        feature_names = [self.idx_to_feature[str(i)] for i in range(self.num_vars)]

        coeff_architecture = self.options["coeff_architecture"]
        # 1. Baseline check
        if coeff_architecture in ["rcd", "baro", "nsigma", "torai"]:
            if coeff_architecture == "rcd":
                res = StatisticalRCA.evaluate_rcd(xs, labels)
            elif coeff_architecture == "baro":
                res = StatisticalRCA.evaluate_baro(xs, labels)
            elif coeff_architecture == "nsigma":
                res = StatisticalRCA.evaluate_nsigma(xs, labels)
            elif coeff_architecture == "torai":
                res = StatisticalRCA.evaluate_torai(xs, labels)
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

                    # preserve fault keyword explicitly
                    lower = name.lower()

                    if "cpu" in lower:
                        metric = "cpu"
                    elif "mem" in lower:
                        metric = "mem"
                    elif "disk" in lower or "io" in lower:
                        metric = "disk"
                    elif "socket" in lower:
                        metric = "socket"
                    elif "lat" in lower or "delay" in lower:
                        metric = "delay"
                    elif "loss" in lower:
                        metric = "loss"
                    else:
                        metric = "unknown"

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
                          k_at_step_all, self.total_params, self.options.get("results_csv")+"_microservice",
                          metric_results={k: v / valid_samples for k, v in results["metric"].items()},
                          node_results={k: v / valid_samples for k, v in results["node"].items()},
                          service_results={k: v / valid_samples for k, v in results["service"].items()},
                          RCA_coverage=(valid_samples/len(xs))*100)
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


