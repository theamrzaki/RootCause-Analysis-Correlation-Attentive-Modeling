import os
from models.senn import SENNGC
import torch.nn as nn
import torch
from utils.utils import (compute_mmd,compute_kl_divergence_old,compute_correlated_kl, sliding_window_view_torch,
                         eval_causal_structure, eval_causal_structure_binary,
                         pot, topk, topk_at_step, write_results)
from numpy.lib.stride_tricks import sliding_window_view
import logging
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
import torch.fft as fft
import numpy as np
import torch
from collections import defaultdict
from dowhy.gcm.shapley import estimate_shapley_values
from dowhy.gcm.shapley import ShapleyConfig, ShapleyApproximationMethods
import math
from torch.utils.tensorboard import SummaryWriter

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
        self.example_normal_window = None  # Placeholder for the normal window example
        self.use_global_attention = options.get("global_attention_over_all_lag")
        print(f'Using global attention AERCA: {self.use_global_attention}')
        #self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device, use_attention = self.use_global_attention)
        #self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device, use_attention = self.use_global_attention)
        #self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device, use_attention = self.use_global_attention)
        swat_informer_config = {
            # Task and prediction setup
            "task_name": "long_term_forecast",
            "pred_len": window_size,   # forecast horizon
            "label_len": window_size,  # decoder label length, usually <= pred_len
            "seq_len": window_size,    # encoder input length (history)

            # Data dimensions
            "enc_in": num_vars,              # number of input features for encoder
            "dec_in": num_vars,              # number of input features for decoder
            "c_out": num_vars,               # output size, typically same as enc_in

            # Transformer architecture
            "d_model": 128,            # embedding size
            "n_heads": 4,              # number of attention heads
            "e_layers": 2,             # number of encoder layers (2-4 typical)
            "d_layers": 1,             # number of decoder layers (1-2 typical)
            "d_ff": 4*128,              # feed-forward network size (usually 4x d_model)
            "factor": 5,               # sparse attention factor
            "activation": "gelu",      # activation function
            "distil": True,            # use distilling in encoder to speed training

            # Embedding and dropout
            "embed": "timeF",          # type of embedding for time features
            "freq": "h",               # frequency of data ('h' for hourly)
            "dropout": 0.05,           # dropout rate to regularize

            # Classification (only used if task is classification)
            "num_class": 10,
        }


                # merge CLI args and model config
        from types import SimpleNamespace

        config = SimpleNamespace(**swat_informer_config)

        self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device, use_attention = self.use_global_attention)
        self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device, use_attention = self.use_global_attention)
        self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device, use_attention = self.use_global_attention)

                
        self.device = device
        self.num_vars = num_vars
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
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.current_epoch = 0
        self.recon_threshold = recon_threshold
        self.root_cause_threshold_encoder = root_cause_threshold_encoder
        self.root_cause_threshold_decoder = root_cause_threshold_decoder
        self.initial_z_score = initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.log_lambda_indep = nn.Parameter(torch.tensor(0.5))  # log of lambda_indep
        self.log_lambda_corr = nn.Parameter(torch.tensor(0.5))   # log of lambda_corr
        self.log_lambda_mmd = nn.Parameter(torch.tensor(0.5))     # log of lambda_mmd
        self.compress_us_layer = nn.Sequential(
            nn.Linear(num_vars, num_vars // 5),
            nn.LayerNorm(num_vars // 5),
            nn.Tanh()  # Optional
        ).to(self.device)
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        self.model_name = 'AERCA_' + data_name + '_ws_' + str(window_size) + '_stride_' + str(stride) + \
                          '_encoder_alpha_' + str(encoder_alpha) + '_decoder_alpha_' + str(decoder_alpha) + \
                          '_encoder_gamma_' + str(encoder_gamma) + '_decoder_gamma_' + str(decoder_gamma) + \
                          '_encoder_lambda_' + str(encoder_lambda) + '_decoder_lambda_' + str(decoder_lambda) + \
                          '_beta_' + str(beta) + '_lr_' + str(lr) + '_epochs_' + str(epochs) + \
                          '_hidden_layer_size_' + str(hidden_layer_size) + '_num_hidden_layers_' + \
                          str(num_hidden_layers)
        #model name (datetime) + (family of exp) +(windwosize) + (lr) + (hidden_layer_size) + (num_hidden_layers)
        family_of_exp = 'SWAT_Default_coeff_'
        #get datetime in str  for the local model name
        from datetime import datetime
        now = datetime.now()
        datetime_str = now.strftime("%d_%H%M%S_")

        local_model_name =family_of_exp + datetime_str+ f"{str(window_size)}_{str(lr)}_{str(hidden_layer_size)}_{str(num_hidden_layers)}" 

        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates
        self.texfilter =    TexFilter(
                        embed_size=26,#TODO: make it a parameter
                        use_gelu=True,             # or use_swish=True for smoother nonlinearity
                        use_skip=True,             # ✅ Preserve original signal paths
                        use_layernorm=True,        # ✅ Stabilize across frequency bins
                        hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                        use_window=False,          # ❌ Avoid muting boundary info
                        sparsity_threshold=0.0     # ✅ Retain all weak signal components
                    )
        self.texfilter.to(self.device)
        self.linear_layer = nn.Linear(26*2,10)
        self.linear_layer.to(self.device)
        self.options = options if options is not None else {}
        # Create an absolute path for saving models and thresholds
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "runs", local_model_name))


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

    def encoding(self, xs):
        windows = sliding_window_view(xs, (self.window_size + 1, self.num_vars))[:, 0, :, :]
        winds = windows[:, :-1, :]
        nexts = windows[:, -1, :]
        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        preds, coeffs,lag_outputs, attn_weights = self.encoder(winds)
        us = preds - nexts
        return us, coeffs,lag_outputs, attn_weights, nexts[self.window_size:], winds[:-self.window_size]

    def decoding(self, us, winds, add_u=True):
        u_windows = sliding_window_view_torch(us, self.window_size + 1)
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]

        preds, coeffs,_,_ = self.decoder(u_winds)
        prev_preds, prev_coeffs,_,_ = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
        return nexts_hat, coeffs, prev_coeffs
    
    def forward(self, x, add_u=True):
        ## Ensure input `x` is a PyTorch tensor (if it's a numpy array, convert it)
        #if isinstance(x, np.ndarray):
        #    x = torch.from_numpy(x).to(self.device)  # Convert numpy array to tensor
        #
        ## Step 1: FFT along the time dimension (assumed dim=1)
        #x_fft = torch.fft.rfft(x, dim=1)  # Now works because `x` is a tensor
        #x_fft = x_fft * self.texfilter(x_fft.unsqueeze(0))
##
        ## Convert complex to real for linear layer input: e.g., concat real+imag
        #feat = torch.cat([x_fft.real,x_fft.imag], dim=-1)  # shape: (..., freq_bins*2)
##
        ## Optionally match expected input shape for linear layer
        #feat = feat.squeeze(0) if feat.dim() == 3 and self.linear_layer.in_features == feat.shape[-1] else feat
        #feat = feat.to(dtype=self.linear_layer.weight.dtype)
        #x_proj = self.linear_layer(feat)  # stays on device
#
#
        ## --- Encoding (must stay in torch) ---
        #us, encoder_coeffs,lag_outputs, attn_weights, nexts, winds = self.encoding(x_proj.cpu().detach().numpy())  # us: (batch, latent_dim) or reshape accordingly
        us, encoder_coeffs,lag_outputs, attn_weights, nexts, winds = self.encoding(x)
        if(self.options["correlated_KL"] == 1):
            # --- KL divergence with full/structured covariance prior ---\
            ##us_expand = self.compress_us_layer(us)
            
            ## Step 1: FFT along the time dimension (assumed dim=1)
            #us_fft = torch.fft.rfft(us, dim=1)  # Now works because `x` is a tensor
            #us_fft = us_fft * self.texfilter(us_fft.unsqueeze(0))
    #
            ## Convert complex to real for linear layer input: e.g., concat real+imag
            #feat = torch.cat([us_fft.real,us_fft.imag], dim=-1)  # shape: (..., freq_bins*2)
    #
            ## Optionally match expected input shape for linear layer
            #feat = feat.squeeze(0) if feat.dim() == 3 and self.linear_layer.in_features == feat.shape[-1] else feat
            #feat = feat.to(dtype=self.linear_layer.weight.dtype)
            #us_fft = self.linear_layer(feat)  # stays on device


            # epoch should be passed into forward() or stored in self.current_epoch
            # warmup settings (tune for SWaT)
            def cosine_warmup(epoch, start_epoch, warmup_epochs):
                if epoch < start_epoch:
                    return 0.0
                progress = (epoch - start_epoch) / warmup_epochs
                progress = min(progress, 1.0)
                return 0.5 * (1 - math.cos(math.pi * progress))

            kl_warmup_epochs   = 50
            corr_start_epoch   = 100
            mmd_start_epoch    = 200
            warmup_for_corr    = 50
            warmup_for_mmd     = 50
            max_lambda         = 10.0


            # Independent KL
            latent_dim = us.shape[1]
            split = latent_dim // 2
            u_indep = us[:, :split]
            u_corr = us[:, split:]

            kl_indep = compute_kl_divergence_old(u_indep, self.device)

            # Correlated KL (more stable shrinkage for SWaT)
            shrinkage = self.options.get("shrinkage", 0.1)
            kl_corr = compute_correlated_kl(u_corr, shrinkage=shrinkage)

            # Fairness loss
            s = (us[:, 0] > us[:, 0].median()).long()
            us_0, us_1 = us[s == 0], us[s == 1]
            fair_loss = compute_mmd(us_0, us_1)

            # weights
            kl_weight   = cosine_warmup(self.current_epoch, 0, kl_warmup_epochs)
            corr_weight = cosine_warmup(self.current_epoch, corr_start_epoch, warmup_for_corr)
            mmd_weight  = cosine_warmup(self.current_epoch, mmd_start_epoch, warmup_for_mmd)

            lambda_indep = torch.clamp(torch.exp(self.log_lambda_indep), 0.0, max_lambda)
            lambda_corr  = torch.clamp(torch.exp(self.log_lambda_corr),  0.0, max_lambda)
            lambda_mmd   = torch.clamp(torch.exp(self.log_lambda_mmd),   0.0, max_lambda)

            kl_div = kl_weight * lambda_indep * kl_indep \
                + kl_weight * corr_weight * lambda_corr * kl_corr \
                + kl_weight * mmd_weight  * lambda_mmd  * fair_loss

        else:
            # --- KL divergence with independent prior ---
            kl_div = compute_kl_divergence_old(us, self.device)
        # --- Decoding ---
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)

        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us,lag_outputs, attn_weights

    def forward_old(self, x, add_u=True):
        us, encoder_coeffs, nexts, winds = self.encoding(x)
        #kl_div = compute_kl_divergence(us)

        # Split latent: e.g., half independent, half correlated
        latent_dim = us.shape[1]
        split = latent_dim // 2
        u_indep = us[:, :split]       # for independent prior
        u_corr = us[:, split:]        # for correlated prior
        lambda_indep=1.0
        lambda_corr=1.0
        shrinkage=0.1
        kl_indep = compute_independent_kl(u_indep)
        kl_corr = compute_correlated_kl(u_corr, shrinkage=shrinkage)
        # Weighted combination
        kl_div = lambda_indep * kl_indep + lambda_corr * kl_corr

        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us
    def forwardaa(self, x, add_u=True):
        # Ensure input `x` is a PyTorch tensor (if it's a numpy array, convert it)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)  # Convert numpy array to tensor
        
        # Step 1: FFT along the time dimension (assumed dim=1)
        x_fft = torch.fft.rfft(x, dim=1)  # Now works because `x` is a tensor
        x_fft = x_fft * self.texfilter(x_fft.unsqueeze(0))

        # Convert complex to real for linear layer input: e.g., concat real+imag
        feat = torch.cat([x_fft.real, x_fft.imag], dim=-1)  # shape: (..., freq_bins*2)

        # Optionally match expected input shape for linear layer
        feat = feat.squeeze(0) if feat.dim() == 3 and self.linear_layer.in_features == feat.shape[-1] else feat
        feat = feat.to(dtype=self.linear_layer.weight.dtype)
        x_proj = self.linear_layer(feat)  # stays on device

        # Step 2: Proceed with encoding/decoding
        us, encoder_coeffs, nexts, winds = self.encoding(x_proj.cpu().detach().numpy())  # Or x_fft_magnitude
        
        # Step 3: Compute KL divergence (ensure `us` is a tensor)
        kl_indep = compute_kl_divergence_old(us,self.device)  # uses N(0, I) prior by default, covariance-aware
                # Split latent: e.g., half independent, half correlated
        latent_dim = us.shape[1]
        split = latent_dim // 2
        u_indep = us[:, :split]       # for independent prior
        u_corr = us[:, split:]        # for correlated prior
        lambda_indep=1.0
        lambda_corr=1.0
        shrinkage=0.07
        #kl_indep = compute_independent_kl(u_indep)
        kl_corr = compute_correlated_kl(u_corr, shrinkage=shrinkage)
        # Weighted combination
        kl_div = lambda_indep * kl_indep + lambda_corr * kl_corr


        # Step 4: Decoding
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)
        
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us

    def compute_spectral_kl_divergence(self,us_fft, device):
        # Compute power spectral density (PSD)
        psd = torch.abs(us_fft) ** 2
        psd_normalized = psd / psd.sum(dim=1, keepdim=True)  # Normalize
        
        # Target distribution (e.g., uniform PSD)
        target_psd = torch.ones_like(psd_normalized) / psd_normalized.shape[1]
        
        # KL divergence between PSDs
        kl = (psd_normalized * (torch.log(psd_normalized + 1e-10) - torch.log(target_psd + 1e-10))).sum(dim=1)
        return kl.mean()

    def _training_step(self, x, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us,lag_outputs, attn_weights = self.forward(x, add_u=add_u)
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

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

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        """
        # 2. RCA loss (new!)
        # Transform to frequency domain
        #us_fft = torch.fft.fft(us, dim=0)
        #us_fft_real = torch.abs(us_fft)  # or us_fft.real
            # RCA loss using z-scores
        # RCA loss using saved z-score normalization (consistent with testing)
        with torch.no_grad():
            us_cpu = us.cpu().numpy()
            us_z_np = -(us_cpu - self.us_mean_encoder) / self.us_std_encoder

            # Approximate per-variable POT threshold by 99th percentile
            pot_thresholds = np.percentile(us_z_np, 80, axis=0)

            # Generate pseudo-labels: 1 if score > variable-specific POT threshold else 0
            pseudo_labels_np = (us_z_np > pot_thresholds).astype(float)

        # Convert numpy arrays to torch tensors
        pseudo_labels = torch.tensor(pseudo_labels_np, dtype=us.dtype, device=us.device)
        us_z = torch.tensor(us_z_np, dtype=us.dtype, device=us.device)

        # Reshape to (batch_size, channel=1, num_vars)
        us_z_batch = us_z.unsqueeze(1)  # shape: (num_samples, 1, num_vars)

        # Forward all samples at once
        rca_scores_batch = self.rca_module(us_z_batch)  # expect shape: (num_samples, num_vars)

        # If output shape is (num_samples, 1, num_vars), squeeze channel dim
        if rca_scores_batch.dim() == 3 and rca_scores_batch.size(1) == 1:
            rca_scores_batch = rca_scores_batch.squeeze(1)

        # Use BCEWithLogitsLoss
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_rca_supervised = criterion(rca_scores_batch, pseudo_labels)
        """

        ## --- EVT part ---
        ## Example: derive peaks from reconstruction errors or some residuals
        #residuals = (nexts - nexts_hat).abs()  # shape (B, ...)
        ## flatten per example and compute a provisional threshold t (e.g., top 5% quantile)
        #B = residuals.shape[0]
        #flat = residuals.view(B, -1)  # (B, D)
        ## compute per-example threshold t as quantile (e.g., 95th percentile)
        #q = 0.95
        #t = torch.quantile(flat, q, dim=1, keepdim=True)  # (B,1)
        #peaks = F.relu(flat - t)  # excesses
#
        ## summarize peaks into features for EVT head; here simple stats
        #mean_peaks = peaks.mean(dim=1)
        #std_peaks = peaks.std(dim=1)
        #max_peaks = peaks.max(dim=1)[0]
        #frac_nonzero = (peaks > 0).float().mean(dim=1)
        #evt_feats = torch.stack([mean_peaks, std_peaks, max_peaks, frac_nonzero], dim=1)  # (B,4)
#
        #gamma, sigma = self.evt_head(evt_feats)  # (B,), (B,)
        #loss_evt = self.gpd_neg_log_likelihood(peaks, gamma, sigma)
        #logging.info('EVT NLL loss: %s', loss_evt.item())
#
        ## Optionally compute a learnable threshold z per example if needed:
        #risk = 1e-2  # or make it adaptive / another output
        #r = flat.numel() / (peaks > 0).sum(dim=1).clamp(min=1.0) * risk  # rough analog; adjust per your formulation
        ## Avoid division by zero
        ## Compute z similar to POT formula:
        #z = t.squeeze(1) + (sigma / (gamma + 1e-8)) * (torch.pow(r.clamp(min=1e-6), -gamma) - 1)
        ## Could add auxiliary loss encouraging consistency between z and observed extremes, etc.


        reg_lambda = 0.01 * (self.log_lambda_indep ** 2 + self.log_lambda_corr ** 2)
        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                loss_kl +
                reg_lambda )  # evt_weight is a hyperparameter)
        logging.info('Total loss: %s', loss.item())
        losses_dict = {
            "loss_recon": loss_recon.item(),
            "loss_encoder_coeffs": loss_encoder_coeffs.item(),
            "loss_decoder_coeffs": loss_decoder_coeffs.item(),
            "loss_prev_coeffs": loss_prev_coeffs.item(),
            "loss_encoder_smooth": loss_encoder_smooth.item(),
            "loss_decoder_smooth": loss_decoder_smooth.item(),
            "loss_prev_smooth": loss_prev_smooth.item(),
            "loss_kl": loss_kl.item(),
        }
        return loss, losses_dict

    def _training(self, xs):
        if len(xs) == 1:
            xs_train = xs[:, :int(0.8 * len(xs[0]))]
            xs_val = xs[:, int(0.8 * len(xs[0])):]
        else:
            xs_train = xs[:int(0.8 * len(xs))]
            xs_val = xs[int(0.8 * len(xs)):]
        # -------------------------------------------
        ### Prepass to compute us_mean_encoder and us_std_encoder
        us_list = []
        self.eval()
        with torch.no_grad():
            for x in xs_train:
                _, _, _, _, _, _, us,_,_ = self.forward(x, add_u=True)
                us_list.append(us.cpu().numpy())

        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0) + 1e-8  # Prevent division by 0

        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'), self.us_mean_encoder)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'), self.us_std_encoder)
        self._log_and_print('=' * 50)
        # -------------------------------------------

        best_val_loss = np.inf
        count = 0
        for epoch in tqdm(range(self.epochs), desc=f'Epoch'):
            count += 1
            self.current_epoch = epoch
            epoch_loss = 0
            self.train()
            for x in xs_train:
                self.optimizer.zero_grad()
                loss, losses_dict = self._training_step(x)
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
                logging.info(f'Saving model name: {self.model_name}.pt')
                best_val_loss = epoch_val_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            if count >= 20:
                logging.info(f'Early stopping')
                print('Early stopping')
                break
            if epoch % 5 == 0:
                self.writer.flush()
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device))
        logging.info('Training complete')
        self._get_recon_threshold(xs_val)
        self._get_root_cause_threshold_encoder(xs_val)
        self._get_root_cause_threshold_decoder(xs_val)

    def _testing_step(self, x, label=None, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us,lag_outputs, attn_weights = self.forward(x, add_u=add_u)

        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size + 1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

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

        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        reg_lambda = 0.01 * (self.log_lambda_indep ** 2 + self.log_lambda_corr ** 2)
        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                loss_kl +
                reg_lambda )  # evt_weight is a hyperparameter)
        logging.info('Total loss: %s', loss.item())

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us,lag_outputs, attn_weights

    def _get_recon_threshold(self, xs):
        self.eval()
        losses_list = []
        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _,_,_ = self._testing_step(x, add_u=False)
                loss_arr = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy().ravel()
                losses_list.append(loss_arr)
        recon_losses = np.concatenate(losses_list)
        self.recon_threshold_value = np.quantile(recon_losses, self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_threshold.npy'), self.recon_threshold_value)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_mean.npy'), self.recon_mean)
        np.save(os.path.join(self.save_dir, f'{self.model_name}_recon_std.npy'), self.recon_std)

    def _get_root_cause_threshold_encoder(self, xs):
        self.eval()
        us_list = []
        with torch.no_grad():
            for x in xs:
                us = self._testing_step(x)[-3]
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
                _, nexts_hat, nexts, _, _, _, _, _,_,_ = self._testing_step(x, add_u=False)
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

    def _testing_root_cause(self, xs, labels):
        # Load model and only the encoder-related parameters required for the POT computations.
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                        map_location=self.device))
        self.eval()
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        # Collect the latent representations from each sample.
        us_list = []
        us_sample_list = []
        attention_list = []
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us,lag_outputs, attn_weights = self._testing_step(x, label, add_u=False)
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                attention_list.append(attn_weights[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())

        #idi PART SHOULD COME HERE (1)

        # Combine all latent representations for POT threshold computation.
        if self.use_global_attention not in {"global","self","both"}:
            us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
            self._log_and_print('=' * 50)
            us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
            us_all_z_score_pot = []
            for i in range(self.num_vars):
                pot_val, _ = pot(us_all_z_score[:, i], self.risk, self.initial_level, self.num_candidates)
                us_all_z_score_pot.append(pot_val)
            us_all_z_score_pot = np.array(us_all_z_score_pot)
        else:
            # Combine latent representations and their attention weights to compute weighted latent vectors
            us_weighted_list = []
            for us, attn_w in zip(us_list, attention_list):
                attn_confidence = attn_w.squeeze(-1).mean(axis=1)  # [T], average over lags
                attn_confidence = attn_confidence / (attn_confidence.max() + 1e-8)  # normalize
                weighted_us = us * attn_confidence[:, None]  # broadcast multiply [T, num_vars]
                us_weighted_list.append(weighted_us)

            # Concatenate all weighted latent vectors
            us_all_weighted = np.concatenate(us_weighted_list, axis=0).reshape(-1, self.num_vars)

            self._log_and_print('=' * 50)

            # Compute z-scores on weighted latent vectors
            us_all_weighted_z_score = (-(us_all_weighted - self.us_mean_encoder) / self.us_std_encoder)

            # Calculate POT thresholds on weighted z-scores
            us_all_z_score_pot = []
            for i in range(self.num_vars):
                pot_val, _ = pot(us_all_weighted_z_score[:, i], self.risk, self.initial_level, self.num_candidates)
                us_all_z_score_pot.append(pot_val)
            us_all_z_score_pot = np.array(us_all_z_score_pot)

        #OR idi PART SHOULD COME HERE (2)
        
        # Compute top-k statistics for each sample using the computed POT thresholds.
        k_all = []
        k_at_step_all = []
        for i in range(len(xs)):
            us_sample = us_sample_list[i]
            if self.use_global_attention not in {"global","self","both"}:
                weighted_z = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            else:
                attn_confidence = attention_list[i].squeeze(-1).mean(axis=1)  # [T]
                attn_confidence = attn_confidence / (attn_confidence.max() + 1e-8)            
                # combine with z-scores: boost high-confidence steps
                z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)  # [T, num_vars]
                weighted_z = z_scores * attn_confidence[:, None]  # broadcast to [T, num_vars]

            k_lst = topk(weighted_z, labels[i][self.window_size * 2:], us_all_z_score_pot)
            k_at_step = topk_at_step(weighted_z, labels[i][self.window_size * 2:])
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
        write_results(self.options,ac_at,k_at_step_all,'./result.csv')

    
    """
        Root cause analysis AC@1: 0.02970
    Root cause analysis AC@3: 0.98845
    Root cause analysis AC@5: 0.99670
    Root cause analysis AC@10: 1.00000
    Root cause analysis Avg@10: 0.87706
    Root cause analysis AC*@1: 0.22772
    Root cause analysis AC*@10: 1.00000
    Root cause analysis AC*@100: 1.00000
    Root cause analysis AC*@500: 1.00000
    Root cause analysis Avg*@500: 0.99686
    Done testing for root cause analysis
    """





    def _testing_root_cause_IDI(self, xs, labels, target_series_idx=None, shapley_config=None):
        # Load model and encoder statistics
        self.load_state_dict(torch.load(
            os.path.join(self.save_dir, f'{self.model_name}.pt'),
            map_location=self.device))
        self.eval()
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        # Collect exogenous variables (u) per sample
        us_list = []
        us_sample_list = []
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us,lag_outputs, attn_weights = self._testing_step(x, label, add_u=False)[-1]  # assume last output is u
                us = us.cpu()
                # store full and post-window parts
                us_list.append(us.numpy())
                us_sample_list.append(us[self.window_size:].numpy())

        # Stack for global POT threshold calculation
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self._log_and_print('=' * 50)

        # Compute z-scores (negated as per prior implementation)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)

        # POT / SPOT threshold per variable
        pot_thresholds = []
        for i in range(self.num_vars):
            pot_val, _ = pot(us_all_z_score[:, i], self.risk, self.initial_level, self.num_candidates)
            pot_thresholds.append(pot_val)
        pot_thresholds = np.array(pot_thresholds)  # shape: (num_vars,)

        # Aggregate metrics (standard AC@k and AC*@k)
        k_all = []
        k_at_step_all = []

        # Precompute parent map for anomaly condition (Granger parents)
        parent_map = self.derive_parents_from_encoder()  # returns dict: var_idx -> list of parent indices

        for i in tqdm(range(len(xs))):
            us_sample = us_sample_list[i]  # shape: (T_window_adjusted?, num_vars)
            # Compute per-variable z-scores for this sample
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)  # shape: (..., num_vars)

            # ROOT CAUSE CANDIDATE SELECTION (IDI-style)
            # Here simplified to last timestep in window; adapt if multiple timesteps
            current_z = z_scores[-1] if z_scores.ndim > 1 else z_scores  # shape: (num_vars,)
            candidates = []
            for j, z in enumerate(current_z):
                if self.is_anomalous_by_spot(z, pot_thresholds[j]):  # implement SPOT-based logic
                    candidates.append(j)

            # Enforce anomaly condition: drop if any parent is also anomalous
            filtered_candidates = []
            for j in candidates:
                parents = parent_map.get(j, [])
                if not any(p in candidates for p in parents):
                    filtered_candidates.append(j)

            # Baseline target anomaly score before any fix (with target series index = none)
            #base_score = self._compute_target_score(xs[i])#same shape as xs[i]
            
            # Determine which series are anomalous at this timestep (for target scoring)
            # current_z is the per-variable z-score (last timestep)
            anomalous_targets = [j for j, z in enumerate(current_z) if self.is_anomalous_by_spot(z, pot_thresholds[j])]

            # Fallback: if nothing is anomalous, use all variables (or you can skip this sample)
            if len(anomalous_targets) == 0:
                anomalous_targets = list(range(self.num_vars))

            # Baseline target anomaly score focused on anomalous targets
            base_score = self._compute_target_score(xs[i], target_series_idx=anomalous_targets)


            # Define set function for Shapley (fixing exogenous variables to their normal mean)
            def set_function(alpha_mask: np.ndarray):
                with torch.no_grad():
                    _, _, _, _, _, _, us_full,_,_ = self.forward(xs[i], add_u=True)
                    us_full = us_full.squeeze(0).cpu().numpy()  # could be (d,) or (T, d)
                    #us_full = us_sample[-1].copy()

                    # Normalize to a vector for the current time: if temporal, take last timestep
                    if us_full.ndim == 2:
                        # us_full is (T, d); we intervene on the most recent (last) u
                        us_current = us_full[-1].copy()  # shape: (d,)
                    else:
                        us_current = us_full.copy()  # shape: (d,)

                    # Apply fixes to the current u vector
                    for idx, flag in enumerate(alpha_mask):
                        if flag == 1:
                            var = filtered_candidates[idx]
                            us_current[var] = self.us_mean_encoder[var]

                    # Depending on how _simulate_with_fixed_u expects its input:
                    # if it expects a single vector u (current), pass that directly.
                    # if it expects a full sequence, rebuild us_full with the last row replaced.
                    if us_full.ndim == 2:
                        intervened_us_sequence = us_full.copy()
                        intervened_us_sequence[-1] = us_current
                        intervened_score = self._simulate_with_fixed_u(xs[i], intervened_us_sequence, target_series_idx)
                    else:
                        intervened_score = self._simulate_with_fixed_u(xs[i], us_current, target_series_idx)

                return np.array([(base_score - intervened_score)*10])  # improvement

            # Compute Shapley values if there are candidates
            if len(filtered_candidates) > 0:
                SHAPLEY_CONFIG_APPROX = ShapleyConfig(
                    approximation_method=ShapleyApproximationMethods.PERMUTATION,
                    num_samples=500,
                    n_jobs=10,
                )

                SHAPLEY_CONFIG_EXACT = ShapleyConfig(
                    approximation_method=ShapleyApproximationMethods.EXACT,
                    n_jobs=10,
                )
                
                shap_config = SHAPLEY_CONFIG_EXACT
                #if len(filtered_candidates) <= 5:
                #    shap_config = SHAPLEY_CONFIG_EXACT
                shap_vals = estimate_shapley_values(
                    set_func=set_function,
                    num_players=len(filtered_candidates),
                    shapley_config=shap_config,
                )
                shap_vals = np.squeeze(shap_vals).reshape(-1)
                # Map back for logging / diagnostics
                root_cause_scores = {filtered_candidates[idx]: shap_vals[idx]
                                    for idx in range(len(filtered_candidates))}
            else:
                root_cause_scores = {}

            # Standard AC@k / AC*@k bookkeeping
            #k_lst = topk(current_z, labels[i][self.window_size * 2:], pot_thresholds)
            #k_at_step = topk_at_step(current_z, labels[i][self.window_size * 2:])
            #k_all.append(k_lst)
            #k_at_step_all.append(k_at_step)

            # build per-variable root-cause score vector (length num_vars)
            # Normalize shapley to [0,1]
            rc_score_vec = np.zeros(self.num_vars, dtype=float)
            if root_cause_scores:
                max_shap = max(root_cause_scores.values())
                for var, shap_val in root_cause_scores.items():
                    rc_score_vec[var] = shap_val / (max_shap + 1e-8)  # normalized

            # Combine: base anomaly (current_z) and Shapley boost
            combined = current_z.copy()  # ensure current_z is in scope here
            # e.g., multiplicative boost
            for var in range(self.num_vars):
                combined[var] =rc_score_vec[var]  #combined[var] * (1 + rc_score_vec[var])  # or combined[var] += rc_score_vec[var]

            k_lst = topk(combined, labels[i][self.window_size * 2:], pot_thresholds)
            k_at_step = topk_at_step(combined, labels[i][self.window_size * 2:])
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)

            # (Optional) log per-sample root causes
            self._log_and_print(f"Sample {i}: filtered_candidates={filtered_candidates}, shapley={root_cause_scores}")

        # Final aggregate metrics
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




    # 1. Extract Granger parents from encoder coefficients.
    def derive_parents_from_encoder(self, threshold_quantile: float = 0.75):
        """
        Builds a parent map var_idx -> list of parent var indices based on encoder coeffs.
        Assumes that during a forward pass you can access encoder_coeffs for a representative normal window.
        encoder_coeffs shape convention assumed: (T, 1, d, d) where T is time steps,
        with entry (t, 0, i, j) being influence from series j on series i at time t.
        We aggregate across time steps by taking median of absolute values, then threshold by quantile.
        """
        self.eval()
        with torch.no_grad():
            _, _, encoder_coeffs, _, _, _, _,_,_ = self.forward(self.example_normal_window, add_u=True)
            coeffs = encoder_coeffs.detach().cpu().numpy()  # shape: (T, 1, d, d)
        
        # Remove singleton dimension
        coeffs = coeffs.squeeze(1)  # shape: (T, d, d)
        
        # Aggregate influence: median of abs values over time steps
        abs_coeffs = np.abs(coeffs)  # (T, d, d)
        agg = np.median(abs_coeffs, axis=0)  # (d, d)
        
        # Threshold to decide edges
        thresh = np.quantile(agg.flatten(), threshold_quantile)
        
        parent_map = defaultdict(list)
        d = agg.shape[0]
        for i in range(d):  # target series
            for j in range(d):  # source series
                if agg[i, j] > thresh:
                    parent_map[i].append(j)
        
        return dict(parent_map)


    # 2. Compute target anomaly score (e.g., z-score of its exogenous variable or reconstruction error)
    def _compute_target_score(self, x_window, target_series_idx=None):
        """
        Returns anomaly scores per target variable for the last timestep.
        x_window: shape (T, d)
        target_series_idx: list or None (if None, compute for all variables)
        Returns numpy array of shape (len(target_series_idx),) or (d,) if None.
        """
        self.eval()
        with torch.no_grad():

            # Forward pass
            _, _, _, _, _, _, us,_,_ = self.forward(x_window, add_u=True)
            us = us.squeeze(0).cpu().numpy()  # shape (d,) or possibly (T, d) depending on your model output

            # If us is per time step, pick last timestep (assume here us is (T, d))
            if us.ndim == 2:
                us_last = us[-1]  # shape (d,)
            else:
                us_last = us  # shape (d,)

            if target_series_idx is None:
                target_series_idx = list(range(us_last.shape[0]))

            # Compute z-scores per target var
            z_scores = (-(us_last[target_series_idx] - self.us_mean_encoder[target_series_idx]) /
                        (self.us_std_encoder[target_series_idx] + 1e-8))

            # Return absolute anomaly magnitude per variable
            return np.abs(z_scores)  # shape (len(target_series_idx),)

    # 3. Simulate with fixed exogenous variables (intervention)
    def _compute_zscore_from_u(self, u_vec, target_series_idx):
        """
        Helper: compute aggregated absolute z-score(s) for target_series_idx from a single u vector.
        Returns a scalar: mean absolute z-score over targets.
        """
        if isinstance(target_series_idx, int):
            target_series_idx = [target_series_idx]
        z = (-(u_vec[target_series_idx] - self.us_mean_encoder[target_series_idx]) /
            (self.us_std_encoder[target_series_idx] + 1e-8))  # shape: (len(targets),)
        #return float(np.mean(np.abs(z)))  # aggregate to scalar
        return float(np.max(np.abs(z)))  # more sensitive to major contributors

    def _simulate_with_fixed_u(self, x_window, modified_u_vec, target_series_idx):
       # """
       # Given x_window and modified exogenous variable(s) u at current time (could be vector or sequence),
       # compute the anomaly score after intervention for target_series_idx.
       # Returns a scalar score (same format as base_score).
       # """
       # # Normalize input u: if sequence (T, d), take last timestep; else assume (d,)
       # if isinstance(modified_u_vec, np.ndarray):
       #     us_arr = modified_u_vec
       # else:
       #     # If it's a torch tensor
       #     us_arr = modified_u_vec.cpu().numpy()
#
       # if us_arr.ndim == 2:
       #     # sequence: take last timestep
       #     u_current = us_arr[-1]
       # else:
       #     u_current = us_arr  # already a vector
#
       # # Compute z-score-based anomaly for target(s)
       # score = self._compute_zscore_from_u(u_current, target_series_idx)
       # return score
        return self._compute_zscore_from_u(modified_u_vec, target_series_idx)


    # 4. SPOT / POT based anomaly decision for a single z-score
    def is_anomalous_by_spot(self, z_value, pot_threshold):
        """
        Decide if a single-variable z-score is anomalous using a POT-derived threshold.
        You can expand this to include temporal smoothing or SPOT logic.
        """
        return np.abs(z_value) > pot_threshold

    def _testing_causal_discover(self, xs, causal_struct_value):
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                        map_location=self.device))
        self.eval()
        encoder_causal_list = []
        with torch.no_grad():
            for x in xs:
                # Only the encoder coefficients are used for causal discovery
                _, _, _, encoder_coeffs, _, _, _, _,_,_ = self._testing_step(x)
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



    def _testing_root_cause_new(self, xs, labels):
        # Load model including the Mamba RCA module
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'),
                                    map_location=self.device))
        self.eval()
        
        # Initialize metrics
        metrics = {
            'AC@1': 0,
            'AC@3': 0,
            'AC@5': 0,
            'AC@10': 0,
            'AP': 0,
            'F1': 0
        }
        total_samples = 0

        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = torch.tensor(labels[i][self.window_size * 2:]).float().to(self.device)
                
                # Get residuals and pass through Mamba RCA
                us = self._testing_step(x, None, add_u=False)[-1]  # (seq_len, num_vars)
                rca_scores = self.rca_module(us.unsqueeze(0))[0]    # (num_vars,)
                
                # Calculate metrics for this sample
                sample_metrics = self._compute_rca_metrics(rca_scores, label)
                
                # Aggregate metrics
                for k in metrics.keys():
                    metrics[k] += sample_metrics[k]
                total_samples += 1

        # Average metrics across all samples
        self._log_and_print('=' * 50)
        for metric, value in metrics.items():
            avg_value = value / total_samples
            self._log_and_print(f'Root cause analysis {metric}: {avg_value:.5f}')

    def _compute_rca_metrics(self, pred_scores, true_labels):
        """Compute all RCA metrics for a single sample"""
        metrics = {}
        
        # Convert to numpy for metric calculation
        pred_scores = pred_scores.cpu().numpy()
        true_labels = true_labels.cpu().numpy()
        
        # Top-K Accuracy
        for k in [1, 3, 5, 10]:
            topk_idx = np.argpartition(pred_scores, -k)[-k:]
            metrics[f'AC@{k}'] = float(np.any(true_labels[topk_idx] > 0.5))
        
        # Average Precision
        sorted_idx = np.argsort(pred_scores)[::-1]
        sorted_labels = true_labels[sorted_idx]
        
        precisions = []
        true_positives = 0
        for k in range(1, len(sorted_idx)+1):
            true_positives += sorted_labels[k-1]
            precisions.append(true_positives / k)
        
        if np.sum(true_labels) > 0:
            metrics['AP'] = np.sum([p * l for p, l in zip(precisions, sorted_labels)]) / np.sum(true_labels)
        else:
            metrics['AP'] = 0.0
        
        # F1 Score
        pred_binary = (pred_scores > 0.5).astype(float)
        if np.sum(pred_binary) + np.sum(true_labels) > 0:
            precision = np.sum(pred_binary * true_labels) / np.sum(pred_binary)
            recall = np.sum(pred_binary * true_labels) / np.sum(true_labels)
            metrics['F1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            metrics['F1'] = 0.0
        
        return metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

class TexFilter(nn.Module):
    def __init__(self, embed_size, scale=0.02, sparsity_threshold=0.01,
                 use_gelu=False, use_swish=False, use_skip=False,
                 use_layernorm=False, hard_threshold=False,
                 use_window=False):
        super().__init__()
        self.embed_size = embed_size
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold

        # Ablation flags
        self.use_gelu = use_gelu
        self.use_swish = use_swish
        self.use_skip = use_skip
        self.use_layernorm = use_layernorm
        self.hard_threshold = hard_threshold
        self.use_window = use_window

        self.w = nn.Parameter(self.scale * torch.randn(2, embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(embed_size))

        if self.use_layernorm:
            self.norm_real = nn.LayerNorm(embed_size, elementwise_affine=False)
            self.norm_imag = nn.LayerNorm(embed_size, elementwise_affine=False)
    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):  # x: [B, F, C] complex
        if self.use_window:
            window = torch.hann_window(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = x * window  # Apply Hanning window

        x_real = x.real
        x_imag = x.imag

        # First layer
        o1_real = torch.einsum('bfc,c->bfc', x_real, self.w[0]) - torch.einsum('bfc,c->bfc', x_imag, self.w[1]) + self.rb1
        o1_imag = torch.einsum('bfc,c->bfc', x_imag, self.w[0]) + torch.einsum('bfc,c->bfc', x_real, self.w[1]) + self.ib1

        # Activation
        if self.use_gelu:
            o1_real = F.gelu(o1_real)
            o1_imag = F.gelu(o1_imag)
        elif self.use_swish:
            o1_real = self.swish(o1_real)
            o1_imag = self.swish(o1_imag)
        else:
            o1_real = F.relu(o1_real)
            o1_imag = F.relu(o1_imag)

        if self.use_skip:
            o1_real = o1_real + x_real
            o1_imag = o1_imag + x_imag

        # Second layer
        o2_real = torch.einsum('bfc,c->bfc', o1_real, self.w1[0]) - torch.einsum('bfc,c->bfc', o1_imag, self.w1[1]) + self.rb2
        o2_imag = torch.einsum('bfc,c->bfc', o1_imag, self.w1[0]) + torch.einsum('bfc,c->bfc', o1_real, self.w1[1]) + self.ib2

        # Hard or soft threshold
        if self.hard_threshold:
            o2_real = torch.where(o2_real.abs() < self.sparsity_threshold, 0.0, o2_real)
            o2_imag = torch.where(o2_imag.abs() < self.sparsity_threshold, 0.0, o2_imag)
        else:
            y = torch.stack([o2_real, o2_imag], dim=-1)
            y = F.softshrink(y, lambd=self.sparsity_threshold)
            o2_real, o2_imag = y.unbind(dim=-1)

        if self.use_layernorm:
            o2_real = self.norm_real(o2_real)
            o2_imag = self.norm_imag(o2_imag)

        y = torch.complex(o2_real, o2_imag)

        return y
