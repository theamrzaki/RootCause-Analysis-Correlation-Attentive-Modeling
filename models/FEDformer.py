import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from layers.inner_models.layers.Embed import DataEmbedding
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (topk, topk_at_step,write_results)
from layers.inner_models.layers.AutoCorrelation import AutoCorrelationLayer
from layers.inner_models.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.inner_models.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.inner_models.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    """
    python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/MSL \
    --model_id MSL \
    --model FEDformer \
    --data MSL \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 128 \
    --d_ff 128 \
    --e_layers 3 \
    --enc_in 55 \
    --c_out 55 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 10
    """
    def __init__(self, configs, version='fourier', mode_select='random', modes=32,epochs=1000):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.task_name = "anomaly_detection"
        self.seq_len = configs.seq_len
        self.label_len = 48
        self.pred_len = configs.pred_len

        configs.enc_in = configs.num_class                             # Number of input features
        configs.dec_in = configs.enc_in                # Number of decoder input features
        configs.c_out = configs.num_class                             # Number of output features

        configs.embed = 'timeF'                        # Embedding type: 'fixed', 'learned', 'timeF', etc.
        configs.freq = 'h'                             # Frequency encoding granularity: 'h' (hour), 'd' (day), etc.
        configs.dropout = 0.1                          # Dropout rate

        configs.d_model = configs.options["attention_dim"]        # Hidden dimension size
        configs.d_ff = configs.options["attention_dim"]           # Feed-forward dimension size
        configs.n_heads = configs.options["num_attention_heads"]  # Number of attention heads

        configs.moving_avg = 25                        # Moving average window size for time series decomposition
        configs.activation = 'gelu'                    # Activation function: 'relu', 'gelu', etc.

        configs.e_layers = 2                           # Number of encoder layers
        configs.d_layers = 1                           # Number of decoder layers

        configs.num_class = configs.num_class          # Used only when task_name == 'classification'

        self.epochs = configs.options['epochs']
        self.device = configs.options['device']
        self.configs = configs
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.device = configs.options['device']

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select,
                                                      num_heads=configs.n_heads)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        # move all classes to device
        self.to(self.device)

        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=configs.options['lr'])

        # Create an absolute path for saving models and thresholds
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(self.save_dir, exist_ok=True)
        correlated_KL =  "correlated_&_normal" if self.configs.options['correlated_KL'] == 1 else "normal_KL"
        family_of_exp = str(self.configs.options["coeff_architecture"]) + '_(no mean)_' + correlated_KL
        from datetime import datetime
        now = datetime.now()
        datetime_str = now.strftime("%d_%H%M%S_")

        self.local_model_name =family_of_exp + datetime_str+ f"{str(self.configs.options['window_size'])}_{str(self.configs.options['lr'])}_{str(self.configs.options['seed'])}_window_{str(self.configs.options['window_size'])}" 
        self.model_name = self.local_model_name + '.pt'
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "runs", self.local_model_name))

        # count all trainable parameters
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

    def _training(self, xs):
        if self.configs.options["dataset_name"] in ["msds","lotka_volterra"]:
            self._training_msds_lotka(xs)
        elif self.configs.options["dataset_name"] in ["swat"]:
            self._training_batches_swat(xs)
        else:
            raise ValueError(f"Unknown dataset {self.configs.options['dataset']} for training")

    def _training_batches_swat(self, xs,batch_size=1000):
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
                if self.configs.options["early_stopping"]: #AERCA paper style early stopping
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

    def _training_msds_lotka(self, xs):
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
                if self.configs.options["early_stopping"]: #AERCA paper style early stopping
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

    def encoding_batch(self, xs):  # xs shape: (batch, T, num_vars)
        if self.configs.options["dataset_name"]in ["lotka_volterra","msds"]:
            #when testing & for lotka volterra training
            windows = sliding_window_view(xs, (self.configs.options["window_size"] + 1, self.configs.options["num_vars"]))[:, 0, :, :]
            winds = windows[:, :-1, :]
            nexts = windows[:, -1, :]
            return np.stack([windows])
        else:#for swat training
            batch_windows = []
            for x in xs:  # each x: (T, num_vars)
                windows = sliding_window_view(x, (self.configs.options["window_size"]+ 1, self.configs.options["num_vars"]))[:, 0, :, :]
                batch_windows.append(windows)
            return np.stack(batch_windows)

    def _training_step(self, x,add_u=True):
        # Forward pass
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        windows = self.encoding_batch(x.cpu().numpy()) # (131, 993, 8, 51)
        winds = windows[:, 0, :-1, :]   # (131, 7, 51) #take the first time step only for faster training
        nexts = windows[:, 0, 1:, :]    #(131, 8, 51)

        winds = torch.tensor(winds, dtype=torch.float32, device=self.device)
        nexts = torch.tensor(nexts, dtype=torch.float32, device=self.device)

        nexts_hat = self.forward(winds)#torch.Size([131, 7, 51])
        
        # === Full reconstruction loss ===
        loss_full_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss (full): %s', loss_full_recon.item())

        # === Total loss ===
        loss = (loss_full_recon) 
        
        # === Logging all losses ===
        losses_dict = {
            "loss_full_recon": loss_full_recon.item()
        }

        return loss, losses_dict
    """
    def _training_step(self, x, add_u=True):#go ober all time steps 
        windows = self.encoding_batch(x.cpu().numpy())  # (B, T, L, D) = (131, 993, 8, 51)

        total_loss = 0.0
        losses_dict = {"loss_full_recon": 0.0}
        count = 0

        for t in tqdm(range(windows.shape[1])):  # loop over 993 time steps
            winds = windows[:, t, :-1, :]   # (B, 7, 51)
            nexts = windows[:, t, 1:, :]    # (B, 8, 51)

            winds = torch.tensor(winds, dtype=torch.float32, device=self.device)
            nexts = torch.tensor(nexts, dtype=torch.float32, device=self.device)

            # Forward pass
            nexts_hat = self.forward(winds)  # (B, 7, 51)

            # Loss
            loss_full_recon = self.mse_loss(nexts_hat, nexts)  
            total_loss += loss_full_recon
            losses_dict["loss_full_recon"] += loss_full_recon.item()
            count += 1

        # Average across all time slices
        total_loss /= count
        losses_dict["loss_full_recon"] /= count

        return total_loss, losses_dict
    """
    # place holders for threshold computations
    def _get_recon_threshold(self, xs):
        pass
    def _get_root_cause_threshold_encoder(self, xs):
        pass
    def _get_root_cause_threshold_decoder(self, xs):
        pass

    def _testing_root_cause(self, xs, labels, alpha: float = 0.5, threshold: float = 0.5):
        """
        Root-cause analysis for iTransformer using reconstruction errors.
        Computes per-variable reconstruction errors (z_scores) over all sliding windows
        and evaluates top-k metrics per sample.
        """
        self.eval()
        k_all, k_at_step_all = [], []

        with torch.no_grad():
            for i in range(len(xs)):
                x, label = xs[i], labels[i]

                # Create sliding windows: shape (num_windows, window_size+1, num_vars)
                windows = sliding_window_view(
                    x, (self.configs.options["window_size"] + 1, self.configs.options["num_vars"])
                )[:, 0, :, :]  # (num_windows, L, N)
                num_windows, L, N = windows.shape

                # Separate input vs target for reconstruction
                winds = windows[:, :-1, :]  # (num_windows, L-1, N)
                nexts = windows[:, -1, :]   # (num_windows, N)
                winds = torch.tensor(winds, dtype=torch.float32, device=self.device)
                nexts = torch.tensor(nexts, dtype=torch.float32, device=self.device)

                # Forward pass
                nexts_hat = self.forward(winds)  # (num_windows, L-1, N) or (num_windows, N)

                # Compute per-variable reconstruction error
                if nexts_hat.dim() == 3:  # (num_windows, L-1, N)
                    z_scores_all = torch.mean(torch.abs(nexts_hat - winds), dim=1)  # (num_windows, N)
                else:
                    z_scores_all = torch.abs(nexts_hat - nexts)  # (num_windows, N)

                z_scores_all = z_scores_all.cpu().numpy()  # (num_windows, N)

                # Align sample_labels with number of windows
                sample_labels_trimmed = label[self.configs.options["window_size"] * 2:
                                            self.configs.options["window_size"] * 2 + num_windows, :]
                if sample_labels_trimmed.shape[0] != z_scores_all.shape[0]:
                    # In case of mismatch, truncate to the smaller size
                    min_len = min(sample_labels_trimmed.shape[0], z_scores_all.shape[0])
                    sample_labels_trimmed = sample_labels_trimmed[:min_len, :]
                    z_scores_all = z_scores_all[:min_len, :]

                # Compute top-k metrics
                k_all.append(topk(z_scores_all, sample_labels_trimmed, threshold=threshold))
                k_at_step_all.append(topk_at_step(z_scores_all, sample_labels_trimmed))

        # Aggregate results
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

        # Save results
        write_results(
            self.configs.options,
            self.local_model_name,
            ac_at,
            k_at_step_all,
            self.total_params,
            self.configs.options.get("results_csv", 'RQ_swat_windows.csv')
        )


    # -------------------------------
    # Ensure _log_and_print is a class method
    # -------------------------------
    def _log_and_print(self, msg, *args):
        """Helper method to log and print testing results."""
        final_msg = msg.format(*args) if args else msg
        logging.info(final_msg)
        print(final_msg)
