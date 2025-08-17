import os
from models.senn import SENNGC
import torch.nn as nn
import torch
import math
from utils.utils import (compute_kl_divergence, sliding_window_view_torch,
                         eval_causal_structure, eval_causal_structure_binary,
                         pot, topk, topk_at_step,write_results)
from utils.utils_current import compute_correlated_kl, compute_mmd
from numpy.lib.stride_tricks import sliding_window_view
import logging
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
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
        self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers,args=options, device=device)
        self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device)
        self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device)
        self.latent_gru = nn.GRU(input_size=num_vars, hidden_size=num_vars, num_layers=2, batch_first=True)
        self._log_and_print('Number of parameters in encoder: {}', self._count_parameters(self.encoder))
        self._log_and_print('Number of parameters in decoder: {}', self._count_parameters(self.decoder))
        self._log_and_print('Number of parameters in decoder_prev: {}', self._count_parameters(self.decoder_prev))
        self.options = options if options is not None else {}
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        self.latent_gru.to(self.device)
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
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
        preds, coeffs = self.encoder(winds)
        us = preds - nexts                    # shape: (B, hidden_size)
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size]

    def decoding(self, us, winds, add_u=True):
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

    def forward(self, x, add_u=True):
        us, encoder_coeffs, nexts, winds = self.encoding(x)
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

    def _training_step(self, x, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)
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
                self.beta * loss_kl+
                reg_lambda)
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
            "reg_lambda": reg_lambda.item()
        }
        return loss, losses_dict

    def _training(self, xs):
        if len(xs) == 1:
            xs_train = xs[:, :int(0.8 * len(xs[0]))]
            xs_val = xs[:, int(0.8 * len(xs[0])):]
        else:
            xs_train = xs[:int(0.8 * len(xs))]
            xs_val = xs[int(0.8 * len(xs)):]
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
                logging.info(f'Saving model name: {self.model_name}.pt')
                best_val_loss = epoch_val_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, f'{self.model_name}.pt'))
            if count >= 20:
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
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)

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

        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us

    def _get_recon_threshold(self, xs):
        self.eval()
        losses_list = []
        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x, add_u=False)
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
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                us = self._testing_step(x, label, add_u=False)[-1]
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())

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
        write_results(self.options,self.local_model_name,ac_at,k_at_step_all,'./result_2.csv')

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