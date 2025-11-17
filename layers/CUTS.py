# built from https://github.com/jarrycyx/UNN/blob/main/CUTS

from layers.inner_models.layers.cuts_parts import MultiLSTM, MultiMLP

import torch
import torch.nn as nn

class CUTSEncoder(nn.Module):
    def __init__(self,
                 num_vars,
                 input_step=10,
                 pred_step=1,
                 hidden_dim=64,
                 mlp_layers=2,
                 model_type="multi_mlp",
                 device="cpu",
                 disable_graph=False):
        super().__init__()
        self.num_vars = num_vars
        self.input_step = input_step
        self.pred_step = pred_step
        self.device = device

        # --- Learnable graph ---
        if disable_graph:
            self.graph = nn.Parameter(torch.ones(num_vars, num_vars, input_step) * 1000)
        else:
            self.graph = nn.Parameter(torch.zeros(num_vars, num_vars, input_step))

        # --- Fitting model ---
        if model_type == "multi_mlp":
            self.fitting_model = MultiMLP(
                in_dim=input_step * num_vars * 1,  # assume single feature per node
                hid_dim=hidden_dim,
                out_dim=num_vars * pred_step,
                mlp_layers=mlp_layers,
                mlp_num=num_vars
            ).to(device)
        elif model_type == "multi_lstm":
            self.fitting_model = MultiLSTM(
                in_dim=num_vars * 1,
                hid_dim=hidden_dim,
                out_dim=num_vars * pred_step,
                mlp_layers=mlp_layers,
                mlp_num=num_vars
            ).to(device)
        else:
            raise NotImplementedError(f"Unknown model_type {model_type}")

    def forward(self, x: torch.Tensor, batch_chunk_size: int = 1000):
        """
        x: (B, O, P) -> B=batch, O=time, P=num_vars
        Returns:
            preds: (B, P)
            coeffs_time_like: (B, O, P, P)
            coeffs_freq_seq: dummy zeros (B, 1, P, P)
        """
        B, O, P = x.shape
        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = x[start:end]  # (B_chunk, O, P)
            Bc = x_chunk.size(0)

            # Expand to CUTS input: (B, n_nodes, m, t, d) where m=d=1
            x_exp = x_chunk.unsqueeze(2).unsqueeze(-1)  # (B, P, 1, O, 1)

            # --- Sample graph ---
            sampled_graph = torch.sigmoid(self.graph)[None].expand(Bc, -1, -1, -1)  # (B, P, P, O)

            # --- Forward through fitting model ---
            y_pred = self.fitting_model(x_exp, sampled_graph)  # (B, P, O, d)
            y_pred = y_pred.squeeze(-1)  # (B, P, O)

            # --- Aggregate prediction across time ---
            preds = y_pred.sum(dim=2)  # (B, P)

            # --- Coeffs time ---
            coeffs_time_seq = sampled_graph.permute(0, 3, 1, 2)  # (B, O, P, P)

            # --- Dummy freq coeffs ---
            coeffs_freq_seq = torch.zeros((Bc, 1, P, P), device=x.device)

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_time_seq)
            coeffs_freq_out.append(coeffs_freq_seq)

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)

        return preds, coeffs_time_like, coeffs_freq_seq
