import torch.nn as nn
import torch
import torch.nn.functional as F

class SENNGC_attention_per_lag(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        super(SENNGC___, self).__init__()

        self.num_vars = num_vars  # p
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.device = device

        self.context_radius = 1  # neighbors on each side
        self.window_size = 2 * self.context_radius + 1  # typically 3

        # gating networks per lag: input is flattened local window (window_size * p), output attention over offsets
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_size * self.num_vars, self.window_size),
                nn.Softmax(dim=1)
            )
            for _ in range(order)
        ])

        # coefficient nets per lag: input is the attended summary vector of size p
        self.coeff_nets = nn.ModuleList()
        for _ in range(order):
            layers = [nn.Linear(self.num_vars, hidden_layer_size), nn.ReLU()]
            for _ in range(self.num_hidden_layers - 1):
                layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()]
            layers += [nn.Linear(hidden_layer_size, self.num_vars ** 2), nn.Tanh()]
            self.coeff_nets.append(nn.Sequential(*layers))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def _gather_local_window(self, inputs, k):
        # returns [B, window_size, p]
        B, K, p = inputs.shape
        r = self.context_radius
        window = []
        for offset in range(-r, r + 1):
            idx = k + offset
            if 0 <= idx < K:
                window.append(inputs[:, idx, :])  # [B, p]
            else:
                window.append(torch.zeros((B, p), device=inputs.device))
        return torch.stack(window, dim=1)  # [B, window_size, p]

    def forward(self, inputs: torch.Tensor):
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(f"Expected input shape [B, {self.order}, {self.num_vars}], got {tuple(inputs.shape)}")
        B = inputs.shape[0]
        device = inputs.device

        preds = torch.zeros((B, self.num_vars), device=device)
        coeffs_list = []

        for k in range(self.order):
            local_window = self._gather_local_window(inputs, k)  # [B, window_size, p]
            flat_window = local_window.view(B, -1)  # [B, window_size * p]
            attn_weights = self.gates[k](flat_window)  # [B, window_size]
            attn_weights = attn_weights.unsqueeze(-1)  # [B, window_size, 1]
            summary = (attn_weights * local_window).sum(dim=1)  # [B, p]

            coeffs_k = self.coeff_nets[k](summary)  # [B, p^2]
            coeffs_k = coeffs_k.view(B, self.num_vars, self.num_vars)  # [B, p, p]
            coeffs_list.append(coeffs_k.unsqueeze(1))

            lag_vec = inputs[:, k, :].unsqueeze(-1)  # [B, p, 1]
            preds = preds + torch.matmul(coeffs_k, lag_vec).squeeze(-1)

        coeffs = torch.cat(coeffs_list, dim=1)  # [B, order, p, p]
        return preds, coeffs



import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.nn as nn
import torch
import torch.nn.functional as F



class LowRankCoeffNet(nn.Module):
    def __init__(self, p, hidden_dim, num_hidden_layers, rank):
        super().__init__()
        layers = [nn.Linear(p, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.left = nn.Linear(hidden_dim, p * rank)
        self.right = nn.Linear(hidden_dim, rank * p)
        self.p = p
        self.rank = rank

    def forward(self, x):  # x: [B, p]
        h = self.net(x)  # [B, hidden_dim]
        left = self.left(h).view(-1, self.p, self.rank)  # [B, p, r]
        right = self.right(h).view(-1, self.rank, self.p)  # [B, r, p]
        coeff_matrix = torch.bmm(left, right)  # [B, p, p]
        return coeff_matrix

import torch
import torch.nn as nn

class LowRankCoeffGRU(nn.Module):
    def __init__(self, p, hidden_dim, num_layers, rank):
        """
        p: number of variables (input size)
        hidden_dim: hidden size for GRU
        num_layers: number of GRU layers
        rank: low-rank factor size
        """
        super().__init__()
        self.p = p
        self.rank = rank
        
        # GRU instead of feedforward
        self.gru = nn.GRU(input_size=p, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Map GRU output to low-rank factors
        self.left = nn.Linear(hidden_dim, p * rank)
        self.right = nn.Linear(hidden_dim, rank * p)

    def forward(self, x):
        """
        x: [B, T, p] where T is sequence length
        """
        # Pass through GRU
        out, _ = self.gru(x)  # [B, T, hidden_dim]

        # We use the last timestep's hidden state
        h = out

        # Low-rank factorization
        left = self.left(h).view(-1, self.p, self.rank)    # [B, p, r]
        right = self.right(h).view(-1, self.rank, self.p)  # [B, r, p]

        coeff_matrix = torch.bmm(left, right)  # [B, p, p]
        return coeff_matrix

class GRUCoeffNet(nn.Module):
    def __init__(self, num_vars, hidden_layer_size, num_hidden_layers, latent_dim):
        super().__init__()
        self.num_vars = num_vars
        self.latent_dim = latent_dim
        
        # GRU layer(s) to process the lag sequence
        self.gru = nn.GRU(input_size=num_vars, hidden_size=latent_dim, num_layers=num_hidden_layers, batch_first=True)
        
        # Map GRU outputs (latent_dim) to flattened coeffs (num_vars^2)
        self.fc = nn.Linear(latent_dim, num_vars * num_vars)

    def forward(self, x):
        """
        x: shape (batch, order, num_vars)
        returns: coeffs tensor of shape (batch, order, num_vars, num_vars)
        """
        gru_out, _ = self.gru(x)  # (batch, order, latent_dim)
        coeffs_flat = self.fc(gru_out)  # (batch, order, num_vars^2)
        coeffs = coeffs_flat.view(x.size(0), x.size(1), self.num_vars, self.num_vars)  # reshape
        return coeffs
    
    
class SENNGC_with_GRU(nn.Module):
    def __init__(
        self,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        device: torch.device,
        attn_blend_init: float = 0.5,
        num_heads: int = 2,
        use_attention: str = "both",  # options: "none", "global", "self", "both"
    ):
        super().__init__()
        print(f"use_attention: {use_attention}")
        assert use_attention in {"global","self","both", "none"}, "Invalid use_attention flag"
        self.num_vars = num_vars
        self.order = order
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_attention = use_attention

        # Per-lag coefficient generators
        #self.coeff_nets = nn.ModuleList()
        #for _ in range(order):
        #    layers = [nn.Linear(self.num_vars, hidden_layer_size), nn.ReLU()]
        #    for _ in range(self.num_hidden_layers - 1):
        #        layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()]
        #    layers += [nn.Linear(hidden_layer_size, self.num_vars ** 2)]
        #    self.coeff_nets.append(nn.Sequential(*layers))
        rank = 8  # or smaller/larger depending on tradeoff
        self.coeff_nets = nn.ModuleList([
            LowRankCoeffGRU(self.num_vars, self.hidden_layer_size, self.num_hidden_layers, rank)
            for _ in range(order)
        ])
        # Scalar lag attention
        if use_attention in {"global", "both"}:
            self.lag_attn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, self.hidden_layer_size),
                nn.ReLU(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # Self-attention over lags
        if use_attention in {"self", "both"}:
            self.self_attn = nn.MultiheadAttention(embed_dim=self.num_vars, num_heads=num_heads, batch_first=True)
            self.self_attn_score = nn.Sequential(
                nn.Linear(self.num_vars, self.hidden_layer_size),
                nn.ReLU(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # blend parameter between learned attention and uniform
        self.attn_blend = nn.Parameter(torch.tensor(attn_blend_init))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, order, p]
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(f"Expected input shape [B, {self.order}, {self.num_vars}], got {tuple(inputs.shape)}")
        B = inputs.shape[0]
        device = inputs.device

        lag_outputs = []
        scalar_logits = []
        coeffs_list = []

        for k in range(self.order):
            lag_vec = inputs[:, k, :]  # [B, p]
            coeff_flat = self.coeff_nets[k](lag_vec)  # [B, p^2]
            coeff_k = coeff_flat.view(B, self.num_vars, self.num_vars)  # [B, p, p]
            coeffs_list.append(coeff_k.unsqueeze(1))

            yk = torch.matmul(coeff_k, lag_vec.unsqueeze(-1)).squeeze(-1)  # [B, p]
            lag_outputs.append(yk.unsqueeze(1))  # [B,1,p]

            if self.use_attention in {"global", "both"}:
                score_k = self.lag_attn(lag_vec.unsqueeze(1))  # [B,1]
                scalar_logits.append(score_k)
            else:
                scalar_logits.append(torch.zeros(B, 1, device=device))

        lag_outputs = torch.cat(lag_outputs, dim=1)     # [B, order, p]
        coeffs = torch.cat(coeffs_list, dim=1)          # [B, order, p, p]
        scalar_logits = torch.cat(scalar_logits, dim=1)  # [B, order]

        # compute scalar attention weights
        if self.use_attention in {"global", "both"}:
            scalar_attn = F.softmax(scalar_logits, dim=1)  # [B, order]
        else:
            scalar_attn = torch.full((B, self.order), 0.0, device=device)

        # compute self-attention weights
        if self.use_attention in {"self", "both"}:
            self_attn_out, _ = self.self_attn(inputs, inputs, inputs, need_weights=False)  # [B, order, p]
            self_scores = self.self_attn_score(self_attn_out).squeeze(-1)  # [B, order]
            self_attn_weights = F.softmax(self_scores, dim=1)
        else:
            self_attn_weights = torch.full((B, self.order), 0.0, device=device)

        # combine according to flag
        if self.use_attention == "none":
            final_attn = torch.full((B, self.order), 1.0 / self.order, device=device)
        else:
            learned = scalar_attn + self_attn_weights  # sum contributions (handles scalar/self/both)
            learned = learned / (learned.sum(dim=1, keepdim=True) + 1e-8)  # normalize
            uniform = torch.full((B, self.order), 1.0 / self.order, device=device)
            mix = torch.sigmoid(self.attn_blend)
            final_attn = mix * learned + (1 - mix) * uniform  # [B, order]

        attn_weights = final_attn.unsqueeze(-1)  # [B, order, 1]
        preds = (attn_weights * lag_outputs).sum(dim=1)  # [B, p]

        return preds, coeffs, lag_outputs, attn_weights



import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Informer, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

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
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
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


class aaaaa(Informer):
    def __init__(self, configs):
        super().__init__(configs)

        self.num_vars = configs.c_out  # p variables in the series
        self.coeff_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, self.num_vars * self.num_vars),
            nn.Tanh()
        )

    def forward(self, x_enc,mask=None):
        x_mark_enc = None
        x_mark_dec = None
        x_dec = torch.zeros((x_enc.shape[0], self.pred_len, self.num_vars), device=x_enc.device)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Get sequence prediction
            if self.task_name == 'long_term_forecast':
                dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            preds = dec_out[:, -self.pred_len:, :]  # [B, L, D]

            # Coefficients from encoder representation
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            coeffs = self.coeff_head(enc_out)  # [B, L, p*p]
            coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1], self.num_vars, self.num_vars)

            B = x_enc.shape[0]
            device = x_enc.device
            final_attn = torch.full((B, self.order), 1.0 / self.order, device=device)
            attn_weights = final_attn.unsqueeze(-1)  # [B, order, 1]
            preds = (attn_weights * lag_outputs).sum(dim=1)  # [B, p]

            return preds, coeffs, None,None

        if self.task_name == 'anomaly_detection':
            preds = self.anomaly_detection(x_enc)
            enc_out = self.enc_embedding(x_enc, None)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            coeffs = self.coeff_head(enc_out).view(enc_out.shape[0], enc_out.shape[1], self.num_vars, self.num_vars)
            return preds, coeffs

        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class GraphFeatureSelector(nn.Module):
    def __init__(
        self,
        num_features: int = 51,
        node_in_dim: int = 1,
        gat_hidden: int = 16,
        num_selected: int = 10,
        proj_out_dim: int = 32,
        temp: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Vectorized, simple graph-structure learning + dynamic GAT + top-k selection.
        - num_features: N (e.g., 51 sensors)
        - node_in_dim: per-node input dim (1 if just scalar sensor reading)
        - gat_hidden: hidden dim for pairwise attention scoring
        - num_selected: k nodes to select per graph
        - proj_out_dim: final output features per timestep (to feed Informer)
        - temp: Gumbel-softmax temperature for adjacency sampling
        """
        super().__init__()
        self.N = num_features
        self.node_in_dim = node_in_dim
        self.gat_hidden = gat_hidden
        self.k = num_selected
        self.proj_out_dim = proj_out_dim
        self.temp = temp
        self.eps = eps

        # Pairwise dependence logits phi (N x N). We don't force symmetry.
        self.phi = nn.Parameter(torch.zeros(self.N, self.N))

        # Dynamic attention MLP: concat(x_i, x_j) -> hidden -> score
        # Input will be 2 * node_in_dim (we keep node_in_dim small = 1)
        self.W_cat = nn.Linear(2 * node_in_dim, gat_hidden, bias=True)
        self.a = nn.Linear(gat_hidden, 1, bias=False)  # maps hidden -> scalar score

        # Linear to produce final node embedding after aggregation
        self.W_node = nn.Linear(node_in_dim, gat_hidden, bias=True)

        # Projection from flattened selected nodes to required encoder input dim
        self.proj = nn.Linear(self.k * gat_hidden, proj_out_dim)

        # small init
        nn.init.xavier_uniform_(self.W_cat.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        """
        x: [B, order, N]   (node values per timestep)
        returns: projected: [B, order, proj_out_dim], selected_indices: [B, order, k]
        """
        B, order, N = x.shape
        assert N == self.N

        device = x.device
        num_graphs = B * order

        # node_attr: [num_graphs, N, node_in_dim]
        node_attr = x.reshape(num_graphs, N).unsqueeze(-1)  # [G, N, 1] if node_in_dim==1

        # ===== learn adjacency A via Gumbel-softmax (row-wise) =====
        # phi: [N, N] -> expand to [G, N, N]
        phi = self.phi.unsqueeze(0).expand(num_graphs, -1, -1)  # [G, N, N]

        # sample gumbel noise
        gumbel = -torch.log(-torch.log(torch.rand_like(phi, device=device) + 1e-9) + 1e-9)
        sampled = (phi + gumbel) / max(self.temp, 1e-6)

        # row-wise softmax to get adjacency probabilities; rows sum to 1
        A = torch.softmax(sampled, dim=-1)  # [G, N, N] ; A[g,i,:] = distribution over j neighbors of i

        # ===== dynamic pairwise attention scores (broadcasted) =====
        # Build pairwise concatenation (xi || xj)
        # xi: [G, N, 1] -> [G, N, 1, 1]; xj: [G, 1, N, 1]
        xi = node_attr.unsqueeze(2)  # [G, N, 1, d]
        xj = node_attr.unsqueeze(1)  # [G, 1, N, d]
        # concat along last dim -> [G, N, N, 2*d]
        pair_cat = torch.cat([xi.expand(-1, -1, N, -1), xj.expand(-1, N, -1, -1)], dim=-1)  # [G,N,N,2d]
        G_, Ni, Nj, two_d = pair_cat.shape
        pair_cat_flat = pair_cat.view(G_ * Ni * Nj, two_d)  # [(G*N*N), 2d]

        # compute dynamic attention score: a^T LeakyReLU(W_cat (xi||xj))
        h = F.leaky_relu(self.W_cat(pair_cat_flat), negative_slope=0.2)  # [(G*N*N), hid]
        scores_flat = self.a(h).view(G_, Ni, Nj)  # [G, N, N]

        # incorporate adjacency A to modulate scores.
        # We add log(A + eps) to scores so adjacency acts as a prior weight
        logA = torch.log(A + self.eps)
        scores = scores_flat + logA  # [G, N, N]

        # attention alpha_ij = softmax_j(scores_{i,j})
        alpha = torch.softmax(scores, dim=-1)  # [G, N, N] ; sum_j alpha_ij = 1

        # ===== aggregate neighbor info: node_emb_i = sum_j alpha_ij * W_node(x_j) =====
        # transform node inputs
        node_proj = self.W_node(node_attr.view(num_graphs * N, -1)).view(num_graphs, N, -1)  # [G, N, hid]
        # multiply alpha [G,N,N] with node_proj [G,N,hid] along j dim:
        # (alpha @ node_proj) -> [G, N, hid]
        node_emb = torch.matmul(alpha, node_proj)  # [G, N, hid]

        # Optionally apply activation
        node_emb = F.relu(node_emb)

        # ===== select top-k nodes per graph by L2 norm of node_emb (fast, deterministic)
        # norms: [G, N]
        norms = node_emb.norm(p=2, dim=-1)  # [G, N]
        topk_vals, topk_idx = torch.topk(norms, self.k, dim=-1)  # [G, k]

        # gather embeddings
        batch_idx = torch.arange(num_graphs, device=device).unsqueeze(1).expand(-1, self.k)
        selected = node_emb[batch_idx, topk_idx]  # [G, k, hid]

        # flatten per graph and project
        selected_flat = selected.view(num_graphs, -1)  # [G, k*hid]
        selected_flat = selected_flat.view(B, order, -1)  # [B, order, k*hid]
        projected = self.proj(selected_flat)  # [B, order, proj_out_dim]

        return projected, topk_idx.view(B, order, self.k), A.view(B, order, N, N)

class SENNGC(Informer):
    def __init__(
        self,
        configs: dict,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        device: torch.device,
        attn_blend_init: float = 0.5,
        num_heads: int = 2,
        use_attention: str = "both",  # options: "none", "global", "self", "both"
    ):
        super().__init__(configs)

        print(f"use_attention: {use_attention}")
        assert use_attention in {"global","self","both", "none"}, "Invalid use_attention flag"
        self.num_vars = num_vars
        self.order = order
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_attention = use_attention

        rank = 8  # or smaller/larger depending on tradeoff
        self.coeff_nets = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(configs.d_model, self.hidden_layer_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_layer_size, self.num_vars * self.num_vars),
                    nn.Tanh()
                )
            for _ in range(order)
        ])
        self.graph_out =  nn.Linear(10,  self.num_vars)
        self.graph_selector = GraphFeatureSelector(
            num_features=num_vars,
            node_in_dim=1,
            gat_hidden=16,
            num_selected=10,
            proj_out_dim=configs.enc_in,   # IMPORTANT: set enc_in same as proj_out_dim
            temp=0.5
        )



        # Scalar lag attention
        if use_attention in {"global", "both"}:
            self.lag_attn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, self.hidden_layer_size),
                nn.ReLU(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # Self-attention over lags
        if use_attention in {"self", "both"}:
            self.self_attn = nn.MultiheadAttention(embed_dim=self.num_vars, num_heads=num_heads, batch_first=True)
            self.self_attn_score = nn.Sequential(
                nn.Linear(self.num_vars, self.hidden_layer_size),
                nn.ReLU(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # blend parameter between learned attention and uniform
        self.attn_blend = nn.Parameter(torch.tensor(attn_blend_init))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, order, p]
        if inputs.dim() != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(f"Expected input shape [B, {self.order}, {self.num_vars}], got {tuple(inputs.shape)}")
        B, order, p = inputs.shape
        device = inputs.device

        # Prepare inputs for Informer decoder forecasting
        x_mark_enc = None
        x_mark_dec = None
        x_dec = torch.zeros((B, self.pred_len, p), device=device)

        # Embed entire sequence at once and encode
        # then in forward:
        #nput_flattened, selected_idx = self.graph_selector(inputs)  # [B, order, configs.enc_in]
        #enc_out = self.enc_embedding(input_flattened, x_mark_enc)
        input_flattened, selected_idx, adj = self.graph_selector(inputs)  # [B, order, enc_in], [B,order,k]
        enc_out = self.enc_embedding(input_flattened, x_mark_enc)

        #input_flattened = self.graph_out(self.graph_selector(inputs))  # [B, order, projected_dim]
        ####input_flattened = inputs
        ####enc_out = self.enc_embedding(input_flattened, x_mark_enc)  # [B, order, d_model]
        #enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, order, d_model]

        # Full forecast with Informer decoder
        dec_out = self.long_forecast(input_flattened, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, p]
        preds = dec_out[:, -self.pred_len:, :]  # [B, pred_len, p]

        coeffs_list = []
        lag_outputs = []
        scalar_logits = []

        # For each lag/time step, compute coeff and lag output
        for k in range(order):
            enc_k = enc_out[:, k, :]  # [B, d_model]
            coeff_flat = self.coeff_nets[k](enc_k)  # [B, p*p]
            coeff_k = coeff_flat.view(B, p, p)  # [B, p, p]
            coeffs_list.append(coeff_k.unsqueeze(1))  # [B, 1, p, p]

            lag_vec = input_flattened[:, k, :]  # [B, p]
            yk = torch.matmul(coeff_k, lag_vec.unsqueeze(-1)).squeeze(-1)  # [B, p]
            lag_outputs.append(yk.unsqueeze(1))  # [B, 1, p]

            if self.use_attention in {"global", "both"}:
                score_k = self.lag_attn(lag_vec.unsqueeze(1))  # [B, 1]
                scalar_logits.append(score_k)
            else:
                scalar_logits.append(torch.zeros(B, 1, device=device))

        lag_outputs = torch.cat(lag_outputs, dim=1)  # [B, order, p]
        coeffs = torch.cat(coeffs_list, dim=1)       # [B, order, p, p]
        scalar_logits = torch.cat(scalar_logits, dim=1)  # [B, order]

        # Scalar attention weights
        if self.use_attention in {"global", "both"}:
            scalar_attn = torch.softmax(scalar_logits, dim=1)  # [B, order]
        else:
            scalar_attn = torch.zeros(B, order, device=device)

        # Self-attention weights
        if self.use_attention in {"self", "both"}:
            self_attn_out, _ = self.self_attn(input_flattened, input_flattened, input_flattened, need_weights=False)  # [B, order, p]
            self_scores = self.self_attn_score(self_attn_out).squeeze(-1)  # [B, order]
            self_attn_weights = torch.softmax(self_scores, dim=1)
        else:
            self_attn_weights = torch.zeros(B, order, device=device)

        # Combine attention
        if self.use_attention == "none":
            final_attn = torch.full((B, order), 1.0 / order, device=device)
        else:
            learned = scalar_attn + self_attn_weights  # sum of scalar + self-attention
            learned = learned / (learned.sum(dim=1, keepdim=True) + 1e-8)  # normalize
            uniform = torch.full((B, order), 1.0 / order, device=device)
            mix = torch.sigmoid(self.attn_blend)
            final_attn = mix * learned + (1 - mix) * uniform  # [B, order]

        attn_weights = final_attn.unsqueeze(-1)  # [B, order, 1]
        preds_combined = (attn_weights * lag_outputs).sum(dim=1)  # [B, p]

        return preds_combined, coeffs, lag_outputs, attn_weights


"""
import argparse
import os

def create_arg_parser():
    
    #Creates and returns the argument parser for the SWaT dataset.
#
    #Returns:
    #    argparse.ArgumentParser: The argument parser for the SWaT dataset.
    
    parser = argparse.ArgumentParser(description='SWaT')

    # Dataset arguments
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets', 'swat'), help='Data directory (default: ./datasets/swat)')
    parser.add_argument('--num_vars', type=int, default=51, help='Number of variables (default: 51)')
    parser.add_argument('--causal_quantile', type=float, default=0.70, help='Causal quantile (default: 0.70)')
    parser.add_argument('--shuffle', type=int, default=1, help='Flag for shuffling data (default: 1)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=4, help='Random seed (default: 4)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='swat', help='Dataset name (default: swat)')

    # AERCA arguments
    parser.add_argument('--window_size', type=int, default=2, help='Window size (default: 1)')
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    parser.add_argument('--encoder_alpha', type=float, default=0.5, help='Encoder alpha (default: 0.5)')
    parser.add_argument('--decoder_alpha', type=float, default=0.5, help='Decoder alpha (default: 0.5)')
    parser.add_argument('--encoder_gamma', type=float, default=0.5, help='Encoder gamma (default: 0.5)')
    parser.add_argument('--decoder_gamma', type=float, default=0.5, help='Decoder gamma (default: 0.5)')
    parser.add_argument('--encoder_lambda', type=float, default=0.5, help='Encoder lambda (default: 0.5)')
    parser.add_argument('--decoder_lambda', type=float, default=0.5, help='Decoder lambda (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.000001, help='Learning rate (default: 0.000001)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs (default: 5000)')
    parser.add_argument('--hidden_layer_size', type=int, default=1000, help='Hidden layer size (default: 1000)')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='Number of hidden layers (default: 8)')
    parser.add_argument('--recon_threshold', type=float, default=0.95, help='Reconstruction threshold (default: 0.95)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99, help='Root cause threshold for encoder (default: 0.99)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99, help='Root cause threshold for decoder (default: 0.99)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Flag for training AERCA (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=3.0, help='Initial Z-score (default: 3.0)')
    parser.add_argument('--risk', type=float, default=1e-5, help='Risk (default: 1e-5)')
    parser.add_argument('--initial_level', type=float, default=0.9, help='Initial level (default: 0.00)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')

    # Dual KL arguments
    parser.add_argument('--correlated_KL', type=int, default=1, help='Flag for correlated KL (default: 1)')
    parser.add_argument('--lambda_indep', type=float, default=1.0, help='Lambda for independence (default: 1.0)')
    parser.add_argument('--lambda_corr', type=float, default=1.0, help='Lambda for correlated (default: 1.0)')
    parser.add_argument('--shrinkage', type=float, default=0.01, help='Shrinkage factor (default: 0.07)') 
    
    # Attention arguments
    parser.add_argument('--global_attention_over_all_lag', type=str)
    parser.add_argument('--local_attention_per_lag', type=int, default=0, help='Flag for using local attention per lag (default: 0)')

    return parser

if __name__ == "__main__":
    try:
        arg_parser = create_arg_parser()
        args = arg_parser.parse_args()
        print(args)
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        
"""