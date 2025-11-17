import math
import time
import warnings
from tqdm import tqdm
from collections import defaultdict
import logging
import os
from torch.utils.tensorboard import SummaryWriter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch.optim as optim
from sknetwork.ranking import PageRank
from torch.optim import lr_scheduler
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


_EPS = 1e-10


class MLPEncoder(nn.Module):  # NOTE
    """MLP encoder module."""

    def __init__(
        self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0.0, factor=True, tol=0.1
    ):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(1, n_hid, bias=True)#12
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.0 * self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0])#.double()
        H1 = F.relu((self.fc1(inputs)))
        x = self.fc2(H1)
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDecoder(nn.Module):  # NOTE
    """MLP decoder module."""
    #MLPDecoder(num_vars, window_size, hidden_layer_size, num_hidden_layers, args=options, device=device).to(device)
    def __init__(
        self, n_in_node, n_in_z, n_out, encoder=None, data_variable_size=None, batch_size=32, n_hid=64, do_prob=0.0
    ):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, origin_A, adj_A_tilt, Wa):
        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        input_z = input_z.to(torch.float64)
        
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa
        #change type of mat_z to float32 to avoid dtype mismatch
        mat_z = mat_z.to(torch.float32)
        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt


# ========================================
# VAE utility functions
# ========================================
def get_triu_indices(num_nodes):  # NOTE
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):  # NOTE
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):  # NOTE
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):  # NOTE
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.0
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):  # NOTE
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.0
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def kl_gaussian_sem(preds):  # NOTE
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5


def nll_gaussian(preds, target, variance, add_const=False):  # NOTE
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def preprocess_adj_new(adj):  # NOTE
    if CONFIG.cuda:
        adj_normalized = torch.eye(adj.shape[0]).double().cuda() - (adj.transpose(0, 1))
    else:
        adj_normalized = torch.eye(adj.shape[0]).double() - (adj.transpose(0, 1))
    return adj_normalized


def preprocess_adj_new1(adj):  # NOTE
    if CONFIG.cuda:
        adj_normalized = torch.inverse(
            torch.eye(adj.shape[0]).double().cuda() - adj.transpose(0, 1)
        )
    else:
        adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double() - adj.transpose(0, 1))
    return adj_normalized


def isnan(x):  # NOTE
    return x != x


def matrix_poly(matrix, d):  # NOTE
    if CONFIG.cuda:
        x = torch.eye(d).double().cuda() + torch.div(matrix, d)
    else:
        x = torch.eye(d).double() + torch.div(matrix, d)
    return torch.matrix_power(x, d)


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):  # NOTE
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss += 2 * tol - torch.sum(torch.abs(A[:, i])) - torch.sum(torch.abs(A[i, :])) + z * z
    return loss


# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):  # NOTE
    result = -A + z_positive * z_positive
    loss = torch.sum(result)

    return loss


class CONFIG:  # NOTE
    """Dataclass with app parameters"""

    def __init__(self):
        pass

    # You must change this to the filename you wish to use as input data!
    # data_filename = "alarm.csv"

    # Epochs
    epochs = 20

    # Batch size (note: should be divisible by sample size, otherwise throw an error)
    batch_size = 50

    # Learning rate (baseline rate = 1e-3)
    lr = 1e-3

    x_dims = 1
    z_dims = 1
    # data_variable_size = 12
    optimizer = "Adam"
    graph_threshold = 0.3
    tau_A = 0.0
    lambda_A = 0.0
    c_A = 1
    use_A_connect_loss = 0
    use_A_positiver_loss = 0
    # no_cuda = True
    seed = 42
    encoder_hidden = 64
    decoder_hidden = 64
    temp = 0.5
    k_max_iter = 1e2
    encoder = "mlp"
    decoder = "mlp"
    no_factor = False
    encoder_dropout = 0.0
    decoder_dropout = (0.0,)
    h_tol = 1e-8
    lr_decay = 200
    gamma = 1.0
    prior = False


CONFIG.cuda = torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor


def causalrca(data, inject_time=None, dataset=None, with_bg=False, with_baro=False, **kwargs):
    
    print(f"with_baro={with_baro}")

    if type(data) == dict: # multimodal
        metric = data["metric"]
        logts = data["logts"]
        # traces_err = data["tracets_err"]
        # traces_lat = data["tracets_lat"]

        # === metric ===
        metric = metric.iloc[::15, :]

        # == metric ==
        normal_metric = metric[metric["time"] < inject_time]
        anomal_metric = metric[metric["time"] >= inject_time]
        normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
        normal_metric = normal_metric[intersect]
        anomal_metric = anomal_metric[intersect]
        metric = pd.concat([normal_metric, anomal_metric], axis=0, ignore_index=True)
        data = metric
        print(f"{normal_metric.shape=}")
        print(f"{anomal_metric.shape=}")
        print(f"{metric.shape=}")
        print("with metric", data.shape)

        # == logts ==
        logts = drop_constant(logts)
        normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
        anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])
        log = pd.concat([normal_logts, anomal_logts], axis=0, ignore_index=True)
        data = pd.concat([data, log], axis=1)
        print(f"{normal_logts.shape=}")
        print(f"{anomal_logts.shape=}")
        print(f"{log.shape=}")
        print("with log", data.shape)
        data.to_csv("debug_withlog.csv", index=False)

        # print(f"{normalize=} {addup=}")

        # # == traces_err ==
        # if dataset == "mm-tt" or dataset == "mm-ob":
        #     traces_err = traces_err.fillna(method='ffill')
        #     traces_err = traces_err.fillna(0)
        #     traces_err = drop_constant(traces_err)

        #     normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        #     anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])
        #     trace = pd.concat([normal_traces_err, anomal_traces_err], axis=0, ignore_index=True)
        #     data = pd.concat([data, trace], axis=1)
        #     print(f"{normal_traces_err.shape=}")
        #     print(f"{anomal_traces_err.shape=}")
        #     print(f"{trace.shape=}")
        #     print("with traces_err", data.shape)
        # 
        #  # == traces_lat ==
        # if dataset == "mm-tt" or dataset == "mm-ob":
        #     traces_lat = traces_lat.fillna(method='ffill')
        #     traces_lat = traces_lat.fillna(0)
        #     traces_lat = drop_constant(traces_lat)
        #     normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        #     anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])
        #     trace = pd.concat([normal_traces_lat, anomal_traces_lat], axis=0, ignore_index=True)
        #     data = pd.concat([data, trace], axis=1)
        #     print(f"{normal_traces_lat.shape=}")
        #     print(f"{anomal_traces_lat.shape=}")
        #     print(f"{trace.shape=}")
        #     print("with traces_lat", data.shape)

        # dump to debug.csv
        # data.to_csv("debug.csv", index=False)
        # drop duplicated columns
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.fillna(0)

    else:
        data = preprocess(
            data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
        )

    # normalize (keep as DataFrame for BARO)
    data = data / data.max()

    # ------------------------------
    # keep a DataFrame copy for BARO/statistics later
    # ------------------------------
    train_df = data.copy()
    node_names = train_df.columns.to_list()

    # ===============================================
    # ---- Create temporal window (T=12) ------------
    # ===============================================
    window_size = 12 #12
    stride = 1
    seqs = []
    for start in range(0, len(train_df) - window_size + 1, stride):
        seqs.append(train_df.iloc[start : start + window_size].to_numpy())  # [T, num_features]
    if len(seqs) == 0:
        raise ValueError("Not enough timesteps to create one sliding window of size 12.")
    seqs = np.stack(seqs)  # shape = [num_seq, T, num_features]
    # convert to tensor shape [num_seq, num_features, T]
    train_tensor = torch.tensor(seqs).permute(0, 2, 1).double()
    num_sequences, data_variable_size, time_window = train_tensor.shape
    print(f"train_tensor.shape = {train_tensor.shape}")

    # For compatibility with old variable names
    data_sample_size = num_sequences

    node_names = data.columns.to_list()

    # graph construction, get the adj
    train_data = data

    # Generate off-diagonal interaction graph
    off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)

    # add adjacency matrix A
    num_nodes = data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))

    encoder = MLPEncoder(
        data_variable_size * time_window,
        CONFIG.x_dims,
        CONFIG.encoder_hidden,
        int(CONFIG.z_dims),
        adj_A,
        batch_size=CONFIG.batch_size,
        do_prob=CONFIG.encoder_dropout,
        factor=CONFIG.factor,
    ).double()

    decoder = MLPDecoder(
        data_variable_size * time_window,
        CONFIG.z_dims,
        CONFIG.x_dims,
        encoder,
        data_variable_size=data_variable_size,
        batch_size=CONFIG.batch_size,
        n_hid=CONFIG.decoder_hidden,
        do_prob=CONFIG.decoder_dropout,
    ).double()

    # ===================================
    # set up training parameters
    # ===================================
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.lr_decay, gamma=CONFIG.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(data_variable_size)
    tril_indices = get_tril_offdiag_indices(data_variable_size)

    if CONFIG.cuda:
        encoder.cuda()
        decoder.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    # compute constraint h(A) value
    def _h_A(A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    prox_plus = torch.nn.Threshold(0.0, 0.0)

    def stau(w, tau):
        w1 = prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def update_optimizer(optimizer, original_lr, c_A):
        """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group["lr"] = lr

        return optimizer, lr

    # ===================================
    # training:
    # ===================================
    def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        encoder.train()
        decoder.train()
        scheduler.step()

        # update optimizer
        optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

        for i in range(1):
            data = train_data[i * data_sample_size : (i + 1) * data_sample_size]
            # use full epoch over the available sequences (we keep your original single-loop)
            # take all sequences as one batch to mimic original behaviour (or break into minibatches if you want)
            # prepare data in the shape expected by encoder/decoder:
            # flatten last two dims -> [batch, vars * time_window, 1]
            #data_batch = train_tensor.reshape(train_tensor.size(0), data_variable_size * time_window, 1)
            #if CONFIG.cuda:
            #    data_batch = data_batch.cuda()
            #data_batch = Variable(data_batch).double()
            data_batch = train_tensor.double()  # shape already [num_seq, num_nodes, 12]
            if CONFIG.cuda:
                data_batch = data_batch.cuda()
            data_batch = Variable(data_batch)
            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(
                data_batch
            )  # keep existing call signature
            edges = logits

            dec_x, output, adj_A_tilt_decoder = decoder(
                data_batch, edges, data_variable_size * CONFIG.x_dims, origin_A, adj_A_tilt_encoder, Wa
            )

            if torch.sum(output != output):
                print("nan error\n")

            target = data_batch
            preds = output
            variance = 0.0

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll
            # add A loss
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

            # other loss term
            if CONFIG.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if CONFIG.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += 0.1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            # compute h(A)
            h_A = _h_A(origin_A, data_variable_size)
            loss += (
                lambda_A * h_A
                + 0.5 * c_A * h_A * h_A
                + 100.0 * torch.trace(origin_A * origin_A)
                + sparse_loss
            )  # +  0.01 * torch.sum(variance * variance)

            # print(loss)
            loss.backward()
            loss = optimizer.step()

            myA.data = stau(myA.data, CONFIG.tau_A * lr)

            if torch.sum(origin_A != origin_A):
                print("nan error\n")

            # compute metrics
            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < CONFIG.graph_threshold] = 0

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        return (
            np.mean(np.mean(kl_train) + np.mean(nll_train)),
            np.mean(nll_train),
            np.mean(mse_train),
            graph,
            origin_A,
        )

    # ===================================
    # main
    # ===================================

    # gamma = 0.5
    gamma = 0.25
    eta = 10

    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = CONFIG.c_A
    lambda_A = CONFIG.lambda_A
    h_A_new = torch.tensor(1.0)
    h_tol = CONFIG.h_tol
    k_max_iter = int(CONFIG.k_max_iter)
    h_A_old = np.inf

    E_loss = []
    N_loss = []
    M_loss = []
    start_time = time.time()
    try:
        for step_k in range(k_max_iter):
            # print(step_k)
            while c_A < 1e20:
                for epoch in range(CONFIG.epochs):
                    # print(epoch)
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(
                        epoch, best_ELBO_loss, lambda_A, c_A, optimizer
                    )
                    # print(f"{ELBO_loss=} {NLL_loss=} {MSE_loss=}")
                    E_loss.append(ELBO_loss)
                    N_loss.append(NLL_loss)
                    M_loss.append(MSE_loss)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                # print("Optimization Finished!")
                # print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, data_variable_size)
                if h_A_new.item() > gamma * h_A_old:
                    c_A *= eta
                else:
                    break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break

        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < 0.1] = 0
        graph[np.abs(graph) < 0.2] = 0
        graph[np.abs(graph) < 0.3] = 0

    except KeyboardInterrupt:
        print("Done!")

    adj = graph
    adj = np.abs(adj.T)
    # n = np.count_nonzero(adj)
    # print(f"There are {n} edges in the graph")

    # PageRank
    try:
        pagerank = PageRank()
        scores = pagerank.fit_predict(np.abs(adj.T))
    except Exception:  # empty graph
        # print("empty graph")
        return {"adj": adj, "node_names": node_names, "ranks": node_names}

    # BARO uses the DataFrame copy (train_df) — not the tensor
    if with_baro:
        split_idx = len(train_df) // 2
        normal_df = train_df.iloc[:split_idx]
        anomal_df = train_df.iloc[split_idx:]

        baro_scores = []
        for col in node_names:
            a = normal_df[col].to_numpy()
            b = anomal_df[col].to_numpy()
            scaler = RobustScaler().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            baro_scores.append(np.max(zscores))
        baro_scores = np.array(baro_scores)

        alpha = 0.6
        scores = alpha * scores + (1 - alpha) * baro_scores
    else:
        baro_scores = np.zeros_like(scores)
        scores = scores
        
    # merge scores and node names, sort by scores
    ranks = list(zip(node_names, scores))
    ranks.sort(key=lambda x: x[1], reverse=True)
    # for n, s in ranks:
    #     print(n, s)

    ranks = [x[0] for x in ranks]

    # postprocess adj, if adj[i,j] != 0 -> adj[i,j] = 1
    adj[adj != 0] = 1

    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import pandas as pd
import time, math

from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# ----------------------------- FedFormer-Style RCA -------------------------
# ---------------------------------------------------------------------------

class CausalRCA(torch.nn.Module):
    """
    A FedFormer-style model wrapper for the full causalrca pipeline.
    Clean, modular, class-based, separated into sub-methods.
    """

    def __init__(self, configs, window_size=12):
        super().__init__()
        self.configs = configs
        self.window_size = window_size

        # containers populated later
        num_nodes = self.data_variable_size
        time_window = self.window_size

        adj_A = np.zeros((num_nodes, num_nodes))

        self.encoder = MLPEncoder(
            num_nodes * time_window,
            CONFIG.x_dims,
            CONFIG.encoder_hidden,
            CONFIG.z_dims,
            adj_A,
            batch_size=CONFIG.batch_size,
            do_prob=CONFIG.encoder_dropout,
            factor=CONFIG.factor,
        ).double()

        self.decoder = MLPDecoder(
            num_nodes * time_window,
            CONFIG.z_dims,
            CONFIG.x_dims,
            self.encoder,
            data_variable_size=num_nodes,
            batch_size=CONFIG.batch_size,
            n_hid=CONFIG.decoder_hidden,
            do_prob=CONFIG.decoder_dropout,
        ).double()

        if CONFIG.cuda:
            self.encoder.cuda()
            self.decoder.cuda()


        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.configs.options['lr'])
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


    # --------------------------------------------------------------------
    # 4) -------------------------- TRAIN ONE EPOCH -----------------------
    # --------------------------------------------------------------------
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

        nexts_hat, logits = self.forward(winds)#torch.Size([131, 7, 51])
                
        # === Full reconstruction loss ===
        loss_full_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss (full): %s', loss_full_recon.item())

        # === KL loss ===
        loss_kl = torch.sum(logits * logits)
        logging.info('KL loss: %s', loss_kl.item())

        # === DAG constraint ===
        origin_A = logits
        h_A = torch.trace((origin_A * origin_A).exp()) - origin_A.size(0)
        logging.info('DAG constraint h(A): %s', h_A.item())

        # === Sparsity ===
        sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(origin_A))
        logging.info('Sparsity loss: %s', sparse_loss.item())

        # === Total loss ===
        loss = (loss_full_recon + loss_kl +
                self.configs.options['lambda_A'] * h_A + 0.5 * self.configs.options['c_A'] * h_A * h_A +
                sparse_loss)
        
        # === Logging all losses ===
        losses_dict = {
            "loss_full_recon": loss_full_recon.item()
        }

        return loss, losses_dict
    
    # --------------------------------------------------------------------
    # 6) ---------------------------- PREDICTION --------------------------
    # --------------------------------------------------------------------
    def predict_graph(self, adj, train_df, with_baro):
        pr = PageRank().fit_predict(np.abs(adj.T))

        if with_baro:
            split = len(train_df) // 2
            normal = train_df.iloc[:split]
            anomal = train_df.iloc[split:]

            baro = []
            for col in self.node_names:
                a = normal[col].values.reshape(-1, 1)
                b = anomal[col].values.reshape(-1, 1)
                scaler = RobustScaler().fit(a)
                z = scaler.transform(b)[:, 0]
                baro.append(np.max(z))
            baro = np.array(baro)

            pr = 0.6 * pr + 0.4 * baro

        ranks = list(zip(self.node_names, pr))
        ranks.sort(key=lambda x: x[1], reverse=True)

        adj_bin = (adj != 0).astype(int)

        return {
            "adj": adj_bin,
            "node_names": self.node_names,
            "ranks": [x[0] for x in ranks]
        }

    # --------------------------------------------------------------------
    # 7) ------------------------------ FORWARD ---------------------------
    # --------------------------------------------------------------------
    def forward(self, batch, **kwargs):
        enc_x, logits, origin_A, adj_A_tilt, z_gap, z_pos, myA, Wa = self.encoder(batch)
        dec_x, output, _ = self.decoder(batch, logits,
                                        self.data_variable_size * CONFIG.x_dims,
                                        origin_A, adj_A_tilt, Wa)

        # --- inference ---
        return output, logits

    # --------------------------------------------------------------------
    # 8) ------------------------ TRAINING -------------------------------
    # --------------------------------------------------------------------
    def _training(self, xs):
        if self.configs.options["dataset_name"] in ["msds","lotka_volterra"]:
            self._training_msds_lotka(xs)
        elif self.configs.options["dataset_name"] in ["swat"]:
            raise NotImplementedError("SWAT dataset training not implemented yet.")
        else:
            raise ValueError(f"Unknown dataset {self.configs.options['dataset']} for training")


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
                loss, _ = self._training_step(x)
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



if __name__ == "__main__":
    data = pd.read_csv("/home/luan/ws/cfm/tmp_data/cartservice_mem/1/data.csv")

    n = 30

    # read inject_time
    with open("/home/luan/ws/cfm/tmp_data/cartservice_mem/1/inject_time.txt", "r") as f:
        inject_time = f.read()
    inject_time = int(inject_time)
    normal_df = data[data["time"] <= inject_time].tail(n)
    anomalous_df = data[data["time"] > inject_time].head(n)
    data = pd.concat([normal_df, anomalous_df], ignore_index=True)

    output = causalrca(data, inject_time=None, dataset="ob")
    print(output)