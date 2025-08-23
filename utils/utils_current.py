import torch
from math import log
from scipy.optimize import minimize
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score
import os

def compute_kl_divergence_old(us, device: torch.device):
    """
    Compute the KL divergence between the empirical distribution of the input samples
    and an isotropic standard Gaussian distribution using PyTorch.

    Parameters:
    samples (Tensor): A 2D tensor with rows as samples and columns as features.

    Returns:
    Tensor: The KL divergence between the empirical distribution of the samples
            and the standard Gaussian distribution.
    """

    # Calculate the empirical mean and covariance matrix of the samples
    mean_p = torch.mean(us, dim=0)
    cov_p = torch.cov(us.t())

    # Dimensionality of the distribution
    d = mean_p.shape[0]

    eigenvalues = torch.linalg.eigvalsh(cov_p)
    condition_number = eigenvalues.max() / eigenvalues.clamp(min=1e-9).min()
    regularization_term = condition_number * 1e-6
    cov_p += torch.eye(d, device=device) * regularization_term
    # Ensure the covariance matrix is full rank
    # cov_p += 1e-9 * torch.eye(d).to(device)

    # Compute the trace term
    trace_term = torch.trace(cov_p)

    # Compute the product of means term (since mean_q is zero, this is just mean_p squared)
    means_term = torch.dot(mean_p, mean_p)

    # # Compute the determinant term
    # log_det_cov_p = torch.logdet(cov_p)
    try:
        L = torch.linalg.cholesky(cov_p)
        log_det_cov_p = 2 * torch.log(torch.diagonal(L)).sum()
    except RuntimeError:
        # Handle the case where Cholesky decomposition fails
        log_det_cov_p = torch.logdet(cov_p)

    # Compute the KL divergence using the formula
    kl_div = means_term + trace_term - d + log_det_cov_p
    if torch.isnan(kl_div).any():
        print('nan')
        print(f'mean_p: {mean_p}')
        print(f'cov_p: {cov_p}')
        print(f'trace_term: {trace_term}')
        print(f'means_term: {means_term}')
        print(f'log_det_cov_p: {log_det_cov_p}')
        print(f'kl_div: {kl_div}')
        raise ValueError('KL divergence is NaN')


    return kl_div




def empirical_covariance(x, shrinkage=0.1, eps=1e-6):
    # x: (batch, dim)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    batch, dim = x.shape
    mean = x.mean(dim=0, keepdim=True)  # (1, dim)
    x_centered = x - mean
    emp_cov = (x_centered.T @ x_centered) / (batch - 1)  # unbiased
    diag = torch.diag(torch.diag(emp_cov))
    cov_shrunk = (1 - shrinkage) * emp_cov + shrinkage * diag
    cov_shrunk = cov_shrunk + eps * torch.eye(dim, device=x.device)
    return mean.squeeze(0), cov_shrunk  # mean: (dim,), cov: (dim, dim)

def kl_multivariate_normal(mean_q, cov_q, mean_p, cov_p):
    # ensure all are tensors
    for name, tensor in [('mean_q', mean_q), ('cov_q', cov_q), ('mean_p', mean_p), ('cov_p', cov_p)]:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)}")

    d = mean_q.shape[0]
    # Cholesky for stability; assume covariances are PSD and well-regularized
    Lp = torch.linalg.cholesky(cov_p)  # lower triangular
    Lq = torch.linalg.cholesky(cov_q)

    log_det_p = 2 * torch.sum(torch.log(torch.diagonal(Lp)))
    log_det_q = 2 * torch.sum(torch.log(torch.diagonal(Lq)))

    inv_Lp = torch.linalg.inv(Lp)
    middle = inv_Lp @ cov_q @ inv_Lp.T
    trace_term = torch.trace(middle)

    delta = (mean_p - mean_q).unsqueeze(1)  # (d,1)
    # use solve instead of cholesky_solve for modern API
    # Solve Sigma_p^{-1} delta: first solve Lp y = delta, then Lp^T x = y
    y = torch.triangular_solve(delta, Lp, upper=False).solution
    tmp = torch.triangular_solve(y, Lp.T, upper=True).solution
    mahalanobis = (delta.squeeze(1) * tmp.squeeze(1)).sum()

    kl = 0.5 * (trace_term + mahalanobis - d + (log_det_p - log_det_q))
    return kl

def compute_kl_divergence(us, prior_mean=None, prior_cov=None, shrinkage=0.1):
   
    #us: tensor of shape (batch, latent_dim)
    #prior_mean: tensor of shape (latent_dim,) or None
    #prior_cov: tensor of shape (latent_dim, latent_dim) or None
    
    if not torch.is_tensor(us):
        us = torch.as_tensor(us, dtype=torch.float32)
    device = us.device
    mean_q, cov_q = empirical_covariance(us, shrinkage=shrinkage)
    latent_dim = mean_q.shape[0]

    if prior_mean is None:
        prior_mean = torch.zeros_like(mean_q, device=device)
    else:
        if not torch.is_tensor(prior_mean):
            prior_mean = torch.as_tensor(prior_mean, dtype=mean_q.dtype, device=device)
        prior_mean = prior_mean.to(device)

    if prior_cov is None:
        prior_cov = torch.eye(latent_dim, device=device)
    else:
        if not torch.is_tensor(prior_cov):
            prior_cov = torch.as_tensor(prior_cov, dtype=cov_q.dtype, device=device)
        prior_cov = prior_cov.to(device)

    kl = kl_multivariate_normal(mean_q, cov_q, prior_mean, prior_cov)
    return kl


def empirical_covariance(x, shrinkage=0.1, eps=1e-6):
    batch, dim = x.shape
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    emp_cov = (x_centered.T @ x_centered) / (batch - 1)
    diag = torch.diag(torch.diag(emp_cov))
    cov_shrunk = (1 - shrinkage) * emp_cov + shrinkage * diag
    cov_shrunk = cov_shrunk + eps * torch.eye(dim, device=x.device)
    return mean.squeeze(0), cov_shrunk

def kl_gaussian(mean_q, cov_q, mean_p, cov_p):
    d = mean_q.shape[0]
    Lp = torch.linalg.cholesky(cov_p)
    Lq = torch.linalg.cholesky(cov_q)

    log_det_p = 2 * torch.sum(torch.log(torch.diagonal(Lp)))
    log_det_q = 2 * torch.sum(torch.log(torch.diagonal(Lq)))

    inv_Lp = torch.linalg.inv(Lp)
    middle = inv_Lp @ cov_q @ inv_Lp.T
    trace_term = torch.trace(middle)

    delta = (mean_p - mean_q).unsqueeze(1)
    y = torch.triangular_solve(delta, Lp, upper=False).solution
    tmp = torch.triangular_solve(y, Lp.T, upper=True).solution
    mahalanobis = (delta.squeeze(1) * tmp.squeeze(1)).sum()

    kl = 0.5 * (trace_term + mahalanobis - d + (log_det_p - log_det_q))
    return kl

def compute_independent_kl(us):
    # prior is standard normal; posterior approximated per-dimension as Gaussian with diag covariance
    # us: (batch, latent_dim)
    mean = us.mean(dim=0)
    var = us.var(dim=0, unbiased=False) + 1e-6  # variance per-dim
    # KL between N(mean, var) and N(0,1): sum over dims
    kl = 0.5 * torch.sum(var + mean**2 - 1 - torch.log(var))
    return kl

def compute_correlated_kl(us, shrinkage=0.1):
    mean_q, cov_q = empirical_covariance(us, shrinkage=shrinkage)
    latent_dim = mean_q.shape[0]
    prior_mean = torch.zeros(latent_dim, device=us.device)
    prior_cov = torch.eye(latent_dim, device=us.device)
    kl = kl_gaussian(mean_q, cov_q, prior_mean, prior_cov)
    return kl


#-----> deepseek utils <-----
import torch
import torch.distributions as dist

def kl_multivariate_normal_deepseek(mean_1, cov_1, mean_2, cov_2):
    """
    Compute KL divergence between two multivariate Gaussians.
    
    Args:
        mean_1: (d,) tensor - mean of learned distribution.
        cov_1:  (d, d) tensor - covariance of learned distribution.
        mean_2: (d,) tensor - mean of prior distribution.
        cov_2:  (d, d) tensor - covariance of prior distribution.
    
    Returns:
        KL( N(mean_1, cov_1) || N(mean_2, cov_2) )
    """
    # Ensure inputs are tensors
    mean_1 = torch.as_tensor(mean_1)
    cov_1 = torch.as_tensor(cov_1)
    mean_2 = torch.as_tensor(mean_2)
    cov_2 = torch.as_tensor(cov_2)
    
    # Add small diagonal noise to covariances for numerical stability
    eps = 1e-6
    cov_1 = cov_1 + eps * torch.eye(cov_1.shape[0], device=cov_1.device)
    cov_2 = cov_2 + eps * torch.eye(cov_2.shape[0], device=cov_2.device)
    
    # Compute terms
    dim = mean_1.shape[-1]
    cov_2_inv = torch.linalg.inv(cov_2)
    diff = mean_2 - mean_1
    
    # Trace term: tr(Σ₂⁻¹ Σ₁)
    trace_term = torch.trace(cov_2_inv @ cov_1)
    
    # Quadratic term: (μ₂ - μ₁)ᵀ Σ₂⁻¹ (μ₂ - μ₁)
    quadratic_term = diff.T @ cov_2_inv @ diff
    
    # Log determinant term: ln(det(Σ₂)/det(Σ₁)) = ln(det(Σ₂)) - ln(det(Σ₁))
    logdet_cov2 = torch.logdet(cov_2)
    logdet_cov1 = torch.logdet(cov_1)
    logdet_term = logdet_cov2 - logdet_cov1
    
    # Combine all terms
    kl = 0.5 * (trace_term + quadratic_term - dim + logdet_term)
    return kl

def compute_kl_divergence_deepseek(us, device):
    """Compute KL between learned exogenous vars and prior N(0, I)."""
    us = torch.tensor(us, device=device).float()  # (batch, time, dim)
    batch, dim = us.shape
    
    # Reshape to (batch*time, dim)
    us_flat = us.reshape(-1, dim)
    
    # Compute mean and covariance
    mean = torch.mean(us_flat, dim=0)  # (dim,)
    cov = torch.cov(us_flat.T)         # (dim, dim)
    
    # Prior: N(0, I)
    prior_mean = torch.zeros(dim, device=device)
    prior_cov = torch.eye(dim, device=device)
    
    return kl_multivariate_normal(mean, cov, prior_mean, prior_cov)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sliding_window_view_torch(x, window_size: int):
    """
    A function to create a 2D sliding window view of a 2D PyTorch tensor.

    Args:
    x (torch.Tensor): The input 2D tensor.
    window_size (int): Window size.

    Returns:
    torch.Tensor: A tensor with the sliding windows.
    """
    # Calculate output shape
    output_shape = (x.size(0) - window_size + 1, window_size, x.size(1))
    # Calculate strides
    strides = (x.stride(0), x.stride(0), x.stride(1))
    # Create a view
    return x.as_strided(size=output_shape, stride=strides)


def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        hamming_dist = np.sum(np.abs(a_true_offdiag - a_pred_offdiag)) / len(a_true_offdiag)
    else:
        precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        hamming_dist = np.sum(np.abs(a_true.flatten() - a_pred.flatten())) / len(a_true.flatten())
    return accuracy, bal_accuracy, precision, recall, hamming_dist


def eval_causal_structure(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))]
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))]
        if np.max(a_true_offdiag) == np.min(a_true_offdiag):
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
            auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
    return auroc, auprc


def construct_training_dataset(data, order):
    # Pack the data, if it is not in a list already
    if not isinstance(data, list):
        data = [data]

    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0
    for r in range(len(data)):
        data_r = data[r]
        # data: T x p
        T_r = data_r.shape[0]
        p_r = data_r.shape[1]
        inds_r = np.arange(order, T_r)
        data_out_r = np.zeros((T_r - order, order, p_r))
        response_r = np.zeros((T_r - order, p_r))
        time_idx_r = np.zeros((T_r - order, ))
        for i in range(T_r - order):
            j = inds_r[i]
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, :]
            time_idx_r[i] = j
        time_idx_r = time_idx_r + offset + 200 * (r >= 1)
        time_idx_r = time_idx_r.astype(int)
        if data_out is None:
            data_out = data_out_r
            response = response_r
            time_idx = time_idx_r
        else:
            data_out = np.concatenate((data_out, data_out_r), axis=0)
            response = np.concatenate((response, response_r), axis=0)
            time_idx = np.concatenate((time_idx, time_idx_r))
        offset = np.max(time_idx_r)
    return data_out, response, time_idx

def grimshaw(peaks:np.array, threshold:float, num_candidates:int=10, epsilon:float=1e-8):
    ''' The Grimshaw's Trick Method

    The trick of thr Grimshaw's procedure is to reduce the two variables
    optimization problem to a signle variable equation.

    Args:
        peaks: peak nodes from original dataset.
        threshold: init threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        gamma: estimate
        sigma: estimate
    '''
    min = peaks.min()
    max = peaks.max()
    mean = peaks.mean()

    if abs(-1 / max) < 2 * epsilon:
        epsilon = abs(-1 / max) / num_candidates

    a = -1 / max + epsilon
    b = 2 * (mean - min) / (mean * min)
    c = 2 * (mean - min) / (min ** 2)

    candidate_gamma = solve(function=lambda t: function(peaks, t),
                            dev_function=lambda t: dev_function(peaks, t),
                            bounds=(a + epsilon, -epsilon),
                            num_candidates=num_candidates
                            )
    candidate_sigma = solve(function=lambda t: function(peaks, t),
                            dev_function=lambda t: dev_function(peaks, t),
                            bounds=(b, c),
                            num_candidates=num_candidates
                            )
    candidates = np.concatenate([candidate_gamma, candidate_sigma])

    gamma_best = 0
    sigma_best = mean
    log_likelihood_best = cal_log_likelihood(peaks, gamma_best, sigma_best)

    for candidate in candidates:
        if candidate == 0 or np.isnan(candidate):
            continue
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = gamma / candidate
        log_likelihood = cal_log_likelihood(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best


def function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    return u * v - 1


def dev_function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    dev_u = (1 / threshold) * (1 - v)
    dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
    return u * dev_v + v * dev_u


def obj_function(x, function, dev_function):
    m = 0
    n = np.zeros(x.shape)
    for index, item in enumerate(x):
        y = function(item)
        m = m + y ** 2
        n[index] = 2 * y * dev_function(item)
    return m, n


def solve(function, dev_function, bounds, num_candidates):
    step = (bounds[1] - bounds[0]) / (num_candidates + 1)
    x0 = np.arange(bounds[0] + step, bounds[1], step)
    optimization = minimize(lambda x: obj_function(x, function, dev_function),
                            x0,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=[bounds]*len(x0)
                            )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)


def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma/sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else:
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood



def pot(data: np.array, risk: float = 1e-2, init_level: float = 0.98, num_candidates: int = 10,
        epsilon: float = 1e-8) -> float:
    ''' Peak-over-Threshold Alogrithm

    References:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        z: threshold searching by pot
        t: init threshold
    '''
    # Set init threshold0
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks,
                            threshold=t,
                            num_candidates=num_candidates,
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else:
        z = t - sigma * log(r)

    return z, t

def topk(z_scores, label, threshold, k_range=500):
    ''' Top-k method

    Args:
        us: anomaly scores
        label: ground truth

    Returns:
        k: the number of top-k nodes
    '''
    z_scores = np.array(z_scores)
    us_above_threshold = np.where(z_scores > threshold, z_scores, 0.0)
    label = np.array(label)
    us_above_threshold = us_above_threshold.flatten()
    label = label.flatten()
    ranking = np.argsort(us_above_threshold)
    label_ind = np.where(label == 1)[0]
    k_lst = []
    for k in range(1, k_range+1):
        count = [1 if i in label_ind else 0 for i in ranking[-k:]]
        k_lst.append(sum(count)/min(k, len(label_ind)))
    return np.array(k_lst)

def topk_at_step(scores, labels, k_range=10):
    k_lst = []
    for i in range(len(labels)):
        if sum(labels[i]) > 0:
            ranking = np.argsort(scores[i])
            label_ind = np.where(labels[i] == 1)[0]
            for k in range(1, k_range + 1):
                count = [1 if i in label_ind else 0 for i in ranking[-k:]]
                k_lst.append(sum(count) / min(k, len(label_ind)))
    return np.array(k_lst).reshape(-1, k_range).mean(axis=0)

def compute_mmd(x, y, kernel='rbf', gamma=1.0):
    """MMD between two sets of samples"""
    def pairwise_dist(a, b):
        return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(2)

    if kernel == 'rbf':
        Kxx = torch.exp(-pairwise_dist(x, x) * gamma).mean()
        Kyy = torch.exp(-pairwise_dist(y, y) * gamma).mean()
        Kxy = torch.exp(-pairwise_dist(x, y) * gamma).mean()
        return Kxx + Kyy - 2 * Kxy
    else:
        raise NotImplementedError("Only RBF kernel is implemented")
    

