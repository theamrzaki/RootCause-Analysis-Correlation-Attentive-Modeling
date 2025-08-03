import torch.nn as nn
import torch

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.
        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        """
        one neural network per past time step (k = 0, ..., K-1)
        each network outputs a generalised coefficient matrix of shape p x p
        (instead of an RNN, which mixes the past time steps together)
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()
        self.context_radius = 1  # context radius for gathering local context (e.g., k-1, k, k+1)
        self.input_dim = num_vars + 2 * self.context_radius * num_vars  # input dim for coeff nets
        self.num_vars = num_vars  # original number of variables (p)
        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(self.input_dim, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, self.num_vars**2), nn.Tanh()))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device
        

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            input_context = self._gather_local_context(inputs, k)  # gather context for the first time step
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(input_context)
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            # coeffs[:, k, :, :] = coeffs_k
            #preds = preds + torch.matmul(coeffs_k, input_context.unsqueeze(dim=2)).squeeze()
            lag_vec = inputs[:, k, :].unsqueeze(-1)  # shape [B, p, 1]
            preds = preds + torch.matmul(coeffs_k, lag_vec).squeeze(-1)
        return preds, coeffs
    
    def _gather_local_context(self, inputs, k):
        # inputs: [B, order, p]
        B, K, p = inputs.shape
        r = self.context_radius  # e.g., 1 for k-1, k, k+1
        window = []
        for offset in range(-r, r+1):
            idx = k + offset
            if 0 <= idx < K:
                window.append(inputs[:, idx, :])  # existing lag
            else:
                window.append(torch.zeros((B, p), device=inputs.device))  # padding
        # concatenate: shape [B, (2r+1)*p]
        return torch.cat(window, dim=1)
