import torch.nn as nn
import torch

from layers.Informer import Informer

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
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        #self.coeff_nets = nn.ModuleList()
##
        ## Instantiate coefficient networks
        #for k in range(order):
        #    modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
        #    if num_hidden_layers > 1:
        #        for j in range(num_hidden_layers - 1):
        #            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
        #    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
        #    self.coeff_nets.append(nn.Sequential(*modules))
        self.rank = 10
        self.coeff_nets = nn.ModuleList()
        for k in range(order):
            layers = [nn.Linear(num_vars, hidden_layer_size), nn.ReLU()]
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                layers.append(nn.ReLU())
            # Output size = p*r (U) + p*r (V)
            layers.append(nn.Linear(hidden_layer_size, 2 * num_vars * self.rank ))
            self.coeff_nets.append(nn.Sequential(*layers))

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
    def forwardaaaaaaa(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            # coeffs[:, k, :, :] = coeffs_k
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs
    

    def forward(self, inputs: torch.Tensor):
        """
        inputs: (B, K, p)
        returns:
            preds: (B, p)
            coeffs: (B, K, p, p) — full matrix for compatibility
        """
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars), device=self.device)

        for k in range(self.order):
            out = self.coeff_nets[k](inputs[:, k, :])
            # Split into U and V
            U_flat, V_flat = torch.split(out, self.num_vars * self.rank, dim=1)
            U = U_flat.view(-1, self.num_vars, self.rank)
            V = V_flat.view(-1, self.num_vars, self.rank)

            coeffs_k = torch.bmm(U, V.transpose(1, 2))  # (B, p, p)

            if coeffs is None:
                coeffs = coeffs_k.unsqueeze(1)
            else:
                coeffs = torch.cat((coeffs, coeffs_k.unsqueeze(1)), 1)

            preds = preds + torch.bmm(coeffs_k, inputs[:, k, :].unsqueeze(2)).squeeze(2)

        return preds, coeffs