import torch.nn as nn
import torch

from layers.SimpleGNN import AttentionCoeffGNN

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int,
                 args: dict,  device: torch.device):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.
        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        super(SENNGC, self).__init__()
        self.args = args

        if args["coeff_architecture"] == "deep_mlp":
            # Networks for amortising generalised coefficient matrices.
            self.coeff_nets = nn.ModuleList()

            ## Instantiate coefficient networks
            for k in range(order):
                modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
                if num_hidden_layers > 1:
                    for j in range(num_hidden_layers - 1):
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
                self.coeff_nets.append(nn.Sequential(*modules))

        elif args["coeff_architecture"] == "gnn_attention":
            self.rank = 20
            self.coeff_nets = nn.ModuleList()
            for k in range(order):
                self.coeff_nets.append(AttentionCoeffGNN(num_vars=num_vars, rank=self.rank ))
        
        total_params = sum(p.numel() for net in self.coeff_nets for p in net.parameters())
        print(f"Total parameters for {order} lags: {total_params}")
        
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
    def forward_normal(self, inputs: torch.Tensor):
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
    

    def forward_gnn(self, inputs: torch.Tensor):
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
        if self.args["coeff_architecture"] == "deep_mlp":
            return self.forward_normal(inputs)
        elif self.args["coeff_architecture"] == "gnn_attention":
            return self.forward_gnn(inputs)