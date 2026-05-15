import torch.nn as nn
import torch


from layers.vlinear_arch import vlinear,MultiModalVLinear
from layers.cLSTM import cLSTM
from layers.CUTS_PLUS import CUTS_PLUS_Wrapper
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

        if args["coeff_architecture"] == "deep_mlp" or args["coeff_architecture"] == "GVAR":
            # Networks for amortising generalised coefficient matrices.
            self.coeff_nets = nn.ModuleList()

            ## Instantiate coefficient networks
            for k in range(order-1):
                modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
                if num_hidden_layers > 1:
                    for j in range(num_hidden_layers - 1):
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
                self.coeff_nets.append(nn.Sequential(*modules))
     
        if args["coeff_architecture"] not in  ["ht","epsilon_diagnosis","rcd","TemporalGNN","cross_time_freq","cross_attention_single_coeff_network","TemporalGNN_Attention","trend_seasonal","rcd","TemporalGNN_Attention_fourier","TemporalGNN_Attention_crossattn","TemporalGNN_Attention_crossattn_Legendre","TemporalGNN_Attention_crossattn_enhanced","causalrca","cuts_mlp","cuts_lstm","GVAR","vlinear","nsigma","baro","circa","torai","cLSTM","CUTS_PLUS"]:
            total_params = sum(p.numel() for net in self.coeff_nets for p in net.parameters())
            print(f"Total parameters for {order} lags: {total_params}")
        
        if args["coeff_architecture"] in "vlinear":
            if self.args["include_logs_and_traces"] == 0:
                self.coeff_net = vlinear(
                    num_vars=num_vars,
                    hidden_dim=hidden_layer_size,
                    order=order,
                    device=device,
                    options = args  # default to None if not specified
                )
            elif self.args["include_logs_and_traces"] == 1:
                #self.coeff_net = MultiModalVLinear(
                #    metric_dim=num_vars,
                #    log_dim=10,
                #    trace_dim=10,
                #    hidden_dim=hidden_layer_size,
                #    order=order,
                #    device=torch.device,
                #    options = args  # default to None if not specified
                #)
                self.coeff_net = vlinear(
                    num_vars=num_vars,
                    hidden_dim=hidden_layer_size,
                    order=order,
                    device=device,
                    options = args  # default to None if not specified
                )


        if args["coeff_architecture"] == "cLSTM":
            self.coeff_net = cLSTM(num_vars, hidden_layer_size)

        if args["coeff_architecture"] == "CUTS_PLUS":
            # Time branch: separate MLP for each lag
            self.coeff_net = CUTS_PLUS_Wrapper(
                num_vars=num_vars,
                hidden_dim=hidden_layer_size,
                order=order,
                device=device,
                options = args  # default to None if not specified
            )
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
        # Shape check using your preferred style
        if inputs.shape[1] != (self.order - 1):
            print(f"WARNING: Expected {self.order-1} steps, got {inputs.shape[1]}")

        coeffs = [] # Using a list is faster than torch.cat in a loop
        preds = torch.zeros((inputs.shape[0], self.num_vars), device=self.device)
        
        for k in range(self.order - 1):
            coeffs_k = self.coeff_nets[k](inputs[:, k, :])
            coeffs_k = coeffs_k.view(inputs.shape[0], self.num_vars, self.num_vars)
            
            coeffs.append(coeffs_k.unsqueeze(1))
            
            # Prediction: A_k * x_k
            # Squeeze(2) ensures we go from [BS, 30, 1] back to [BS, 30]
            preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(2)).squeeze(2)

        return preds, torch.cat(coeffs, dim=1), None

    def forward_temporal(self, inputs: torch.Tensor):
        """
        inputs: (B, order, num_vars)
        TemporalGNN processes the entire lag sequence recurrently.
        """
        preds, coeffs, attn_weights = self.coeff_net(inputs)  # let TemporalGNN return preds + coeffs
        return preds, coeffs, attn_weights
        
    def forward_simple_nextstep(self, inputs: torch.Tensor):
        """
        Simple forward pass for next-step prediction without returning coefficients.
        """
        preds,_ = self.coeff_net(inputs)
        return preds
    
    def forward(self, inputs: torch.Tensor):
        if self.args["coeff_architecture"] == "deep_mlp" or self.args["coeff_architecture"] == "GVAR":
            return self.forward_normal(inputs)

        elif self.args["coeff_architecture"] in ["vlinear","CUTS_PLUS"]:
            return self.forward_temporal(inputs)
        
        elif self.args["coeff_architecture"] == "cLSTM":
            return self.forward_simple_nextstep(inputs), None, None
