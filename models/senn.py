import torch.nn as nn
import torch


from layers.vlinear_arch import vlinear
from layers.cLSTM import cLSTM
from layers.cMLP import cMLP
from layers.CUTS_PLUS import CUTS_PLUS_Wrapper
from layers.Eadro import MainModel as Eadro 
from layers.Anofusion import AnoFusionWrapper as Anofusion

from layers.iTransformer import Model as iTransformer
from layers.TimeMixerpp import Model as TimeMixerpp

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int,
                 graph_structure: torch.Tensor, 
                 args: dict, device: torch.device):
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
        self.graph_structure = graph_structure #only used for Topology-aware models (Eadro)
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
     
        if args["coeff_architecture"] not in  ["ht","epsilon_diagnosis","rcd","TemporalGNN","cross_time_freq","cross_attention_single_coeff_network","TemporalGNN_Attention","trend_seasonal","rcd","TemporalGNN_Attention_fourier","TemporalGNN_Attention_crossattn","TemporalGNN_Attention_crossattn_Legendre","TemporalGNN_Attention_crossattn_enhanced","causalrca","cuts_mlp","cuts_lstm","GVAR","vlinear","nsigma","baro","circa","torai","cLSTM","cMLP","CUTS_PLUS","Fits","Dlinear","iTransformer","TimeMixerpp"]:
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
                num_pods = args["num_pods"]

                md = args["num_metrics"]
                ld = args["num_log_features"]
                td = args["num_trace_features"]
                self.coeff_net = MultiModalVLinear(
                    md=md,
                    ld=ld,
                    td=td,
                    N=num_pods,
                    h=hidden_layer_size,
                    order=order,
                    device=device,
                    opt=args
                )
                
        if args["coeff_architecture"] == "cLSTM":
            self.coeff_net = cLSTM(num_vars, hidden_layer_size)

        if args["coeff_architecture"] == "cMLP":
            self.coeff_net = cMLP(
                num_series=num_vars,
                lag=order-1,#as in vlinear, we use order-1
                hidden=[hidden_layer_size],
                activation='relu'
            )
        if args["coeff_architecture"] == "CUTS_PLUS":
            # Time branch: separate MLP for each lag
            self.coeff_net = CUTS_PLUS_Wrapper(
                num_vars=num_vars,
                hidden_dim=hidden_layer_size,
                order=order,
                device=device,
                options = args  # default to None if not specified
            )

        
        #Non Causal Architectures (FITS, DLinear, iTransformer, TimeMixerpp)
        if args["coeff_architecture"] in ["Fits","Dlinear","iTransformer","TimeMixerpp"]:
            class SimpleConfig:
                """Lightweight config container (like an empty struct)."""
                pass


            config = SimpleConfig()

            # ===== Basic Parameters =====
            config.win_size = args["window_size"]             # Window size
            config.DSR = 1                              # Downsampling rate
            config.cutfreq = 0                          # Cut frequency for FITS (0 = auto)

            if config.cutfreq == 0:
                config.cutfreq = int((config.win_size / config.DSR) / 2)

            assert (config.win_size / config.DSR) / 2 >= config.cutfreq, \
                'cutfreq should be smaller than half of the window size after downsampling'

            # ===== Sequence Parameters =====
            config.seq_len = config.win_size // config.DSR -1
            config.pred_len = 0                         # No prediction horizon for anomaly detection
            config.individual = False
            config.num_class = 1 #not used for anomaly detection
            config.task_name = "anomaly_detection"
            config.output_attention = True  # store_true equivalent (bool, not string)

            # ===== Embedding Parameters =====
            config.embed = "timeF"                      # Time features encoding: [timeF, fixed, learned]
            config.freq = "h"                           # Frequency (hourly)
            config.dropout = 0.1

            # ===== Model Architecture =====
            config.d_model = args["hidden_layer_size"]
            config.factor = 1
            config.n_heads = 8
            config.d_ff = args["hidden_layer_size"]
            config.enc_in = args["num_vars"]
            config.activation = "gelu"
            config.e_layers = 2
            config.device = device
            print("seq_len:", config.seq_len, "pred_len:", config.pred_len, "d_model:", config.d_model, "d_ff:", config.d_ff, "enc_in:", config.enc_in)
    

            if args["coeff_architecture"] == "Fits":
                self.coeff_net = Fits(configs=config)
            elif args["coeff_architecture"] == "Dlinear":
                self.coeff_net = DLinear(configs=config)
            elif args["coeff_architecture"] == "iTransformer":
                self.coeff_net = iTransformer(configs=config)
            elif args["coeff_architecture"] == "TimeMixerpp":
                self.coeff_net = TimeMixerpp(configs=config)



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
        if self.args["coeff_architecture"] in ["Eadro","Anofusion"]:
             
            graph = torch.from_numpy(self.graph_structure).to(self.device)

            B, T, N, F = inputs.shape
            x = inputs  # KEEP FULL TIME

            md = self.args["num_metrics"]
            ld = self.args["num_log_features"]
            td = self.args["num_trace_features"]  
            metrics = x[:, :, :, :md]                     # [B,T,N,md]
            logs    = x[:, :, :, md:md+ld]                # [B,T,N,ld]
            traces  = x[:, :, :, md+ld:md+ld+td]          # [B,T,N,2K]
            preds, _ = self.coeff_net(graph,metrics,logs,traces)
        else:
            preds,_ = self.coeff_net(inputs)
        return preds
    
    def forward(self, inputs: torch.Tensor):
        if self.args["coeff_architecture"] == "deep_mlp" or self.args["coeff_architecture"] == "GVAR":
            return self.forward_normal(inputs)

        elif self.args["coeff_architecture"] in ["vlinear","CUTS_PLUS"]:
            return self.forward_temporal(inputs)
        
        elif self.args["coeff_architecture"] in ["cLSTM","cMLP","Fits","Dlinear","iTransformer","TimeMixerpp"]:
            return self.forward_simple_nextstep(inputs), None, None
