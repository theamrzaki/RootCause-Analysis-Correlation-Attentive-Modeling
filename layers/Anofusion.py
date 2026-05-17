import torch 
import sys
sys.path.append('./')
from layers.layers.GTblock import GTN
from layers.layers.GATGRU import *
import torch.nn as nn

class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

class AnoFusionWrapper(nn.Module):
    def __init__(self, num_services, window_size,
                 metric_dim, log_dim,trace_dim, hidden_dim=64):
        super().__init__()

        self.metric_proj = nn.Linear(metric_dim, hidden_dim)
        self.log_proj    = nn.Linear(log_dim, hidden_dim)
        self.trace_proj  = nn.Linear(trace_dim, hidden_dim)

        #self.linear_x = nn.Linear(30, 20)

        self.anofusion = Net(
            node_num=num_services,
            edge_types=1,
            window_samples_num=window_size,
            dropout=0.2
        )

        # FIX: explicit output projection
        self.out_dim = metric_dim + log_dim + trace_dim  # or any desired output dimension
        self.post_proj = nn.Linear(hidden_dim, self.out_dim)  # adjust if GAT_GRU outputs != 20

    def forward(self, graph, data_node, data_log, data_edge):
        B, T, N, _ = data_node.shape

        dn = self.metric_proj(data_node)
        dl = self.log_proj(data_log)
        #tr = self.trace_proj(data_edge.mean(dim=3))
        tr = self.trace_proj(data_edge)

        X = torch.cat([dn, dl, tr], dim=-1)  # [B,T,N,3]
        #X = self.linear_x(X)                 # [B,T,N,20]

        θ = min(self.anofusion.window_samples_num, T)
        Xw = X[:, -θ:]                       # [B,θ,N,20]

        A = graph.unsqueeze(0).unsqueeze(0).repeat(B, θ, 1, 1, 1)

        Xw = Xw.view(B*θ, N, -1)  # [Bθ,N,20]
        A  = A.view(B*θ, N, N, 1)
        device = Xw.device
        X_pred = self.anofusion(Xw, A, device)       # [Bθ,N,F]
        X_pred = self.post_proj(X_pred)      # [Bθ,N,out_dim]
        X_pred = torch.relu(X_pred)        # <<< REQUIRED

        return X_pred,None#X_pred.view(B, θ, N, self.out_dim), None

class Net(nn.Module):
    def __init__(self, node_num, edge_types, window_samples_num, dropout):
        super(Net, self).__init__()
        self.edge_types = edge_types
        self.num_channels = edge_types
        self.node_num = node_num
        self.window_samples_num = window_samples_num
        self.dropout = dropout
        self.GTN = GTN(edge_types=self.edge_types, num_channels=self.num_channels, num_layers=5, norm=False)
        self.GAT_GRU = GAT_GRU(self.window_samples_num, self.node_num, self.num_channels)
        self.flatten = nn.Flatten()
        self.linT = nn.Linear(self.window_samples_num, self.window_samples_num // 2)
    
        self.all = self.window_samples_num * self.node_num
        self.Dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(self.all, 64)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(64, 2)
        self._final_softmax = nn.Softmax(dim=1)
      
        
    def forward(self, X, A, device):
        X = self.Dropout(X)
        A = A.view((-1, self.node_num, self.node_num, self.edge_types))
        X = X.view((-1, self.node_num, X.shape[-1]))
        # GTN
        #device = X.device
        A = self.GTN(A, device)
        # GAT and GRU
        out_T = self.GAT_GRU(X, A, device)
        return out_T