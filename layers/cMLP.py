import torch
import torch.nn as nn

class cMLP(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=256, device="cpu", options=None):
        super(cMLP, self).__init__()
        self.p = num_vars
        self.lag = order
        self.device = device

        # To be faithful to the GC() function, each network needs 
        # a structure where the first layer separates lags and variables.
        self.networks = nn.ModuleList([
            ComponentMLP(num_vars, order, hidden_dim)
            for _ in range(num_vars)
        ])

    def forward(self, X):
        # Reference expects cat on dim 2 (the sensor dim)
        # X: [Batch, Time, Sensors]
        preds = torch.cat([net(X) for net in self.networks], dim=-1)
        
        # vlinear signature compatibility
        coeffs_freq = self.GC(threshold=False)
        B, _, _ = X.shape
        coeffs_time = coeffs_freq.unsqueeze(0).unsqueeze(0).repeat(B, self.lag, 1, 1)
        
        return preds, coeffs_time, coeffs_freq

    def GC(self, threshold=False, ignore_lag=True):
        # Faithful extraction of Granger Causality
        if ignore_lag:
            # Norm across the hidden units (0) and the lags (2)
            # Result: [P] vector for each network
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            # Result: [P, Lag] for each network
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        
        GC = torch.stack(GC).to(self.device)
        return (GC > 0).int() if threshold else GC

class ComponentMLP(nn.Module):
    def __init__(self, p, lag, hidden):
        super().__init__()
        # To match the GC norm logic: weight must be [hidden, p * lag] 
        # but interpreted as [hidden, p, lag] for the norm dim (0, 2)
        self.layers = nn.ModuleList([
            nn.Linear(p * lag, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ])

    def forward(self, X):
        B, T, P = X.shape
        x_flat = X.reshape(B, -1) # Flatten T and P
        return self.layers[0:3](x_flat) # Simplified for example
        # In actual practice:
        out = x_flat
        for layer in self.layers:
            out = layer(out)
        return out