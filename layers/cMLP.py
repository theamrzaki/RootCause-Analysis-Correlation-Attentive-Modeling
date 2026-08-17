

import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act

class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        super().__init__()

        self.activation = activation_helper(activation)

        # Input:
        #   [B, T, P]
        #
        # After transpose:
        #   [B, P, T]
        #
        # Conv1d:
        #   in_channels  = P
        #   kernel_size  = lag
        #
        # This learns from the previous `lag` time points.
        layers = [
            nn.Conv1d(
                in_channels=num_series,
                out_channels=hidden[0],
                kernel_size=lag
            )
        ]

        for d_in, d_out in zip(
            hidden,
            hidden[1:] + [1]
        ):
            layers.append(
                nn.Conv1d(
                    in_channels=d_in,
                    out_channels=d_out,
                    kernel_size=1
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        # X: [B, T, P]
        X = X.transpose(2, 1)  # [B, P, T]

        for i, layer in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)

            X = layer(X)

        # [B, 1, T-lag+1] -> [B, T-lag+1, 1]
        return X.transpose(2, 1)


class cMLP(nn.Module):
    def __init__(
        self,
        num_series,
        lag,
        hidden,
        activation='relu'
    ):
        super().__init__()

        self.p = num_series
        self.lag = lag

        # One MLP for each target variable
        self.networks = nn.ModuleList([
            MLP(
                num_series=num_series,
                lag=lag,
                hidden=hidden,
                activation=activation
            )
            for _ in range(num_series)
        ])

    def forward(self, X):
        """
        X: [B, T, P]

        Returns:
            preds_last: [B, P]
        """

        # Each network predicts one variable:
        #
        # network_i(X):
        #   [B, T, P] -> [B, T-lag+1, 1]
        #
        outputs = [
            network(X)
            for network in self.networks
        ]

        # [B, T-lag+1, P]
        preds = torch.cat(outputs, dim=2)

        # Same interface as your cLSTM:
        # only return prediction at the last time step
        preds_last = preds[:, -1, :]  # [B, P]

        return preds_last, None

    def GC(self, threshold=True, ignore_lag=True):
        """
        Extract learned Granger causality.

        Returns:
            ignore_lag=True:
                [P, P]

            ignore_lag=False:
                [P, P, lag]
        """

        if ignore_lag:
            GC = [
                torch.norm(
                    net.layers[0].weight,
                    dim=(0, 2)
                )
                for net in self.networks
            ]
        else:
            GC = [
                torch.norm(
                    net.layers[0].weight,
                    dim=0
                )
                for net in self.networks
            ]

        GC = torch.stack(GC)

        if threshold:
            return (GC > 0).int()
        else:
            return GC