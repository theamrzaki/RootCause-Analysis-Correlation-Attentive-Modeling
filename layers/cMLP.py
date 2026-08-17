

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_series, hidden):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_series,
            hidden_size=hidden,
            batch_first=True
        )

        self.proj = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, X, hidden=None):
        # X: [B, T, P]

        if hidden is None:
            B = X.size(0)
            h0 = torch.zeros(1, B, self.lstm.hidden_size, device=X.device)
            c0 = torch.zeros(1, B, self.lstm.hidden_size, device=X.device)
            hidden = (h0, c0)

        out, hidden = self.lstm(X, hidden)  # [B, T, H]

        out = out.transpose(1, 2)           # [B, H, T]
        out = self.proj(out)                # [B, 1, T]
        out = out.transpose(1, 2)           # [B, T, 1]

        return out, hidden

class cLSTM(nn.Module):
    def __init__(self, num_series, hidden):
        super().__init__()

        self.p = num_series

        # one LSTM per variable
        self.networks = nn.ModuleList([
            LSTM(num_series, hidden) for _ in range(num_series)
        ])

    def forward(self, X, hidden=None):
        """
        X: [B, T, P]
        returns:
            preds: [B, T, P]
        """

        if hidden is None:
            hidden = [None] * self.p

        outputs = []
        new_hidden = []

        for i in range(self.p):
            out_i, h_i = self.networks[i](X, hidden[i])
            outputs.append(out_i)
            new_hidden.append(h_i)

        preds = torch.cat(outputs, dim=2)  # [B, T, P]
        preds_last = preds[:, -1, :]       # [B, P]

        return preds_last, new_hidden