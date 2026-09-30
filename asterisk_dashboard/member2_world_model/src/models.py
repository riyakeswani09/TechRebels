
import torch
from torch import nn

class GRUWorldModel(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, horizon=5, dropout=.2):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(n_features, hidden, layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0)
        self.state_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_features)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, horizon)
        )

    def forward(self, x):
        out, h = self.encoder(x)
        z = h[-1]
        future_state = self.state_head(z).unsqueeze(1).repeat(1,self.horizon,1)
        risk_logits = self.risk_head(z)
        return future_state, risk_logits

class TemporalTransformerWorldModel(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=4, layers=3,
                 horizon=5, dropout=.15):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

        # Direct multi-horizon prediction: no autoregressive K-step rollout.
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*2, horizon*n_features)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        h = self.input_proj(x) + self.pos[:, :x.size(1)]
        h = self.norm(self.encoder(h))
        z = h[:, -1]
        state = self.state_head(z).view(x.size(0), self.horizon, -1)
        risk_logits = self.risk_head(z)
        return state, risk_logits
