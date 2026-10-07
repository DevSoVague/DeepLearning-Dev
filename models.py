"""
Model definitions for the final (v12) turbofan RUL models.

Copied from dev_contri_1_2_final.ipynb so the shipped checkpoints in
checkpoints/checkpoints_v12/ can be loaded without running the notebook.
The classes and their default hyperparameters are unchanged; only the
notebook-only plotting helpers are left out.

    from models import build_model
    model = build_model("transformer", grl=False)      # 14 input sensors
    model.load_state_dict(torch.load(path)["model"])
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

N_FEATURES = 14  # sensors kept after dropping the 7 flat ones


# ── Gradient reversal (used by the GRL variants) ─────────────────────────────
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        lam, = ctx.saved_tensors
        return -lam.item() * grad, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lam)


def ganin_lambda(epoch, n_epochs, high=1.0):
    p = epoch / n_epochs
    return high * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ── TCN ──────────────────────────────────────────────────────────────────────
class TCNResidualBlock(nn.Module):
    def __init__(self, n_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_ch, n_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(n_ch, n_ch, kernel_size, dilation=dilation, padding=pad)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(n_ch)
        self.norm2 = nn.LayerNorm(n_ch)

    def _trim(self, x, T):
        return x[:, :, :T]

    def forward(self, x):
        T = x.size(2)
        h = F.relu(self.norm1(self._trim(self.conv1(x), T).transpose(1, 2)).transpose(1, 2))
        h = self.drop(h)
        h = F.relu(self.norm2(self._trim(self.conv2(h), T).transpose(1, 2)).transpose(1, 2))
        return h + x


class TCN(nn.Module):
    def __init__(self, n_features, n_channels=32, kernel_size=3, n_blocks=4, dropout=0.2):
        super().__init__()
        self.enc_dim = n_channels
        self.input_proj = nn.Conv1d(n_features, n_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNResidualBlock(n_channels, kernel_size, 2 ** i, dropout) for i in range(n_blocks)])
        self.rul_head = nn.Sequential(nn.Linear(n_channels, 16), nn.ReLU(), nn.Linear(16, 1))
        self.state_head = nn.Sequential(nn.Linear(n_channels, 16), nn.ReLU(),
                                        nn.Linear(16, 1), nn.Sigmoid())

    def encode(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        return x.mean(dim=2)

    def forward(self, x):
        return self.rul_head(self.encode(x)).squeeze(-1)

    def forward_with_state(self, x):
        h = self.encode(x)
        return self.rul_head(h).squeeze(-1), self.state_head(h).squeeze(-1)


# ── Lightweight Transformer ──────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LightweightTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.enc_dim = d_model
        self.d_model = d_model
        self.nhead = nhead
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.rul_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.state_head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(),
                                        nn.Linear(16, 1), nn.Sigmoid())

    def encode(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        return x.mean(dim=1)

    def encode_with_attn(self, x):
        """Return (encoding, attention weights averaged over heads and layers)."""
        x = self.pos_enc(self.input_proj(x))
        attn_maps = []
        for layer in self.encoder.layers:
            x2, attn = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
            x2 = layer.norm1(x + layer.dropout1(x2))
            x2 = layer.norm2(x2 + layer.dropout2(layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x2))))))
            x = x2
            attn_maps.append(attn.detach().cpu())
        return x.mean(dim=1), torch.stack(attn_maps).mean(0)

    def forward(self, x):
        return self.rul_head(self.encode(x)).squeeze(-1)

    def forward_with_state(self, x):
        h = self.encode(x)
        return self.rul_head(h).squeeze(-1), self.state_head(h).squeeze(-1)


# ── TCN-GRU hybrid ───────────────────────────────────────────────────────────
class TCNGRUHybrid(nn.Module):
    def __init__(self, n_features, tcn_channels=32, tcn_blocks=3,
                 kernel_size=3, gru_hidden=64, dropout=0.2):
        super().__init__()
        self.enc_dim = tcn_channels + gru_hidden * 2
        self.tcn_proj = nn.Conv1d(n_features, tcn_channels, kernel_size=1)
        self.tcn_blocks = nn.ModuleList([
            TCNResidualBlock(tcn_channels, kernel_size, 2 ** i, dropout)
            for i in range(tcn_blocks)])
        self.gru = nn.GRU(n_features, gru_hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.gru_norm = nn.LayerNorm(gru_hidden * 2)
        self.rul_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.enc_dim, 64),
            nn.ReLU(), nn.Linear(64, 1))
        self.state_head = nn.Sequential(
            nn.Linear(self.enc_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def encode(self, x):
        t = x.transpose(1, 2)
        t = self.tcn_proj(t)
        for b in self.tcn_blocks:
            t = b(t)
        out, _ = self.gru(x)
        return torch.cat([t.mean(dim=2), self.gru_norm(out[:, -1, :])], dim=1)

    def forward(self, x):
        return self.rul_head(self.encode(x)).squeeze(-1)

    def forward_with_state(self, x):
        h = self.encode(x)
        return self.rul_head(h).squeeze(-1), self.state_head(h).squeeze(-1)


# ── GRL wrapper (domain-adversarial variants) ────────────────────────────────
class GRLWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(encoder.enc_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def predict_rul(self, x):
        return self.encoder(x)

    def forward(self, x_src, x_tgt):
        h_src = self.encoder.encode(x_src)
        h_tgt = self.encoder.encode(x_tgt)
        rul_pred = self.encoder.rul_head(h_src).squeeze(-1)
        state_pred = self.encoder.state_head(h_src).squeeze(-1)
        h_all = torch.cat([h_src, h_tgt], dim=0)
        dom_logits = self.domain_head(self.grl(h_all)).squeeze(-1)
        return rul_pred, state_pred, dom_logits


# ── Factory matching the notebook's hyperparameters ──────────────────────────
def build_encoder(arch, n_features=N_FEATURES):
    if arch == "tcn":
        return TCN(n_features, n_channels=32, dropout=0.2)
    if arch == "transformer":
        return LightweightTransformer(n_features, d_model=64, nhead=4,
                                      dim_feedforward=128, dropout=0.1)
    if arch == "tcn_gru":
        return TCNGRUHybrid(n_features, tcn_channels=32, gru_hidden=64, dropout=0.2)
    raise ValueError(f"unknown arch {arch!r} (tcn, transformer, tcn_gru)")


def build_model(arch, grl=False, n_features=N_FEATURES):
    enc = build_encoder(arch, n_features)
    return GRLWrapper(enc) if grl else enc
