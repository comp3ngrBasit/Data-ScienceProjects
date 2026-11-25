"""Direct multi-output LSTM forecaster (PyTorch).

Input: the last ``lookback`` steps of [scaled target, cyclical hour/day/year encodings, off-day flag].
Output: all ``horizon`` future steps at once. Early stopping on the chronological validation block,
split-conformal prediction intervals from validation residuals.
"""
from __future__ import annotations

import numpy as np

from app.config import RANDOM_SEED, SETTINGS
from app.data.feature_engineering import calendar_features
from app.models.forecasting.models import BaseForecaster, Forecast
from app.utils.logging import get_logger

log = get_logger("forecast.lstm")


def _features(spec) -> np.ndarray:
    cal = calendar_features(spec.y.index, spec.country, fourier_k=1)
    cols = ["year_sin1", "year_cos1", "is_offday"]
    dow = 2 * np.pi * spec.y.index.dayofweek / 7
    extra = np.column_stack([np.sin(dow), np.cos(dow)])
    if spec.season > 1:
        cols += ["day_sin1", "day_cos1"]
    return np.column_stack([cal[cols].to_numpy(float), extra])


class LSTMForecaster(BaseForecaster):
    def fit(self, spec, train_end, val_end):
        import torch
        from torch import nn

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.set_num_threads(2)
        self.name = "LSTM"
        self.lookback = 168 if spec.season > 1 else 56
        y = spec.y
        tr = y.loc[:train_end]
        self.mu, self.sd = float(tr.mean()), float(tr.std())
        z = ((y - self.mu) / self.sd).to_numpy(np.float32)
        X = np.column_stack([z, _features(spec)]).astype(np.float32)
        self.X, self.z, self.index = X, z, y.index
        H, L = self.horizon, self.lookback
        pos_train_end = y.index.get_loc(train_end)
        pos_val_end = y.index.get_loc(val_end)
        stride = 3 if spec.season > 1 else 1
        tr_orig = np.arange(L - 1, pos_train_end - H + 1, stride)
        va_orig = np.arange(pos_train_end, pos_val_end - H + 1, max(1, H // 4))

        def batch(origins):
            xs = np.stack([X[o - L + 1: o + 1] for o in origins])
            ys = np.stack([z[o + 1: o + 1 + H] for o in origins])
            return torch.from_numpy(xs), torch.from_numpy(ys)

        Xtr, Ytr = batch(tr_orig)
        Xva, Yva = batch(va_orig)

        class Net(nn.Module):
            def __init__(self, n_in, hidden, out):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.head(o[:, -1])

        net = Net(X.shape[1], 64, H)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        loss_fn = nn.MSELoss()
        best, best_state, patience, bad = np.inf, None, 3, 0
        self.history = []
        epochs = SETTINGS.forecast.lstm_epochs
        g = torch.Generator().manual_seed(RANDOM_SEED)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            tot = 0.0
            for i in range(0, len(perm), 256):
                b = perm[i: i + 256]
                opt.zero_grad()
                loss = loss_fn(net(Xtr[b]), Ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(Xva), Yva).item()
            self.history.append({"epoch": ep + 1, "train_mse": tot / len(Xtr), "val_mse": vl})
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred_va = net(Xva).numpy() * self.sd + self.mu
        resid = (Yva.numpy() * self.sd + self.mu) - pred_va
        self.q = float(np.quantile(np.abs(resid), 1 - self.alpha))
        self.net = net
        self.params = {"lookback": L, "hidden": 64, "layers": 2, "epochs_run": len(self.history),
                       "best_val_mse": best, "train_samples": len(Xtr)}
        return self

    def predict(self, spec, origin):
        import torch

        o = self.index.get_loc(origin)
        x = torch.from_numpy(self.X[o - self.lookback + 1: o + 1][None])
        with torch.no_grad():
            m = self.net(x).numpy()[0] * self.sd + self.mu
        return Forecast(m, m - self.q, m + self.q)
