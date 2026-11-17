"""Action classifiers behind one interface.

* Baselines on window-summary features: Random Forest, XGBoost, SVM (scikit-learn / xgboost).
* Temporal models on the per-frame sequence (T x F): bidirectional LSTM and a Temporal
  Convolutional Network (PyTorch, CPU or CUDA).

All expose ``fit(X_seq, y)``, ``predict_proba(X_seq)``, ``save(dir)`` and ``load(dir)``.
``X_seq`` is always (N, T, F); classical models summarise it internally.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from app.actions.features import summarise

log = logging.getLogger(__name__)
ALGORITHMS = ["random_forest", "xgboost", "svm", "lstm", "tcn"]


def _summ(X: np.ndarray) -> np.ndarray:
    return np.stack([summarise(x) for x in X]) if len(X) else np.zeros((0, 1))


class _Base:
    algorithm = ""

    def __init__(self, classes: list[str], params: dict | None = None, device: str = "cpu"):
        self.classes = list(classes)
        self.params = params or {}
        self.device = device
        self.info: dict = {}

    def _y(self, y: list[str]) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.classes)}
        return np.array([idx[v] for v in y])


class SklearnModel(_Base):
    def __init__(self, algorithm: str, classes, params=None, device="cpu"):
        super().__init__(classes, params, device)
        self.algorithm = algorithm
        self.model = None

    def _make(self, n_classes: int):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        if self.algorithm == "random_forest":
            return RandomForestClassifier(n_estimators=int(self.params.get("n_estimators", 300)),
                                          max_depth=self.params.get("max_depth"), class_weight="balanced_subsample",
                                          min_samples_leaf=1, n_jobs=-1, random_state=0)
        if self.algorithm == "svm":
            return make_pipeline(StandardScaler(), SVC(C=float(self.params.get("C", 3.0)), kernel="rbf", gamma="scale",
                                                       probability=True, class_weight="balanced", random_state=0))
        if self.algorithm == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(n_estimators=int(self.params.get("n_estimators", 200)), max_depth=int(self.params.get("max_depth", 4)),
                                 learning_rate=float(self.params.get("learning_rate", 0.1)), subsample=0.9, colsample_bytree=0.6,
                                 objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                                 tree_method="hist", n_jobs=0, random_state=0, verbosity=0)
        raise ValueError(self.algorithm)

    def fit(self, X, y):
        yi = self._y(y)
        present = np.unique(yi)
        self._present = present
        # Train on the labels present; remap so estimators see 0..k-1 (xgboost requires it)
        remap = {c: i for i, c in enumerate(present)}
        self.model = self._make(len(present))
        Xs = _summ(X)
        kw = {}
        if self.algorithm == "xgboost":
            from sklearn.utils.class_weight import compute_sample_weight

            kw["sample_weight"] = compute_sample_weight("balanced", yi)
        if self.algorithm == "svm" and len(present) > 1 and min(np.bincount(yi)[present]) < 2:
            # SVC(probability=True) needs >=2 samples per class for its internal CV; duplicate singletons
            reps = [i for i in range(len(yi)) if np.bincount(yi)[yi[i]] < 2]
            Xs = np.concatenate([Xs, Xs[reps]])
            yi = np.concatenate([yi, yi[reps]])
        self.model.fit(Xs, np.array([remap[v] for v in yi]), **kw)
        return self

    def predict_proba(self, X):
        Xs = _summ(X)
        p = self.model.predict_proba(Xs)
        out = np.zeros((len(Xs), len(self.classes)), np.float32)
        out[:, self._present] = p
        return out

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "present": self._present}, d / "model.joblib")

    def load(self, d: Path):
        obj = joblib.load(d / "model.joblib")
        self.model, self._present = obj["model"], obj["present"]
        return self


# ----------------------------------------------------------------------------- temporal (PyTorch)
def _torch_modules():
    import torch
    from torch import nn

    class LSTMNet(nn.Module):
        def __init__(self, f, c, hidden=64, layers=2, dropout=0.3):
            super().__init__()
            self.norm = nn.LayerNorm(f)
            self.lstm = nn.LSTM(f, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0)
            self.att = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, c))

        def forward(self, x):
            h, _ = self.lstm(self.norm(x))
            w = torch.softmax(self.att(h), dim=1)  # attention over time
            return self.head((w * h).sum(1))

    class TCNBlock(nn.Module):
        def __init__(self, cin, cout, dilation, dropout):
            super().__init__()
            self.conv1 = nn.Conv1d(cin, cout, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(cout, cout, 3, padding=dilation, dilation=dilation)
            self.drop = nn.Dropout(dropout)
            self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
            self.act = nn.ReLU()

        def forward(self, x):
            y = self.drop(self.act(self.conv1(x)))
            y = self.drop(self.act(self.conv2(y)))
            return self.act(y + self.res(x))

    class TCNNet(nn.Module):
        def __init__(self, f, c, channels=64, dropout=0.2):
            super().__init__()
            self.norm = nn.LayerNorm(f)
            self.blocks = nn.Sequential(TCNBlock(f, channels, 1, dropout), TCNBlock(channels, channels, 2, dropout),
                                        TCNBlock(channels, channels, 4, dropout))
            self.head = nn.Linear(2 * channels, c)

        def forward(self, x):
            y = self.blocks(self.norm(x).transpose(1, 2))
            return self.head(torch.cat([y.mean(2), y.amax(2)], dim=1))

    return torch, nn, LSTMNet, TCNNet


class TorchModel(_Base):
    def __init__(self, algorithm: str, classes, params=None, device="cpu"):
        super().__init__(classes, params, device)
        self.algorithm = algorithm
        self.net = None
        self.feat_dim = None

    def _build(self, f):
        torch, nn, LSTMNet, TCNNet = _torch_modules()
        c = len(self.classes)
        return LSTMNet(f, c, hidden=int(self.params.get("hidden", 64))) if self.algorithm == "lstm" else TCNNet(f, c, channels=int(self.params.get("channels", 64)))

    def fit(self, X, y):
        torch, nn, *_ = _torch_modules()
        torch.manual_seed(0)
        np.random.seed(0)
        yi = self._y(y)
        self.feat_dim = X.shape[2]
        self.net = self._build(self.feat_dim).to(self.device)
        counts = np.bincount(yi, minlength=len(self.classes)).astype(np.float32)
        w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / max((counts > 0).sum(), 1), 0.0)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=self.device), label_smoothing=0.05)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.params.get("lr", 2e-3)), weight_decay=1e-3)
        epochs = int(self.params.get("epochs", 60))
        bs = int(self.params.get("batch_size", 32))
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(yi, dtype=torch.long)
        start = time.monotonic()
        self.net.train()
        for _ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]
                xb = Xt[b].to(self.device)
                # light augmentation: gaussian noise + random time shift (roll by up to 2 steps)
                xb = xb + 0.02 * torch.randn_like(xb)
                shift = int(np.random.randint(-2, 3))
                if shift:
                    xb = torch.roll(xb, shift, dims=1)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yt[b].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        self.info = {"epochs": epochs, "train_seconds": round(time.monotonic() - start, 2), "device": self.device,
                     "final_loss": round(float(loss.item()), 4) if len(Xt) else None}
        return self

    def predict_proba(self, X):
        torch, *_ = _torch_modules()
        self.net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 256):
                xb = torch.tensor(X[i:i + 256], dtype=torch.float32, device=self.device)
                out.append(torch.softmax(self.net(xb), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    def save(self, d: Path):
        torch, *_ = _torch_modules()
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), d / "model.pt")
        (d / "net.json").write_text(json.dumps({"feat_dim": self.feat_dim}))

    def load(self, d: Path):
        torch, *_ = _torch_modules()
        self.feat_dim = json.loads((d / "net.json").read_text())["feat_dim"]
        self.net = self._build(self.feat_dim).to(self.device)
        self.net.load_state_dict(torch.load(d / "model.pt", map_location=self.device, weights_only=True))
        self.net.eval()
        return self


def make_model(algorithm: str, classes: list[str], params: dict | None = None, device: str = "cpu"):
    if algorithm in ("random_forest", "xgboost", "svm"):
        return SklearnModel(algorithm, classes, params, "cpu")
    if algorithm in ("lstm", "tcn"):
        return TorchModel(algorithm, classes, params, device)
    raise ValueError(f"Unknown algorithm {algorithm}. Choose from {ALGORITHMS}")
