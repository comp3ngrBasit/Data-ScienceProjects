"""Next-action predictors: Markov baseline, XGBoost, GRU / LSTM and Transformer sequence models.

All share one interface: ``fit(train, fb, calib)`` -> ``predict_proba(examples)`` returning an (n, C)
array over ``fb.classes`` that sums to 1 per row. Fighter statistics used as features are always computed
without the example's own fight (``exclude_fight=e.fight_key``), so a fight never informs its own features.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar

from app.predictions.dataset import RARE, Example, map_target
from app.predictions.features import FeatureBuilder

ARCHITECTURES = ("markov", "xgboost", "gru", "lstm", "transformer")
EPS = 1e-6


def _norm(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, EPS, None)
    return P / P.sum(axis=1, keepdims=True)


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    if abs(T - 1.0) < 1e-9:
        return _norm(P)
    logit = np.log(np.clip(P, EPS, None)) / T
    logit -= logit.max(axis=1, keepdims=True)
    E = np.exp(logit)
    return E / E.sum(axis=1, keepdims=True)


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: the T minimising negative log-likelihood on held-out predictions."""
    if len(y) < 5:
        return 1.0

    def nll(logT):
        Q = apply_temperature(P, math.exp(logT))
        return -float(np.mean(np.log(Q[np.arange(len(y)), y])))

    r = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20)), method="bounded")
    return float(round(math.exp(r.x), 4))


class Predictor:
    arch = "base"

    def __init__(self, params: dict | None = None, device: str = "cpu"):
        self.params = params or {}
        self.device = device
        self.temperature = 1.0
        self.fb: FeatureBuilder | None = None
        self.info: dict = {}

    def y(self, examples: list[Example]) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.fb.classes)}
        return np.array([idx[map_target(e.target, self.fb.classes)] for e in examples])

    def fit(self, train: list[Example], fb: FeatureBuilder, calib: list[Example] | None = None):
        raise NotImplementedError

    def raw_proba(self, examples: list[Example]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, examples: list[Example]) -> np.ndarray:
        return apply_temperature(self.raw_proba(examples), self.temperature)

    def calibrate(self, calib: list[Example]):
        if calib:
            self.temperature = fit_temperature(self.raw_proba(calib), self.y(calib))

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"arch": self.arch, "params": self.params, "temperature": self.temperature,
                                                  "info": self.info, "features": self.fb.state()}, indent=1))

    def load_meta(self, d: Path) -> dict:
        m = json.loads((d / "meta.json").read_text())
        self.params, self.temperature, self.info = m["params"], m["temperature"], m["info"]
        return m


# ============================================================================ Markov baseline
class MarkovPredictor(Predictor):
    """Interpolated (Witten-Bell) back-off Markov model of orders 0..K, global then fighter-specific.

    Contexts seen fewer than ``min_count`` times get no weight (minimum sample threshold); order 0 uses
    additive smoothing, so every class keeps a non-zero probability.
    """
    arch = "markov"

    def fit(self, train, fb, calib=None):
        self.fb = fb
        self.K = int(self.params.get("max_order", 3))
        self.min_count = int(self.params.get("min_count", 2))
        self.alpha = float(self.params.get("alpha", 0.5))
        self.glob: dict[tuple, Counter] = defaultdict(Counter)
        self.fight: dict[tuple, Counter] = defaultdict(Counter)
        for e in train:
            t = map_target(e.target, fb.classes)
            cats = [s.category for s in e.history]
            for k in range(0, self.K + 1):
                if k > len(cats):
                    break
                ctx = tuple(cats[len(cats) - k:]) if k else ()
                self.glob[ctx][t] += 1
                self.fight[(e.fighter_key,) + ctx][t] += 1
        self.info = {"contexts": len(self.glob), "fighter_contexts": len(self.fight), "train_examples": len(train)}
        if calib:
            self.calibrate(calib)
        return self

    def _wb(self, counts: Counter, base: np.ndarray) -> np.ndarray:
        n = sum(counts.values())
        if n < self.min_count:
            return base
        types = len(counts)
        lam = n / (n + types)
        ml = np.array([counts.get(c, 0) / n for c in self.fb.classes])
        return lam * ml + (1 - lam) * base

    def dist(self, fighter: str, cats: list[str]) -> np.ndarray:
        C = len(self.fb.classes)
        uni = self.glob.get((), Counter())
        n = sum(uni.values())
        P = np.array([(uni.get(c, 0) + self.alpha) / (n + self.alpha * C) for c in self.fb.classes])
        for k in range(1, self.K + 1):
            if k > len(cats):
                break
            P = self._wb(self.glob.get(tuple(cats[-k:]), Counter()), P)
        for k in range(0, self.K + 1):  # fighter-specific refinement, from general to specific
            if k > len(cats):
                break
            ctx = (fighter,) + (tuple(cats[-k:]) if k else ())
            P = self._wb(self.fight.get(ctx, Counter()), P)
        return P

    def raw_proba(self, examples):
        return _norm(np.stack([self.dist(e.fighter_key, [s.category for s in e.history]) for e in examples]))

    def support(self, fighter: str, cats: list[str]) -> dict:
        out = {}
        for k in range(1, self.K + 1):
            if k > len(cats):
                break
            ctx = tuple(cats[-k:])
            out[k] = {"global": sum(self.glob.get(ctx, Counter()).values()),
                      "fighter": sum(self.fight.get((fighter,) + ctx, Counter()).values())}
        return out

    def save(self, d):
        super().save(d)
        joblib.dump({"K": self.K, "min_count": self.min_count, "alpha": self.alpha,
                     "glob": {k: dict(v) for k, v in self.glob.items()}, "fight": {k: dict(v) for k, v in self.fight.items()}},
                    d / "markov.joblib")

    def load(self, d, fb):
        self.load_meta(d)
        self.fb = fb
        st = joblib.load(d / "markov.joblib")
        self.K, self.min_count, self.alpha = st["K"], st["min_count"], st["alpha"]
        self.glob = defaultdict(Counter, {k: Counter(v) for k, v in st["glob"].items()})
        self.fight = defaultdict(Counter, {k: Counter(v) for k, v in st["fight"].items()})
        return self


# ============================================================================ XGBoost
def class_weights(y: np.ndarray, C: int) -> np.ndarray:
    """Square-root inverse-frequency weights: helps rare classes without letting them dominate."""
    cnt = np.bincount(y, minlength=C).astype(float)
    w = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    w = w / (w[y].mean() if len(y) else 1.0)
    return w


class XGBPredictor(Predictor):
    arch = "xgboost"

    def _X(self, examples):
        return np.stack([self.fb.tabular(e, exclude_fight=e.fight_key) for e in examples])

    def fit(self, train, fb, calib=None):
        from xgboost import XGBClassifier

        self.fb = fb
        X, y = self._X(train), self.y(train)
        self.present = sorted(set(y.tolist()))
        remap = {c: i for i, c in enumerate(self.present)}
        yy = np.array([remap[v] for v in y])
        w = class_weights(y, len(fb.classes))[y]
        p = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8,
             "min_child_weight": 1.0, "reg_lambda": 1.0, **self.params}
        t0 = time.monotonic()
        if len(self.present) == 1:
            self.model = None
        else:
            self.model = XGBClassifier(objective="multi:softprob", num_class=len(self.present), tree_method="hist",
                                       device="cuda" if self.device.startswith("cuda") else "cpu",
                                       eval_metric="mlogloss", random_state=0, n_jobs=0, **p)
            self.model.fit(X, yy, sample_weight=w)
        self.info = {"train_examples": len(train), "features": X.shape[1], "train_seconds": round(time.monotonic() - t0, 2),
                     "params": p}
        if calib:
            self.calibrate(calib)
        return self

    def raw_proba(self, examples):
        C = len(self.fb.classes)
        P = np.full((len(examples), C), EPS)
        if self.model is None:
            P[:, self.present[0]] = 1.0
        else:
            P[:, self.present] = self.model.predict_proba(self._X(examples))
        return _norm(P)

    def importance(self, top: int = 15) -> list[dict]:
        if self.model is None:
            return []
        names = self.fb.tabular_names()
        imp = self.model.get_booster().get_score(importance_type="gain")
        rows = sorted(((names[int(k[1:])], v) for k, v in imp.items()), key=lambda r: -r[1])[:top]
        tot = sum(v for _, v in rows) or 1.0
        return [{"feature": n, "gain_share": round(v / tot, 4)} for n, v in rows]

    def save(self, d):
        super().save(d)
        joblib.dump({"model": self.model, "present": self.present}, d / "xgb.joblib")

    def load(self, d, fb):
        self.load_meta(d)
        self.fb = fb
        st = joblib.load(d / "xgb.joblib")
        self.model, self.present = st["model"], st["present"]
        return self


# ============================================================================ sequence networks
def _nets():
    import torch
    from torch import nn

    class SeqNet(nn.Module):
        def __init__(self, kind: str, vocab: int, static_dim: int, C: int, emb=16, hidden=48, layers=1, dropout=0.2, heads=2):
            super().__init__()
            self.kind = kind
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            step_dim = emb + 4
            if kind == "gru":
                self.rnn = nn.GRU(step_dim, hidden, num_layers=layers, batch_first=True)
            elif kind == "lstm":
                self.rnn = nn.LSTM(step_dim, hidden, num_layers=layers, batch_first=True)
            else:
                self.inp = nn.Linear(step_dim, hidden)
                self.pos = nn.Embedding(64, hidden)  # learned positional encoding
                enc = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True)
                self.tf = nn.TransformerEncoder(enc, num_layers=max(1, layers + 1))
            self.static = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + 32, C))

        def forward(self, ids, num, mask, static):
            x = torch.cat([self.emb(ids), num], dim=-1)
            lengths = mask.sum(1).long().clamp(min=1)
            if self.kind in ("gru", "lstm"):
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, h = self.rnn(packed)
                if self.kind == "lstm":
                    h = h[0]
                z = h[-1]
            else:
                pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
                hseq = self.tf(self.inp(x) + self.pos(pos), src_key_padding_mask=mask < 0.5)
                z = hseq[torch.arange(ids.shape[0], device=ids.device), lengths - 1]  # last real step
            return self.head(torch.cat([z, self.static(static)], dim=-1))

    return torch, nn, SeqNet


class SeqPredictor(Predictor):
    def __init__(self, arch: str, params=None, device="cpu"):
        super().__init__(params, device)
        self.arch = arch

    def _tensors(self, examples):
        torch, _, _ = _nets()
        L = int(self.params.get("seq_len", 10))
        ids, num, mask = zip(*(self.fb.steps(e, L) for e in examples), strict=False)
        st = np.stack([self.fb.static(self.fb.static_raw(e, e.fight_key)) for e in examples])
        dev = self.device
        return (torch.tensor(np.stack(ids), device=dev), torch.tensor(np.stack(num), device=dev),
                torch.tensor(np.stack(mask), device=dev), torch.tensor(st, device=dev))

    def fit(self, train, fb, calib=None):
        torch, nn, SeqNet = _nets()
        torch.manual_seed(int(self.params.get("seed", 0)))
        self.fb = fb
        if fb.mean is None:
            fb.fit_normaliser(np.stack([fb.static_raw(e, e.fight_key) for e in train]))
        C = len(fb.classes)
        p = {"hidden": 48, "layers": 1, "dropout": 0.2, "epochs": 120, "lr": 3e-3, "batch": 64, "patience": 15, **self.params}
        self.params = p
        ids, num, mask, st = self._tensors(train)
        y = torch.tensor(self.y(train), device=self.device)
        self.static_dim = st.shape[1]
        self.net = SeqNet(self.arch, len(fb.vocab), self.static_dim, C, hidden=p["hidden"], layers=p["layers"], dropout=p["dropout"]).to(self.device)
        w = torch.tensor(class_weights(self.y(train), C), dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=1e-3)
        cal = self._tensors(calib) + (torch.tensor(self.y(calib), device=self.device),) if calib else None
        best, best_state, bad, t0, epochs = float("inf"), None, 0, time.monotonic(), 0
        g = torch.Generator().manual_seed(0)
        for ep in range(p["epochs"]):
            self.net.train()
            perm = torch.randperm(len(y), generator=g).to(self.device)
            for i in range(0, len(y), p["batch"]):
                b = perm[i:i + p["batch"]]
                opt.zero_grad()
                loss = loss_fn(self.net(ids[b], num[b], mask[b], st[b]), y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            epochs = ep + 1
            if cal is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = float(nn.functional.cross_entropy(self.net(*cal[:4]), cal[4]))
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.info = {"train_examples": len(train), "epochs": epochs, "early_stopped": cal is not None,
                     "train_seconds": round(time.monotonic() - t0, 2), "device": self.device,
                     "parameters": int(sum(x.numel() for x in self.net.parameters()))}
        if calib:
            self.calibrate(calib)
        return self

    def raw_proba(self, examples):
        torch, _, _ = _nets()
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(examples), 512):
                t = self._tensors(examples[i:i + 512])
                out.append(torch.softmax(self.net(*t), dim=-1).cpu().numpy())
        return _norm(np.concatenate(out)) if out else np.zeros((0, len(self.fb.classes)))

    def save(self, d):
        torch, _, _ = _nets()
        super().save(d)
        (d / "net.json").write_text(json.dumps({"static_dim": self.static_dim}))
        torch.save(self.net.state_dict(), d / "net.pt")

    def load(self, d, fb):
        torch, _, SeqNet = _nets()
        self.load_meta(d)
        self.fb = fb
        self.static_dim = json.loads((d / "net.json").read_text())["static_dim"]
        p = self.params
        self.net = SeqNet(self.arch, len(fb.vocab), self.static_dim, len(fb.classes), hidden=p["hidden"], layers=p["layers"],
                          dropout=p["dropout"]).to(self.device)
        self.net.load_state_dict(torch.load(d / "net.pt", map_location=self.device, weights_only=True))
        self.net.eval()
        return self


def make_predictor(arch: str, params: dict | None = None, device: str = "cpu") -> Predictor:
    if arch == "markov":
        return MarkovPredictor(params, device)
    if arch == "xgboost":
        return XGBPredictor(params, device)
    if arch in ("gru", "lstm", "transformer"):
        return SeqPredictor(arch, params, device)
    raise ValueError(f"Unknown architecture {arch}")


def load_predictor(d: Path, fb: FeatureBuilder, device: str = "cpu") -> Predictor:
    meta = json.loads((d / "meta.json").read_text())
    return make_predictor(meta["arch"], meta["params"], device).load(d, fb)


__all__ = ["ARCHITECTURES", "RARE", "MarkovPredictor", "XGBPredictor", "SeqPredictor", "make_predictor", "load_predictor",
           "apply_temperature", "fit_temperature"]
