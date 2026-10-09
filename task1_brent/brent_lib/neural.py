"""Small optional CPU networks. All target dates are checked before fitting."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

QUANTILES = np.arange(1, 20) / 20
STAGE4_HIDDEN = 8


def make_windows(values, as_of, lookback=12, horizon=21):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected finite positive prices")
    if not 0 <= as_of < len(values):
        raise ValueError("Invalid as_of")
    origins = np.arange(lookback - 1, as_of - horizon + 1)
    if len(origins) < 2:
        raise ValueError("Not enough past observations")
    log = np.log(values[: as_of + 1])
    x = np.stack([log[t - lookback + 1 : t + 1] for t in origins])
    # Direct cumulative log-return; last known price is the baseline.
    y = np.stack([log[t + 1 : t + horizon + 1] - log[t] for t in origins])
    return x, y, origins


def chronological_inner_split(origins, horizon, share=0.8):
    boundary = origins[int(len(origins) * share)]
    train = np.flatnonzero(origins + horizon < boundary)
    valid = np.flatnonzero(origins >= boundary)
    if len(train) < 24 or len(valid) < 5:
        raise ValueError("Insufficient purged inner split")
    return train, valid


def correct_quantiles(predictions):
    p = np.asarray(predictions, dtype=float)
    if p.shape[-1] != 19 or not np.isfinite(p).all():
        raise ValueError("Expected 19 finite quantiles")
    shape = p.shape
    iso = IsotonicRegression(increasing=True)
    return np.stack(
        [iso.fit_transform(QUANTILES, row) for row in p.reshape(-1, 19)]
    ).reshape(shape)


def fit_neural(
    values, as_of, kind, *, horizon=21, lookback=12, seeds=(11, 23, 37), epochs=160
):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, origins = make_windows(values, as_of, lookback, horizon)
    itrain, ival = chronological_inner_split(origins, horizon)
    is_quantile = kind == "qrnn"
    nout = horizon * (19 if is_quantile else 1)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = STAGE4_HIDDEN
            if kind == "mlp":
                self.encoder = nn.Sequential(nn.Linear(lookback, hidden), nn.Tanh())
            elif kind in ("rnn", "lstm", "gru"):
                cls = nn.RNN if kind == "rnn" else nn.LSTM if kind == "lstm" else nn.GRU
                self.encoder = cls(1, hidden, batch_first=True)
            elif kind == "cnn":
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, hidden, 3), nn.Tanh(), nn.Flatten()
                )
            elif kind == "qrnn":
                self.encoder = nn.Sequential(nn.Linear(lookback, hidden), nn.Tanh())
            else:
                raise ValueError(kind)
            self.head = nn.Linear(
                hidden * (lookback - 2) if kind == "cnn" else hidden, nout
            )

        def forward(self, z):
            if kind == "mlp":
                z = self.encoder(z)
            elif kind in ("rnn", "lstm", "gru"):
                z = self.encoder(z.unsqueeze(-1))[0][:, -1, :]
            elif kind == "cnn":
                z = self.encoder(z.unsqueeze(1))
            elif kind == "qrnn":
                z = self.encoder(z)
            else:
                raise ValueError(kind)
            result = self.head(z)
            return result.reshape(-1, horizon, 19) if is_quantile else result

    tau = torch.tensor(QUANTILES, dtype=torch.float32)

    def loss(pred, target):
        if is_quantile:
            error = target.unsqueeze(-1) - pred
            return torch.maximum(tau * error, (tau - 1) * error).mean()
        return ((pred - target) ** 2).mean()

    def scaled(indices):
        xm, xs = x[indices].mean(0), np.maximum(x[indices].std(0), 1e-6)
        ym, ys = y[indices].mean(0), np.maximum(y[indices].std(0), 1e-6)
        return xm, xs, ym, ys

    def tensors(params):
        xm, xs, ym, ys = params
        return torch.tensor((x - xm) / xs, dtype=torch.float32), torch.tensor(
            (y - ym) / ys, dtype=torch.float32
        )

    seed_forecasts, chosen_epochs, validation_losses = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        net = Network()
        opt = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.001)
        xx, yy = tensors(scaled(itrain))
        best, best_epoch, stale = float("inf"), 1, 0
        for ep in range(1, epochs + 1):
            net.train()
            opt.zero_grad()
            loss(net(xx[itrain]), yy[itrain]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss(net(xx[ival]), yy[ival]))
            if val < best - 1e-5:
                best, best_epoch, stale = val, ep, 0
            else:
                stale += 1
            if stale >= 20:
                break
        torch.manual_seed(seed)
        net = Network()
        opt = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.001)
        params = scaled(np.arange(len(x)))
        xx, yy = tensors(params)
        for _ in range(best_epoch):
            net.train()
            opt.zero_grad()
            loss(net(xx), yy).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        xm, xs, ym, ys = params
        current = np.log(np.asarray(values)[as_of - lookback + 1 : as_of + 1])
        net.eval()
        with torch.no_grad():
            pred = net(
                torch.tensor(((current - xm) / xs)[None, :], dtype=torch.float32)
            )[0].numpy()
        pred = pred * (ys[:, None] if is_quantile else ys) + (
            ym[:, None] if is_quantile else ym
        )
        seed_forecasts.append(np.asarray(values)[as_of] * np.exp(pred))
        chosen_epochs.append(best_epoch)
        validation_losses.append(best)

    raw = np.asarray(seed_forecasts)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite neural prediction")
    metadata = {
        "model": kind,
        "hidden_size": STAGE4_HIDDEN,
        "layers": 1,
        "parameter_count": int(sum(p.numel() for p in Network().parameters())),
        "epochs": chosen_epochs,
        "selected_epoch": int(np.median(chosen_epochs)),
        "validation_loss": float(np.median(validation_losses)),
        "seeds": list(seeds),
        "training_pairs": len(x),
        "max_training_target": int(origins[-1] + horizon),
        "inner_train_max_target": int(origins[itrain[-1]] + horizon),
        "inner_validation_first_origin": int(origins[ival[0]]),
    }
    if is_quantile:
        metadata["crossing_fraction_raw"] = float(
            (np.diff(raw, axis=-1) < 0).any(axis=-1).mean()
        )
        raw = np.asarray([correct_quantiles(p) for p in raw])
    return raw.mean(0), raw, metadata
