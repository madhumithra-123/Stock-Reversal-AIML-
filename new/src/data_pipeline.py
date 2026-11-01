from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Channels of the sequence that goes into the TCN. All of them are relative to the previous close,
# so a 10 dollar stock and a 1000 dollar stock look alike.
PRICE_COLUMNS = ["open_rel", "high_rel", "low_rel", "close_rel", "volume_rel"]

# Indicator vector that goes into the MLP (names as in the block diagram). Everything that is a
# price level (SMA, EMA, MACD, ATR, Bollinger bands) is divided by the close.
INDICATOR_COLUMNS = [
    "sma_5", "sma_20", "ema_12", "ema_26",
    "macd", "macd_signal", "rsi_14", "atr_14",
    "bb_mid", "bb_upper", "bb_lower", "bb_width",
    "volume_change",
]


# ----------------------------------------------------------------------------- loading

def _find_column(columns, candidates):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    cols = {
        "date": _find_column(df.columns, ["date", "datetime", "timestamp"]),
        "open": _find_column(df.columns, ["open"]),
        "high": _find_column(df.columns, ["high"]),
        "low": _find_column(df.columns, ["low"]),
        "close": _find_column(df.columns, ["close"]),
        "volume": _find_column(df.columns, ["volume"]),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError(f"{path}: missing OHLCV columns: {missing}")
    adj_col = _find_column(df.columns, ["adj_close", "adjusted_close", "adjust_close"])

    out = pd.DataFrame({
        "date": pd.to_datetime(df[cols["date"]], errors="coerce"),
        "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
        "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
        "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        "volume": pd.to_numeric(df[cols["volume"]], errors="coerce").fillna(0.0),
    })
    if adj_col is not None:
        out["adj"] = pd.to_numeric(df[adj_col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]

    # dividends / splits would otherwise look like sudden price jumps
    if "adj" in out.columns:
        factor = (out["adj"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for c in ["open", "high", "low", "close"]:
            out[c] = out[c] * factor
        out = out.drop(columns="adj")

    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------- features

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, high, low, volume = x["close"], x["high"], x["low"], x["volume"]
    prev_close = close.shift(1)

    # sequence channels
    x["open_rel"] = x["open"] / prev_close - 1
    x["high_rel"] = x["high"] / prev_close - 1
    x["low_rel"] = x["low"] / prev_close - 1
    x["close_rel"] = close / prev_close - 1
    x["volume_rel"] = np.log((volume + 1) / (volume.rolling(20).mean() + 1))

    # indicators, all computed from past data only
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    bb_std = close.rolling(20).std()
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std

    x["sma_5"] = sma_5 / close - 1
    x["sma_20"] = sma_20 / close - 1
    x["ema_12"] = ema_12 / close - 1
    x["ema_26"] = ema_26 / close - 1
    x["macd"] = macd / close
    x["macd_signal"] = macd_signal / close
    x["rsi_14"] = rsi / 100.0
    x["atr_14"] = atr / close
    x["bb_mid"] = sma_20 / close - 1
    x["bb_upper"] = bb_upper / close - 1
    x["bb_lower"] = bb_lower / close - 1
    x["bb_width"] = (bb_upper - bb_lower) / sma_20.replace(0, np.nan)
    x["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).clip(-1, 3)

    x = x.replace([np.inf, -np.inf], np.nan)
    keep = ["date", "open", "high", "low", "close", "volume"] + PRICE_COLUMNS + INDICATOR_COLUMNS
    return x[keep].dropna().reset_index(drop=True)


def build_stock_dataframe(path: Path, cfg) -> pd.DataFrame:
    df = add_indicators(load_ohlcv_csv(path))
    if len(df) < cfg.seq_len + cfg.trend_window + cfg.lookahead + 100:
        raise ValueError(f"{path.name}: not enough usable rows after preprocessing.")
    return df


# ----------------------------------------------------------------------------- labels

def find_reversal_events(close: np.ndarray, cfg, threshold: float):
    """
    A reversal point is a turning point of the trend:
      bottom: the lowest close of +- swing_window days, the price had FALLEN by at least `threshold`
              from its high of the previous trend_window days, and RISES by at least `threshold`
              within the next `lookahead` days;
      top   : the mirror image.
    Returns the indices of the reversal points and a mask of days where the rule could be checked
    (it needs enough history before and enough days after).
    """
    c = pd.Series(close)
    w, H, B = cfg.swing_window, cfg.lookahead, cfg.trend_window

    lo = c.rolling(2 * w + 1, center=True).min()
    hi = c.rolling(2 * w + 1, center=True).max()

    nxt = c.shift(-1)
    fwd_max = nxt[::-1].rolling(H, min_periods=H).max()[::-1]
    fwd_min = nxt[::-1].rolling(H, min_periods=H).min()[::-1]
    prv = c.shift(1)
    back_max = prv.rolling(B, min_periods=B).max()
    back_min = prv.rolling(B, min_periods=B).min()

    bottom = (c == lo) & (back_max / c - 1 >= threshold) & (fwd_max / c - 1 >= threshold)
    top = (c == hi) & (1 - back_min / c >= threshold) & (1 - fwd_min / c >= threshold)

    known = (lo.notna() & fwd_max.notna() & back_max.notna()).to_numpy()
    events = np.flatnonzero((bottom | top).to_numpy() & known)
    return events, known


def make_labels(n: int, events: np.ndarray, known: np.ndarray, target_window: int):
    """
    y_t = 1 when a reversal point happens today or within the next `target_window` days,
    so the model is asked to see a reversal coming, not to notice one that already happened.
    label_ok marks the days whose label is fully known.
    """
    y = np.zeros(n, dtype=np.float32)
    for e in events:
        y[max(0, e - target_window): e + 1] = 1.0
    label_ok = pd.Series(known[::-1].astype(float)).rolling(target_window + 1).min().to_numpy()[::-1] == 1
    return y, label_ok


def distance_to_events(n: int, events: np.ndarray, cap: int = 999) -> np.ndarray:
    """|t - t_rev| in trading days for the nearest reversal point"""
    if len(events) == 0:
        return np.full(n, cap, dtype=np.float32)
    idx = np.arange(n)
    pos = np.searchsorted(events, idx)
    left = events[np.clip(pos - 1, 0, len(events) - 1)]
    right = events[np.clip(pos, 0, len(events) - 1)]
    return np.minimum(np.minimum(np.abs(idx - left), np.abs(idx - right)), cap).astype(np.float32)


# ----------------------------------------------------------------------------- samples

def create_split_samples(frames: Dict[str, pd.DataFrame], cfg):
    """
    1. Cut dates: 70 / 15 / 15 of the combined calendar, the same for every stock.
    2. Scaler is fitted on training rows only.
    3. Labels and distances are computed on each stock's full history (they look into the future,
       that is allowed for a target), then the last rows before each cut are dropped so that no
       training target depends on prices from the next period.
    """
    all_dates = sorted(set().union(*[set(f["date"]) for f in frames.values()]))
    n_dates = len(all_dates)
    train_end = all_dates[int(n_dates * cfg.train_frac) - 1]
    val_end = all_dates[int(n_dates * (cfg.train_frac + cfg.val_frac)) - 1]

    feature_cols = PRICE_COLUMNS + INDICATOR_COLUMNS
    train_matrix = np.vstack([
        f.loc[f["date"] <= train_end, feature_cols].to_numpy(dtype=np.float64)
        for f in frames.values()
    ])
    scaler = StandardScaler().fit(train_matrix)

    purge = cfg.lookahead + cfg.swing_window + cfg.target_window + int(np.ceil(3 * cfg.sigma))
    n_price = len(PRICE_COLUMNS)

    data = {"train": [], "val": [], "test": []}
    true_events = {"train": {}, "val": {}, "test": {}}
    thresholds = {}

    for symbol, frame in frames.items():
        dates = frame["date"].to_numpy()
        close = frame["close"].to_numpy(dtype=np.float64)
        n = len(frame)

        n_train_rows = int((dates <= np.datetime64(train_end)).sum())
        if n_train_rows < cfg.min_train_rows:
            print(f"[SKIP] {symbol}: only {n_train_rows} training rows")
            continue

        if cfg.reversal_threshold is not None:
            thr = float(cfg.reversal_threshold)
        else:
            vol = pd.Series(close)[dates <= np.datetime64(train_end)].pct_change().std()
            thr = float(np.clip(cfg.threshold_k * vol * np.sqrt(cfg.lookahead), 0.01, 0.15))
        thresholds[symbol] = thr

        events, known = find_reversal_events(close, cfg, thr)
        y, label_ok = make_labels(n, events, known, cfg.target_window)
        dist = distance_to_events(n, events)
        r = np.exp(-(dist ** 2) / (2.0 * cfg.sigma ** 2)).astype(np.float32)

        scaled = scaler.transform(frame[feature_cols].to_numpy(dtype=np.float64)).astype(np.float32)
        price_s, ind_s = scaled[:, :n_price], scaled[:, n_price:]

        idx_train = np.flatnonzero(dates <= np.datetime64(train_end))
        idx_val = np.flatnonzero((dates > np.datetime64(train_end)) & (dates <= np.datetime64(val_end)))
        idx_test = np.flatnonzero(dates > np.datetime64(val_end))
        splits = {
            "train": idx_train[:-purge] if len(idx_train) > purge else idx_train[:0],
            "val": idx_val[:-purge] if len(idx_val) > purge else idx_val[:0],
            "test": idx_test,
        }

        for name, rows in splits.items():
            rows = rows[(rows >= cfg.seq_len - 1) & label_ok[rows]]
            if len(rows) == 0:
                continue
            xp = np.stack([price_s[t - cfg.seq_len + 1: t + 1] for t in rows])
            xi = ind_s[rows]
            data[name].append((symbol, xp, xi, y[rows], r[rows], dates[rows]))
            in_range = events[(events >= rows.min()) & (events <= rows.max())]
            true_events[name][symbol] = dates[in_range]

    return data, scaler, true_events, thresholds, (train_end, val_end)


def merge_split(parts):
    if not parts:
        raise RuntimeError("No usable samples were created.")
    xp = np.concatenate([p[1] for p in parts], axis=0)
    xi = np.concatenate([p[2] for p in parts], axis=0)
    y = np.concatenate([p[3] for p in parts], axis=0)
    r = np.concatenate([p[4] for p in parts], axis=0)
    dates = np.concatenate([p[5] for p in parts], axis=0)
    symbols = np.concatenate([np.full(len(p[3]), p[0], dtype=object) for p in parts])
    return xp, xi, y, r, dates, symbols
