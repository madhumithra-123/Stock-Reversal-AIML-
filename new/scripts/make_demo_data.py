"""Writes three synthetic OHLCV csv files to data/demo, only to check that the pipeline runs."""
from pathlib import Path

import numpy as np
import pandas as pd

out = Path("data/demo")
out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
dates = pd.bdate_range("2012-01-02", periods=1500)

for name in ["DEMO1", "DEMO2", "DEMO3"]:
    n = len(dates)
    drift = np.repeat(rng.choice([-1, 0, 1], size=n // 40 + 1), 40)[:n] * 0.0008
    wave = 0.004 * np.sin(np.arange(n) / rng.uniform(7, 15))
    ret = drift + wave + rng.normal(0, 0.014, n)
    close = 50 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Adj Close": close, "Volume": volume,
    }).to_csv(out / f"{name}.csv", index=False)

print(f"wrote 3 files to {out}/")
