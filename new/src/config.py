from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    seq_len: int = 30

    # Reversal label generation
    trend_window: int = 20            # days looked back for the move that came before the turning point
    lookahead: int = 10               # days looked forward for the move after the turning point
    swing_window: int = 5             # a turning point is the lowest / highest close of +- swing_window days
    reversal_threshold: Optional[float] = None   # fixed swing size (0.025 = 2.5%); None = per-stock automatic
    threshold_k: float = 0.75         # automatic swing size = k * daily volatility * sqrt(lookahead)
    target_window: int = 2            # days before the turning point that are also labelled 1

    # Reversal-aware loss
    lambda_: float = 2.0
    sigma: float = 3.0

    # Model
    indicator_hidden: int = 64
    indicator_out: int = 32
    tcn_channels: int = 64
    fusion_hidden: int = 64
    dropout: float = 0.20

    # Data split (by calendar date, same cut dates for every stock)
    train_frac: float = 0.70
    val_frac: float = 0.15

    # Training
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    seed: int = 42

    # Decision threshold, picked on validation data
    max_false_alarm: float = 0.30

    # stocks with less training history than this are skipped (their swing size cannot be estimated)
    min_train_rows: int = 250
