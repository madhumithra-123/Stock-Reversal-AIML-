# Reversal-aware stock trend reversal prediction (PyTorch, terminal only)

TCN (price sequence) + MLP (technical indicators) with a fused reversal head, trained with the
reversal-aware loss

    L_total = (1 + lambda * R_t) * BCE(y_t, p_t),    R_t = exp(-(t - t_rev)^2 / (2 sigma^2))

where `t_rev` is the nearest reversal point. A plain BCE model (`L = BCE`, the same as lambda = 0) is trained
alongside it as the baseline, with the same data, seed, model and threshold rule, and the two are compared.

See `CODE_GUIDE.md` for an explanation of every file and every function, and for the list of changes
made to the first version.

## Install
    python -m venv .venv
    .venv\Scripts\activate          # Windows
    pip install -r requirements.txt

## Data

    git clone https://github.com/yumoxu/stocknet-dataset.git data/stocknet  

## Run
    python -m src.train --data_dir data/stocknet/price/raw --epochs 30          # trains BOTH losses and prints a comparison table
    python -m src.train --data_dir data/stocknet/price/raw --stocks GOOG --epochs 30     #for single stock
    python -m src.train --data_dir data/stocknet/price/raw --loss bce             # plain BCE only
    python -m src.train --data_dir data/stocknet/price/raw --loss reversal_aware  # reversal-aware loss only
    python -m src.train --data_dir data/stocknet/price/raw --epochs 20 --stocks AAPL,MSFT,GOOG
    python -m src.train --data_dir data/stocknet/price/raw --lambda_ 2.0 --sigma 3.0


Quick test without StockNet: `python scripts/make_demo_data.py` then `python -m src.train --data_dir data/demo --epochs 3`.

## Output files (in `outputs/`)
| File | Content |
|---|---|
| `reversal_aware_model.pt`, `test_predictions.csv`, `metrics.json` | reversal-aware model |
| `bce_model.pt`, `bce_test_predictions.csv`, `bce_metrics.json` | plain BCE model |
| `comparison.json` | both metric sets and the number of metrics each loss wins |

## Definitions
- **Reversal point:** the lowest (bottom) or highest (top) close of +-5 days, where the price had moved
  the other way by at least a swing size over the previous 20 days and moves back by at least the same
  size over the next 10 days. Swing size = 0.75 x daily volatility x sqrt(10) per stock, from training data.
- **Label:** `y_t = 1` if a reversal point happens today or within the next 2 days.
- **Split:** 70 / 15 / 15 by calendar date, the same cut dates for every stock. Rows near each cut are
  dropped because their labels look ahead. Scaler fitted on training rows only.
- **Decision threshold:** chosen on validation data (best F1 with false-alarm rate <= 30%), then frozen.
- **Early stopping:** best validation PR-AUC.

## note
- Accuracy is not meaningful when reversals are rare, report PR-AUC, precision, recall and false alarms.
- Labels use future prices, inputs never do.
- The comparison is one run with one seed. Repeat with `--seed 1`, `--seed 2`, ... and compare averages before claiming that one loss is better.
