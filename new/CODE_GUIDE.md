# Code guide

## 1. How the program flows

```
price csv files
   -> load_ohlcv_csv         clean one stock, adjust for splits / dividends
   -> add_indicators         sequence channels + indicator vector (all relative to the close)
   -> find_reversal_events   where are the turning points?  (uses the future, for labels only)
   -> make_labels            y_t = 1 if a turning point is today or within 2 days
   -> distance_to_events     |t - t_rev| for every day  ->  R_t = exp(-d^2 / (2 sigma^2))
   -> create_split_samples   70/15/15 split by date, scaler on train, 30-day windows
   -> ReversalAwareModel     TCN(window) + MLP(indicators) -> fusion -> one logit
   -> ReversalAwareLoss      (1 + lambda * R_t) * BCE
   -> train / validate       best epoch = best validation PR-AUC
   -> select_threshold       decision threshold from validation data
   -> test metrics + predicted reversal dates
```

## 2. File by file

### src/config.py
One dataclass `Config` with every setting.
- `seq_len` days in each input window (30).
- `trend_window` days looked back to find the move before a turning point (20).
- `lookahead` days looked forward to find the move after it (10).
- `swing_window` a turning point is the lowest / highest close of +-5 days.
- `reversal_threshold` fixed swing size. `None` means automatic per stock.
- `threshold_k` automatic swing size = k x daily volatility x sqrt(lookahead).
- `target_window` days before a turning point that are also labelled 1 (2).
- `lambda_`, `sigma` the two numbers of the reversal-aware loss.
- model sizes, `dropout`, training settings, `train_frac` / `val_frac`, `max_false_alarm`.

### src/data_pipeline.py
- `PRICE_COLUMNS`: the 5 channels fed to the TCN: open, high, low, close relative to the previous close,
  and log(volume / 20-day average volume).
- `INDICATOR_COLUMNS`: the 13 indicators fed to the MLP.
- `_find_column`: finds a column such as "Adj Close" whatever the capitalisation.
- `load_ohlcv_csv`: reads one csv, converts to numbers, drops bad rows, sorts by date, and multiplies
  open / high / low / close by adj_close / close so dividends do not look like jumps.
- `add_indicators`: computes the channels and indicators using only past data. SMA, EMA, MACD, ATR and the
  Bollinger bands are divided by the close, RSI is divided by 100, volume change is clipped to [-1, 3].
  Then rows with missing values (the first ~26 days) are dropped.
- `build_stock_dataframe`: the two functions above plus a minimum-length check.
- `find_reversal_events`: the labelling rule. A bottom is the lowest close of +-5 days, after the price
  fell by at least the swing size from its recent high and before it rises by at least the swing size.
  A top is the mirror image. Returns the day numbers of the turning points and a mask of days where the rule
  could be checked (not near the ends of the series).
- `make_labels`: `y_t = 1` for the turning point day and the `target_window` days before it. Also returns
  `label_ok`, which is False when the future needed for the label is not available.
- `distance_to_events`: for each day the number of trading days to the nearest turning point.
- `create_split_samples`: the main assembly function.
  1. Cut dates at 70% and 85% of the combined calendar.
  2. StandardScaler fitted on training rows of all stocks.
  3. Per stock: swing size, events, labels, distances, R_t.
  4. Rows of each split; the last rows before each cut are dropped (purge) because their labels and R_t
     look into the next period.
  5. Each sample = the last 30 scaled days (for the TCN) + the scaled indicator vector of that day (for the MLP).
- `merge_split`: stacks the per-stock samples into arrays.

### src/model.py (unchanged from your version)
- `Chomp1d`: cuts the extra padding at the end so a convolution never sees the future.
- `TemporalBlock`: two causal dilated convolutions with batch norm, ReLU, dropout and a residual link.
- `TemporalEncoder`: four blocks with dilation 1, 2, 4, 8, reads the last time step.
- `IndicatorEncoder`: two-layer MLP for the indicator vector.
- `ReversalAwareModel`: concatenates both encodings, one fusion layer, one output logit.

### src/losses.py
- `ReversalAwareLoss`: `mean((1 + lambda * R_t) * BCE)`. Days close to a turning point count up to
  `1 + lambda` times, whether the label there is 1 or 0.

- `BCELoss`: plain `mean(BCE)`, the baseline. Same call signature, ignores `R_t`. Equal to `ReversalAwareLoss(lambda_=0)`.
- `build_criterion(name, lambda_)`: returns the loss for `"reversal_aware"` or `"bce"`.

### src/metrics.py
- `select_threshold`: on validation data, the threshold with the best F1 among those whose false-alarm rate
  is at most 30%.
- `classification_metrics`: accuracy, precision, recall, F1, PR-AUC, false-alarm rate, confusion counts and
  the no-skill PR-AUC (share of positive days).
- `predicted_reversal_events`: turns runs of consecutive alert days into one predicted date (the most
  probable day of the run).
- `event_matching`: a predicted date is correct if it is within 3 days of a real turning point, each real
  turning point can be matched once.

### src/train.py
`main()` reads the options (`--loss both|reversal_aware|bce`, default `both`), builds the data, trains with AdamW and gradient clipping, keeps the epoch with the
best validation PR-AUC, picks the decision threshold on validation data, evaluates once on the test period,
prints the metrics and predicted dates, and saves the model (with scaler and threshold), the predictions and
metrics.json. `train_one` trains one model with one loss (same seed and batch order for both) and saves its files; `print_comparison` prints the side-by-side test table and writes comparison.json. Helpers: `make_loader` (drops a last batch of one sample while training, batch norm cannot
handle it), `run_epoch`, `predict`, `discover_csvs`, `prepare_stocks`.

## 3. What was changed from the first version, and why (12 changes)

| # | Problem in the first version | Change | Why it matters |
|---|---|---|---|
| 1 | The label marked a "reversal" for every day followed by a 2.5% opposite move within 10 days, and the 5 labelled days included days after the move. About 50% of training days were positive and many positives were already visible in the inputs. | Reversal = real turning point (lowest / highest of +-5 days with a swing before and after). Positive days are the turning point and the 2 days before it. | Reversals become rare (about 9 to 19% of days) and the model has to anticipate, not notice. |
| 2 | Raw prices (open, high, low, close, volume levels) were scaled across all stocks together. | All inputs are relative to the previous close; indicators are divided by the close. | A 10 dollar and a 1000 dollar stock are comparable; prices outside the training range no longer break the test period. |
| 3 | Each stock was split 70/15/15 by its own row count. | Split by calendar date, same cut for all stocks. | Late-listed stocks no longer put test-period days in the training set of other stocks. |
| 4 | The last 10 days of every split were given label 0 although the label could not be checked. | Those days are excluded, and rows near each cut are purged. | No false negatives, no label leakage across the cut. |
| 5 | Decision threshold fixed at 0.5. The saved run alarmed on 96% of normal days (recall 1.0). | Threshold chosen on validation data with a false-alarm cap. | Metrics now describe a usable alert level. |
| 6 | Early stopping on the validation loss. | Early stopping on validation PR-AUC. | The best epoch no longer depends on how the loss is weighted. |
| 7 | Possible crash when a training batch had one sample (batch norm). | `drop_last` for training. | Avoids a random failure. |
| 8 | Proximity R_t was computed in three places, two of them wrong or unused (sigma fixed to 1, calendar days). A slow per-sample date search. | One vectorised trading-day computation using the `sigma` option. | Correct and fast. |
| 9 | The README used `scripts/make_demo_data.py`, which did not exist. | Script added. | The quick test now works. |
| 10 | Checkpoint had no scaler or threshold. | Both are saved. | The model can be reused for new data. |
| 11 | "Timing accuracy" was really recall of reversal points. | Reported as event recall and event precision within +-3 days. | Honest names. |
| 12 | Prices not adjusted for dividends. | Adj Close used when present. | Dividends are not mistaken for drops. |

## 4. BCE comparison (added)
The program now also trains a plain BCE model and compares it with the reversal-aware model on the test period. Both use the same data, split, scaler, architecture, seed, early stopping (validation PR-AUC) and threshold rule, so only the loss differs. The table shows accuracy, precision, recall, F1, PR-AUC, false-alarm rate and event recall/precision, with the better loss per metric.

Not changed: the architecture (TCN + indicator MLP + fusion), the loss formula, the metrics list, the
`python -m src.train` interface and the output files.
