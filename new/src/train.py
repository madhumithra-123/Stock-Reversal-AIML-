from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data_pipeline import (
    INDICATOR_COLUMNS,
    PRICE_COLUMNS,
    build_stock_dataframe,
    create_split_samples,
    merge_split,
)
from .losses import LOSS_NAMES, build_criterion
from .metrics import (
    classification_metrics,
    event_matching,
    predicted_reversal_events,
    select_threshold,
)
from .model import ReversalAwareModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_csvs(data_dir: Path, stocks=None):
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        csvs = sorted(data_dir.rglob("*.csv"))
    if stocks:
        wanted = {s.strip().upper() for s in stocks.split(",") if s.strip()}
        csvs = [p for p in csvs if p.stem.upper() in wanted]
    return csvs


def prepare_stocks(csvs, cfg):
    frames = {}
    for path in csvs:
        try:
            frames[path.stem] = build_stock_dataframe(path, cfg)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
    return frames


def make_loader(xp, xi, y, r, batch_size, shuffle, seed=0):
    ds = TensorDataset(
        torch.tensor(xp, dtype=torch.float32),
        torch.tensor(xi, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(r, dtype=torch.float32),
    )
    gen = torch.Generator().manual_seed(seed)
    # drop_last: BatchNorm fails on a batch with a single sample
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, generator=gen)


def run_epoch(model, loader, optimizer, criterion, device, training):
    model.train(training)
    total_loss, count = 0.0, 0

    with torch.set_grad_enabled(training):
        for xp, xi, y, r in loader:
            xp, xi, y, r = xp.to(device), xi.to(device), y.to(device), r.to(device)
            loss = criterion(model(xp, xi), y, r)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += float(loss.item()) * len(y)
            count += len(y)

    return total_loss / max(count, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for xp, xi, y, _ in loader:
        logits = model(xp.to(device), xi.to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(y.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0


def train_one(loss_name, cfg, device, loaders, data, true_events, scaler, out):
    """Train one model with the given loss, pick the threshold on validation data, test once, save files."""
    train_loader, val_loader, test_loader = loaders
    tag = "reversal_aware" if loss_name == "reversal_aware" else "bce"

    # same seed -> same starting weights and same batch order for both losses
    set_seed(cfg.seed)
    model = ReversalAwareModel(
        price_features=len(PRICE_COLUMNS), indicator_features=len(INDICATOR_COLUMNS),
        indicator_hidden=cfg.indicator_hidden, indicator_out=cfg.indicator_out,
        tcn_channels=cfg.tcn_channels, fusion_hidden=cfg.fusion_hidden, dropout=cfg.dropout,
    ).to(device)

    criterion = build_criterion(loss_name, cfg.lambda_)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # the best epoch is the one with the best validation PR-AUC, which does not depend on the loss used
    best_ap, best_state, patience_left = -1.0, None, cfg.patience

    print(f"\n========== TRAINING ({loss_name}) ==========")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, False)
        yv, pv = predict(model, val_loader, device)
        val_ap = safe_ap(yv, pv)

        print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.6f} | "
              f"val_loss={val_loss:.6f} | val_PR-AUC={val_ap:.4f}")

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # decision threshold from validation data, then frozen for the test set
    yv, pv = predict(model, val_loader, device)
    threshold = select_threshold(yv, pv, cfg.max_false_alarm)
    y_test, p_test = predict(model, test_loader, device)
    metrics = classification_metrics(y_test, p_test, threshold)

    # per-stock predicted reversals and event matching
    pooled = {"true": 0, "predicted": 0, "matched": 0}
    all_events, pred_rows = [], []
    for symbol, xp, xi, y, r, dates in data["test"]:
        _, probs = predict(model, make_loader(xp, xi, y, r, cfg.batch_size, False), device)
        events = predicted_reversal_events(dates, probs, threshold)
        match = event_matching(true_events["test"].get(symbol, []), events, tolerance_days=3)
        for k in pooled:
            pooled[k] += match[k]
        all_events += [{"stock": symbol, **e} for e in events]
        for d, a, p in zip(dates, y, probs):
            pred_rows.append({
                "stock": symbol, "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "actual_reversal": int(a), "reversal_probability": float(p),
                "predicted_reversal": int(p >= threshold),
            })

    metrics["event_recall_pm3d"] = pooled["matched"] / max(pooled["true"], 1)
    metrics["event_precision_pm3d"] = pooled["matched"] / max(pooled["predicted"], 1)
    metrics["true_reversal_points"] = pooled["true"]
    metrics["predicted_reversal_points"] = pooled["predicted"]
    metrics["best_val_pr_auc"] = float(best_ap)

    # file names: the reversal-aware model keeps the original names, BCE gets a "bce_" prefix
    prefix = "" if tag == "reversal_aware" else "bce_"
    model_file = "reversal_aware_model.pt" if tag == "reversal_aware" else "bce_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "loss": loss_name,
        "config": cfg.__dict__,
        "price_columns": PRICE_COLUMNS,
        "indicator_columns": INDICATOR_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": threshold,
    }, out / model_file)
    pd.DataFrame(pred_rows).to_csv(out / f"{prefix}test_predictions.csv", index=False)
    with open(out / f"{prefix}metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n================ TEST METRICS ({loss_name}) ================")
    print(f"Decision threshold (from validation) : {metrics['threshold']:.4f}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1             : {metrics['f1']:.4f}")
    print(f"PR-AUC         : {metrics['pr_auc']:.4f}")
    print(f"False Alarm    : {metrics['false_alarm_rate']:.4f}")
    print(f"Reversal points found within +-3 days : {pooled['matched']} of {pooled['true']} "
          f"(event recall {metrics['event_recall_pm3d']:.3f}, event precision {metrics['event_precision_pm3d']:.3f})")
    print("===============================================")
    return metrics, all_events, model_file


COMPARE_ROWS = [
    ("threshold", "Decision threshold"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("pr_auc", "PR-AUC"),
    ("false_alarm_rate", "False alarm rate"),
    ("event_recall_pm3d", "Event recall (+-3d)"),
    ("event_precision_pm3d", "Event precision (+-3d)"),
    ("predicted_reversal_points", "Predicted reversals"),
]
# metrics where a smaller value is better
LOWER_IS_BETTER = {"false_alarm_rate"}
# rows that are shown but not judged (not a quality measure)
NOT_JUDGED = {"threshold", "predicted_reversal_points"}


def print_comparison(results, out):
    ra, bce = results["reversal_aware"], results["bce"]
    print("\n================ COMPARISON: reversal-aware loss vs plain BCE (test set) ================")
    print(f"{'Metric':26s} {'Reversal-aware':>16s} {'BCE':>12s} {'Better':>16s}")
    wins = {"reversal_aware": 0, "bce": 0}
    for key, label in COMPARE_ROWS:
        a, b = ra[key], bce[key]
        if key in NOT_JUDGED:
            better = "-"
        elif abs(a - b) < 1e-12:
            better = "tie"
        else:
            a_better = (a < b) if key in LOWER_IS_BETTER else (a > b)
            better = "reversal-aware" if a_better else "BCE"
            wins["reversal_aware" if a_better else "bce"] += 1
        fmt = (lambda v: f"{int(v)}") if key == "predicted_reversal_points" else (lambda v: f"{v:.4f}")
        print(f"{label:26s} {fmt(a):>16s} {fmt(b):>12s} {better:>16s}")
    print(f"No-skill PR-AUC (share of positive test days): {ra['no_skill_pr_auc']:.4f}")
    print(f"Metrics won -> reversal-aware: {wins['reversal_aware']}, BCE: {wins['bce']}")
    print("Note: one run, one seed. The difference can be noise; repeat with --seed 1, 2, 3 before drawing conclusions.")
    print("=========================================================================================")
    with open(out / "comparison.json", "w", encoding="utf-8") as f:
        json.dump({"reversal_aware": ra, "bce": bce, "wins": wins}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/stocknet/price/raw")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stocks", type=str, default="")
    parser.add_argument("--loss", type=str, default="both", choices=list(LOSS_NAMES) + ["both"],
                        help="reversal_aware, bce, or both (default: train both and compare)")
    parser.add_argument("--lambda_", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="fixed swing size for reversal labels, e.g. 0.025. Default: automatic per stock")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_false_alarm", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_print", type=int, default=40,
                        help="maximum number of predicted reversals printed at the end")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs, lambda_=args.lambda_, sigma=args.sigma,
        reversal_threshold=args.threshold, seq_len=args.seq_len,
        batch_size=args.batch_size, seed=args.seed, max_false_alarm=args.max_false_alarm,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csvs = discover_csvs(Path(args.data_dir), args.stocks)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {args.data_dir}. Expected StockNet price/raw/*.csv")
    print(f"Stocks found: {len(csvs)}")

    frames = prepare_stocks(csvs, cfg)
    if not frames:
        raise RuntimeError("No stock files could be prepared.")

    data, scaler, true_events, thresholds, (train_end, val_end) = create_split_samples(frames, cfg)
    train, val, test = merge_split(data["train"]), merge_split(data["val"]), merge_split(data["test"])

    print(f"Stocks used: {len(thresholds)} | average swing size for a reversal: "
          f"{100 * np.mean(list(thresholds.values())):.1f}%")
    print(f"Split dates -> train up to {train_end.date()}, val up to {val_end.date()}, then test")
    print(f"Samples -> train={len(train[2])}, val={len(val[2])}, test={len(test[2])}")
    print(f"Positive ratios -> train={train[2].mean():.4f}, val={val[2].mean():.4f}, test={test[2].mean():.4f}")

    loaders = (
        make_loader(*train[:4], cfg.batch_size, True, cfg.seed),
        make_loader(*val[:4], cfg.batch_size, False),
        make_loader(*test[:4], cfg.batch_size, False),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = list(LOSS_NAMES) if args.loss == "both" else [args.loss]
    results, saved = {}, []
    events_ra = []  # predicted reversals printed at the end (reversal-aware model if trained, else BCE)
    for name in names:
        # the training loader is re-created so both losses see exactly the same batch order
        loaders = (make_loader(*train[:4], cfg.batch_size, True, cfg.seed), loaders[1], loaders[2])
        metrics, events, model_file = train_one(name, cfg, device, loaders, data, true_events, scaler, out)
        results[name] = metrics
        saved.append(model_file)
        if name == "reversal_aware" or args.loss == "bce":
            events_ra = events

    if len(results) == 2:
        print_comparison(results, out)

    shown = "reversal-aware" if "reversal_aware" in results else "BCE"
    print(f"\n========== PREDICTED REVERSALS ({shown} model) ==============")
    if events_ra:
        print(f"{'Stock':8s} {'Date':12s} Probability")
        for row in events_ra[:args.max_print]:
            print(f"{row['stock']:8s} {row['date']:12s} {row['probability']:.4f}")
        if len(events_ra) > args.max_print:
            csv_name = "test_predictions.csv" if "reversal_aware" in results else "bce_test_predictions.csv"
            print(f"... {len(events_ra) - args.max_print} more in {out}/{csv_name}")
    else:
        print("No predicted reversals above the decision threshold.")

    files = saved + (["bce_test_predictions.csv", "bce_metrics.json"] if "bce" in results else []) \
        + (["test_predictions.csv", "metrics.json"] if "reversal_aware" in results else []) \
        + (["comparison.json"] if len(results) == 2 else [])
    print(f"\nSaved in {out}/: " + ", ".join(files))


if __name__ == "__main__":
    main()
