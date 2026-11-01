from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def select_threshold(y_true, probabilities, max_false_alarm=0.30):
    """
    Decision threshold chosen on VALIDATION data: the best F1 among thresholds whose
    false-alarm rate is at most max_false_alarm. (Plain best-F1 can end up as
    'alarm every day' when the model is weak, so the false-alarm cap is needed.)
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities)
    if len(np.unique(y_true)) < 2:
        return 0.5

    prec, rec, thr = precision_recall_curve(y_true, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    neg = max(int((y_true == 0).sum()), 1)
    far = np.array([((p >= t) & (y_true == 0)).sum() / neg for t in thr])
    ok = far <= max_false_alarm
    if not ok.any():
        return 0.5
    return float(thr[np.where(ok, f1, -1.0).argmax()])


def classification_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) > 1

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if has_both else 0.0,
        "no_skill_pr_auc": float(y_true.mean()),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def predicted_reversal_events(dates, probabilities, threshold=0.5):
    """one predicted reversal per run of consecutive days above the threshold (the most probable day)"""
    dates = np.asarray(dates)
    probabilities = np.asarray(probabilities)

    positive = probabilities >= threshold
    events = []
    i = 0
    while i < len(positive):
        if not positive[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(positive) and positive[j + 1]:
            j += 1
        best = i + int(np.argmax(probabilities[i:j + 1]))
        events.append({
            "date": str(np.datetime_as_string(np.datetime64(dates[best]), unit="D")),
            "probability": float(probabilities[best]),
        })
        i = j + 1
    return events


def event_matching(true_event_dates, predicted_events, tolerance_days=3):
    """
    A predicted reversal is correct when it is within tolerance_days of a true reversal point.
    Every true reversal can be matched once.
    Returns the number of true reversals, predicted reversals and matches.
    """
    true_dates = [np.datetime64(d, "D") for d in true_event_dates]
    pred_dates = [np.datetime64(e["date"], "D") for e in predicted_events]

    matched = set()
    correct = 0
    for p in pred_dates:
        best, best_d = None, None
        for idx, t in enumerate(true_dates):
            if idx in matched:
                continue
            d = abs(int((p - t).astype("timedelta64[D]").astype(int)))
            if best_d is None or d < best_d:
                best, best_d = idx, d
        if best is not None and best_d <= tolerance_days:
            matched.add(best)
            correct += 1
    return {"true": len(true_dates), "predicted": len(pred_dates), "matched": correct}
