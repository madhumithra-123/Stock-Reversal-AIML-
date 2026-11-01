import torch.nn as nn
import torch.nn.functional as F


class ReversalAwareLoss(nn.Module):
    """
    L_total = mean((1 + lambda * R_t) * BCE(y_t, logit_t))

    R_t is supplied by the data pipeline from the distance to the nearest reversal point.
    """

    def __init__(self, lambda_=2.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, logits, targets, reversal_proximity):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = 1.0 + self.lambda_ * reversal_proximity
        return (weights * bce).mean()


class BCELoss(nn.Module):
    """
    Plain binary cross-entropy: L = mean(BCE(y_t, logit_t)).

    This is the baseline for the comparison. It has the same call signature as
    ReversalAwareLoss, but ignores reversal_proximity (every day has weight 1).
    It is the same as ReversalAwareLoss with lambda = 0.
    """

    def forward(self, logits, targets, reversal_proximity=None):
        return F.binary_cross_entropy_with_logits(logits, targets)


LOSS_NAMES = ("reversal_aware", "bce")


def build_criterion(name, lambda_=2.0):
    if name == "reversal_aware":
        return ReversalAwareLoss(lambda_)
    if name == "bce":
        return BCELoss()
    raise ValueError(f"Unknown loss '{name}'. Choose from {LOSS_NAMES}.")
