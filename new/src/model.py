import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        y = self.conv1(x)
        y = self.chomp1(y)
        y = F.relu(self.norm1(y))
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = F.relu(self.norm2(y))
        y = self.drop2(y)

        return F.relu(y + residual)


class TemporalEncoder(nn.Module):
    def __init__(self, in_ch, channels=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            TemporalBlock(in_ch, channels, 3, 1, dropout),
            TemporalBlock(channels, channels, 3, 2, dropout),
            TemporalBlock(channels, channels, 3, 4, dropout),
            TemporalBlock(channels, channels, 3, 8, dropout),
        )

    def forward(self, x):
        # x: [B, T, F] -> TCN expects [B, F, T]
        x = x.transpose(1, 2)
        y = self.net(x)
        return y[:, :, -1]


class IndicatorEncoder(nn.Module):
    def __init__(self, in_features, hidden=64, out_features=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ReversalAwareModel(nn.Module):
    def __init__(
        self,
        price_features,
        indicator_features,
        indicator_hidden=64,
        indicator_out=32,
        tcn_channels=64,
        fusion_hidden=64,
        dropout=0.2,
    ):
        super().__init__()

        self.indicator_encoder = IndicatorEncoder(
            indicator_features, indicator_hidden, indicator_out, dropout
        )
        self.temporal_encoder = TemporalEncoder(
            price_features, tcn_channels, dropout
        )

        self.fusion = nn.Sequential(
            nn.Linear(indicator_out + tcn_channels, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_hidden, 1)

    def forward(self, price_sequence, indicator_now):
        temporal = self.temporal_encoder(price_sequence)
        indicator = self.indicator_encoder(indicator_now)
        fused = torch.cat([temporal, indicator], dim=1)
        fused = self.fusion(fused)
        return self.head(fused).squeeze(1)
