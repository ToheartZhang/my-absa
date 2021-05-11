import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smooth_eps = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        smooth_pos = target.eq(n_class)
        target = target.masked_fill(smooth_pos, 0)
        target_matrix = torch.full_like(
            pred, fill_value=self.smooth_eps / (n_class - 1)
        )
        target_matrix = target_matrix.scatter(
            dim=1, index=target.unsqueeze(1), value=1 - self.smooth_eps
        )
        target_matrix = target_matrix.masked_fill(
            smooth_pos.unsqueeze(1), 1.0 / n_class
        )

        pred = F.log_softmax(pred, dim=-1)
        loss = -(pred * target_matrix).sum(dim=-1).mean()
        return loss