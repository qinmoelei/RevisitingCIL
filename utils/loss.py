import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "sum":
            return F_loss.sum()
        elif self.reduction == "mean":
            return F_loss.mean()


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes=30, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=self.dim))


def kl_divergence_loss(logits:torch.tensor, targets:torch.tensor):
    batch_size, class_num = logits.size()
    targets_one_hot = torch.zeros(batch_size, class_num).cuda()
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, targets_one_hot, reduction="batchmean")


huber_loss = nn.SmoothL1Loss()
hinge_loss = nn.MultiMarginLoss()

if __name__ == "__main__":
    a = torch.randn(10, 30)
    b = torch.randn(10)
