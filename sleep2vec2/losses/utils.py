import torch


def contrastive_accuracy(logits_12: torch.Tensor, logits_21: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred12 = logits_12.argmax(dim=-1)
        pred21 = logits_21.argmax(dim=-1)
        acc = 0.5 * ((pred12 == labels).float().mean() + (pred21 == labels).float().mean())
    return acc
