import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss for segmentation tasks.
    - BCE stabilizes early optimization gradients.
    - Dice directly optimizes the region-overlap metric (IoU/Overlap).
    """
    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE Loss using raw logits for numerical stability
        bce_loss = self.bce(logits, targets)

        # Soft Dice Loss computed via Sigmoid probabilities
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        return (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss)
