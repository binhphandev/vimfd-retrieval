import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_embeds: torch.Tensor,   # (B, 256) đã L2-norm
        text_embeds: torch.Tensor,    # (B, 256) đã L2-norm
        logit_scale: torch.Tensor,    # scalar
    ) -> torch.Tensor:

        # Similarity matrix (B, B)
        # image_embeds @ text_embeds.T = cosine similarity (vì đã L2-norm)
        logits_per_image = logit_scale * image_embeds @ text_embeds.T  # (B, B)
        logits_per_text  = logits_per_image.T                          # (B, B)

        # Label: diagonal là positive pair
        # [0,1,2,...,B-1] — ảnh thứ i khớp với text thứ i
        labels = torch.arange(image_embeds.size(0), device=image_embeds.device)

        # Cross-entropy 2 chiều (symmetric)
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text  = F.cross_entropy(logits_per_text,  labels)

        return (loss_image + loss_text) / 2