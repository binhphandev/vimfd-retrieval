from xml.parsers.expat import model
import torch
import torch.nn as nn
from src.models.encoders import VisualEncoder, TextEncoder
from src.models.projection import ProjectionHead
from config import (
    VISUAL_BACKBONE_DIM,
    TEXT_BACKBONE_DIM,
    LOGIT_SCALE_INIT,
    LOGIT_SCALE_MAX,
)


class ViMFRModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.visual_encoder = VisualEncoder()
        self.text_encoder = TextEncoder()

        # ProjectionHead riêng cho từng encoder
        self.visual_proj = ProjectionHead(in_dim=VISUAL_BACKBONE_DIM)
        self.text_proj = ProjectionHead(in_dim=TEXT_BACKBONE_DIM)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * LOGIT_SCALE_INIT)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.visual_encoder(pixel_values)  # (B, 768)
        return self.visual_proj(features)  # (B, 256)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        features = self.text_encoder(input_ids, attention_mask)  # (B, 768)
        return self.text_proj(features)  # (B, 256)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Clamp logit_scale để tránh temperature quá nhỏ
        logit_scale = self.logit_scale.exp().clamp(max=LOGIT_SCALE_MAX)

        return image_embeds, text_embeds, logit_scale

def load_state_dict_compat(model, state_dict):
    prefix = "visual_encoder.backbone."
    nested = prefix + "vision_model."
    model_keys = set(model.state_dict().keys())

    fixed = {}
    for k, v in state_dict.items():
        if k not in model_keys and k.startswith(prefix):
            if not k.startswith(nested) and (nested + k[len(prefix) :]) in model_keys:
                k = nested + k[len(prefix) :]  # ckpt phẳng -> model lồng
            elif k.startswith(nested) and (prefix + k[len(nested) :]) in model_keys:
                k = prefix + k[len(nested) :]  # ckpt lồng -> model phẳng
        fixed[k] = v

    res = model.load_state_dict(fixed, strict=False)
    missing = [k for k in res.missing_keys if "position_ids" not in k]
    unexpected = [k for k in res.unexpected_keys if "position_ids" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint không khớp model.\nMissing: {missing[:10]}\nUnexpected: {unexpected[:10]}"
        )
    return model
