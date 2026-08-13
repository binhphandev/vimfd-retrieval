import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel, AutoTokenizer
from config import EMBED_DIM, PROJECTION_DROPOUT
from config import CLIP_MODEL, UNFREEZE_LAST_N, PHOBERT_MODEL


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(CLIP_MODEL)
        self._freeze_then_unfreeze()

    def _freeze_then_unfreeze(self):
        # Bước 1: freeze toàn bộ
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Bước 2: unfreeze N block cuối
        if UNFREEZE_LAST_N > 0:
            blocks = self.backbone.vision_model.encoder.layers
            for block in blocks[-UNFREEZE_LAST_N:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Luôn unfreeze layer norm cuối cùng
            for param in self.backbone.vision_model.post_layernorm.parameters():
                param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pooler_output = CLS token đã qua post_layernorm → shape (B, 512)
        output = self.backbone(pixel_values=pixel_values)
        return output.pooler_output


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(PHOBERT_MODEL)
        self._freeze_then_unfreeze()

    def _freeze_then_unfreeze(self):
        # Bước 1: freeze toàn bộ
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Bước 2: unfreeze N block cuối
        if UNFREEZE_LAST_N > 0:
            blocks = self.backbone.encoder.layer
            for block in blocks[-UNFREEZE_LAST_N:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Luôn unfreeze layer norm cuối
            for param in self.backbone.embeddings.LayerNorm.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy CLS token — index 0 của sequence
        cls_output = output.last_hidden_state[:, 0, :]   # shape (B, 768)
        return cls_output