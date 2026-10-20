import torch
from transformers import AutoModel


class DreamModelWrapper(torch.nn.Module):
    """
    Thin wrapper around the DREAM model to normalize position_ids shape.

    Our decoding functions pass position_ids as 1D (seq_len,) for single-sequence
    decoding, and 2D (B, seq_len) for batched decoding. DREAM's RotaryEmbedding
    expects 2D unconditionally (does position_ids[:, None, :]), so we unsqueeze
    1D inputs here rather than touching the shared decoding code.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        return self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
        return cls(model)
