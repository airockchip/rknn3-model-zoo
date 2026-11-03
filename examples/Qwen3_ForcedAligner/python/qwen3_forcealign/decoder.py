"""Full-sequence timestamp head without autoregressive decoding or KV cache."""

import torch
from torch import nn


class TextAllLogits(nn.Module):
    def __init__(self, thinker):
        super().__init__()
        self.model = thinker.model
        self.lm_head = thinker.lm_head

    def forward(self, input_embeds, attention_mask, position_ids):
        n = input_embeds.shape[1]
        indices = torch.arange(n, device=input_embeds.device)
        allowed = (indices[None, :] <= indices[:, None])[
            None, None
        ] & attention_mask[:, None, None, :].bool()
        causal_mask = torch.where(
            allowed,
            torch.zeros(
                (), dtype=input_embeds.dtype, device=input_embeds.device
            ),
            torch.full(
                (),
                torch.finfo(input_embeds.dtype).min,
                dtype=input_embeds.dtype,
                device=input_embeds.device,
            ),
        )
        h = self.model(
            inputs_embeds=input_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        return self.lm_head(h)
