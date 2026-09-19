from __future__ import annotations

from typing import Any, Sequence

from .training import CompletionOnlyCollator


class CompletionLogitsCollator(CompletionOnlyCollator):
    """Add Qwen3 loss-selection tensors while preserving the original batch."""

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(features)
        if not self.return_tensors:
            raise ValueError("CompletionLogitsCollator requires return_tensors=True")

        import torch
        import torch.nn.functional as F

        labels = batch["labels"]
        shifted = F.pad(labels, (0, 1), value=self.label_pad_token_id)[..., 1:].contiguous()
        selected_positions = torch.nonzero(shifted.ne(self.label_pad_token_id).any(dim=0), as_tuple=False).flatten()
        if selected_positions.numel() == 0:
            raise ValueError("completion logits selection requires at least one supervised label")

        batch["logits_to_keep"] = selected_positions
        batch["shift_labels"] = shifted.index_select(dim=1, index=selected_positions).contiguous()
        return batch
