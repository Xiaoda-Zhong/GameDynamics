"""Minimal ordered integer-action adapter for the 17-frame Wan VACE pilot.

The adapter adds a learned signal to the *noisy* patch embedding output, before
WanVACETransformer3DModel flattens temporal and spatial positions. It does not
touch the native VACE O0 conditioning or the control patch embedding.
"""

from contextlib import contextmanager
from collections.abc import Iterator

import torch
from torch import nn


NUM_ACTIONS = 6
ACTION_NAMES = (
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "NOOP",
    "MOVE_FORWARD_LEFT",
    "MOVE_FORWARD_RIGHT",
)
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}
NUM_TRANSITIONS = 16
ACTION_DIM = 32
ACTIONS_PER_LATENT = 4
TRANSFORMER_WIDTH = 1536
NUM_LATENT_FRAMES = 5
PATCH_HEIGHT = 16
PATCH_WIDTH = 28


class VACEActionAdapter(nn.Module):
    """Map 16 ordered action IDs to four future VACE temporal latent slots."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(NUM_ACTIONS, ACTION_DIM)
        self.projection = nn.Linear(ACTIONS_PER_LATENT * ACTION_DIM, TRANSFORMER_WIDTH)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def _check_action_ids(action_ids: torch.Tensor) -> None:
        if not isinstance(action_ids, torch.Tensor):
            raise TypeError("action_ids must be a torch.Tensor")
        if action_ids.ndim != 2 or action_ids.shape[1] != NUM_TRANSITIONS:
            raise ValueError(f"action_ids must have shape [B,{NUM_TRANSITIONS}], got {tuple(action_ids.shape)}")
        if action_ids.shape[0] < 1:
            raise ValueError("action_ids batch must be nonempty")
        if action_ids.dtype != torch.long:
            raise TypeError(f"action_ids must be torch.int64, got {action_ids.dtype}")
        if torch.any((action_ids < 0) | (action_ids >= NUM_ACTIONS)):
            raise ValueError(f"action_ids must be in [0,{NUM_ACTIONS - 1}]")

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Return `[B,5,1536]`, with an exact zero at observed latent slot 0."""
        self._check_action_ids(action_ids)
        action_ids = action_ids.to(device=self.embedding.weight.device)
        embedded = self.embedding(action_ids)  # [B,16,32], original order
        groups = embedded.reshape(action_ids.shape[0], 4, ACTIONS_PER_LATENT * ACTION_DIM)
        future = self.projection(groups)  # [B,4,1536]
        observed_zero = future.new_zeros((action_ids.shape[0], 1, TRANSFORMER_WIDTH))
        return torch.cat((observed_zero, future), dim=1)

    def add_to_patch_output(self, action_ids: torch.Tensor, patch_output: torch.Tensor) -> torch.Tensor:
        """Add the action signal to `[B,1536,5,16,28]` noisy patch features."""
        self._check_action_ids(action_ids)
        if not isinstance(patch_output, torch.Tensor):
            raise TypeError("patch_embedding must return a torch.Tensor")
        expected = (action_ids.shape[0], TRANSFORMER_WIDTH, NUM_LATENT_FRAMES, PATCH_HEIGHT, PATCH_WIDTH)
        if tuple(patch_output.shape) != expected:
            raise ValueError(f"noisy patch output must have shape {expected}, got {tuple(patch_output.shape)}")
        bias = self(action_ids).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return patch_output + bias.to(device=patch_output.device, dtype=patch_output.dtype)

    @contextmanager
    def inject(self, transformer: nn.Module, action_ids: torch.Tensor) -> Iterator[None]:
        """Temporarily inject actions at the frozen noisy patch embedding output.

        The caller owns the transformer and must keep its pretrained parameters
        frozen. A fresh hook is removed even if the transformer call raises.
        """
        self._check_action_ids(action_ids)
        patch_embedding = getattr(transformer, "patch_embedding", None)
        if not isinstance(patch_embedding, nn.Conv3d):
            raise TypeError("transformer.patch_embedding must be nn.Conv3d")
        if (
            patch_embedding.in_channels != 16
            or patch_embedding.out_channels != TRANSFORMER_WIDTH
            or patch_embedding.kernel_size != (1, 2, 2)
            or patch_embedding.stride != (1, 2, 2)
        ):
            raise ValueError("transformer.patch_embedding does not match the audited Wan VACE configuration")

        def add_actions(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            return self.add_to_patch_output(action_ids, output)

        handle = patch_embedding.register_forward_hook(add_actions)
        try:
            yield
        finally:
            handle.remove()
