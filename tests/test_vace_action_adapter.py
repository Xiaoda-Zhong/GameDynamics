"""Focused invariants for the M6B-1 VACE action adapter."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from training.wan.vace_action_adapter import VACEActionAdapter


class VACEActionAdapterTest(unittest.TestCase):
    def test_zero_init_hook_and_ordered_temporal_mapping(self):
        adapter = VACEActionAdapter()
        assert sum(parameter.numel() for parameter in adapter.parameters()) == 198_336
        assert set(adapter.state_dict()) == {"embedding.weight", "projection.weight", "projection.bias"}

        x_actions = torch.full((1, 16), 3, dtype=torch.long)
        y_actions = x_actions.clone()
        x_actions[0, :2] = torch.tensor([1, 2])
        y_actions[0, :2] = torch.tensor([2, 1])

        patch = nn.Conv3d(16, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        patch.forward = lambda x: x.new_ones((x.shape[0], 1536, 5, 16, 28))
        transformer = SimpleNamespace(patch_embedding=patch)
        input_latents = torch.empty((1, 16, 5, 32, 56))
        baseline = patch(input_latents)
        with adapter.inject(transformer, x_actions):
            first = patch(input_latents)
        with adapter.inject(transformer, y_actions):
            second = patch(input_latents)
        assert torch.equal(first, baseline)
        assert torch.equal(second, baseline)
        assert not patch._forward_hooks

        with torch.no_grad():
            adapter.embedding.weight.zero_()
            adapter.embedding.weight[1, 0] = 1
            adapter.embedding.weight[2, 0] = 2
            adapter.projection.weight[0, 0] = 1
            adapter.projection.weight[0, 32] = -1
        first_bias = adapter(x_actions)
        second_bias = adapter(y_actions)
        assert torch.equal(first_bias[:, 0], torch.zeros_like(first_bias[:, 0]))
        assert torch.equal(first_bias[:, 2:], second_bias[:, 2:])
        assert first_bias[0, 1, 0].item() == -1
        assert second_bias[0, 1, 0].item() == 1

        adapter(x_actions)[:, 1:, 0].sum().backward()
        assert adapter.embedding.weight.grad.abs().sum() > 0
        assert adapter.projection.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    unittest.main()
