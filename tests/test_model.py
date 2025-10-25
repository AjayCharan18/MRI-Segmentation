from __future__ import annotations

import torch

from model import UNet


def test_unet_forward_pass():
    model = UNet(in_channels=1, out_channels=1)
    inputs = torch.rand(2, 1, 32, 32, 32)
    outputs = model(inputs)

    assert outputs.shape == inputs.shape
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
