from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # LoRA low-rank decomposition: W' = W + A @ B
        # A: [out_features, lora_dim], B: [lora_dim, in_features]
        # Keep LoRA layers in float32 for better training
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)

        # Initialize LoRA weights
        # Standard practice: Initialize A with small random values, B with zeros
        # This ensures the initial LoRA contribution is zero (W + A@B = W initially)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

        # Ensure LoRA layers are trainable (they should be by default, but being explicit)
        self.lora_a.weight.requires_grad_(True)
        self.lora_b.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the base linear layer output (from HalfLinear parent class)
        # HalfLinear handles the float16 conversion internally and returns in x.dtype
        base_output = super().forward(x)

        # Compute LoRA adaptation in float32
        # x is in original dtype (float32), LoRA layers are in float32
        lora_output = self.lora_b(self.lora_a(x))

        # Combine: output = base + LoRA (both in x.dtype)
        return base_output + lora_output


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            # Use LoRALinear instead of regular Linear layers
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        # Same structure as BigNet, but using LoRA-enabled blocks
        # LayerNorm stays in full precision
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
