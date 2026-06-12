import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    print("\n[Error] Missing mamba_ssm. Please install mamba-ssm and causal-conv1d.\n")
    Mamba = None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


class BiMamba(nn.Module):
    """Bidirectional Mamba adapter for image restoration features."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_linear = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        out_fwd = self.mamba_fwd(x)

        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])

        out = torch.cat([out_fwd, out_bwd], dim=-1)
        return self.output_linear(out)


class VSSBlock(nn.Module):
    """Visual state-space block that preserves [B, C, H, W] tensor layout."""

    def __init__(self, hidden_dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        del drop_path

        if Mamba is None:
            raise ImportError("Mamba module not found.")

        self.ln_1 = LayerNorm2d(hidden_dim)
        self.self_attention = BiMamba(dim=hidden_dim, d_state=16, d_conv=4, expand=2)

        self.ln_2 = LayerNorm2d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.ln_1(x)
        _, _, h, w = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x = self.self_attention(x_flat)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = identity + x
        return x + self.mlp(self.ln_2(x))


if __name__ == "__main__":
    print("Testing VSSBlock...")
    block = VSSBlock(hidden_dim=64).cuda()
    dummy_input = torch.randn(2, 64, 128, 128).cuda()
    output = block(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == dummy_input.shape
    print("Test passed.")
