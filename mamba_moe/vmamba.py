import torch
import torch.nn as nn
from einops import rearrange
try:
    from mamba_ssm import Mamba
except ImportError:
    print("\n[Error] 缺少 mamba_ssm 库。请运行: pip install mamba-ssm causal-conv1d\n")
    Mamba = None

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

class BiMamba(nn.Module):
    """
    双向 Mamba 适配器
    标准的 Mamba 是因果的(Causal)，只看前文。
    对于图像任务，我们需要同时利用前向和后向信息。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # 正向 Mamba
        self.mamba_fwd = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        # 反向 Mamba
        self.mamba_bwd = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
        # 融合层
        self.output_linear = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x input: [B, L, C] (Sequence format)
        
        # 1. 正向扫描
        out_fwd = self.mamba_fwd(x)
        
        # 2. 反向扫描 (翻转序列 -> Mamba -> 翻转回来)
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # 3. 拼接并融合
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.output_linear(out)
        
        return out

class VSSBlock(nn.Module):
    """
    Visual State Space Block (适配版)
    输入输出严格保持 [B, C, H, W]
    """
    def __init__(self, hidden_dim, drop_path=0):
        super().__init__()
        
        if Mamba is None:
            raise ImportError("Mamba module not found.")

        self.ln_1 = LayerNorm2d(hidden_dim)
        
        # 双向 Mamba 用于处理图像
        self.self_attention = BiMamba(
            dim=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2  # 内部维度会扩展到 2*hidden_dim，这是 Mamba 的标准做法
        )
        
        self.ln_2 = LayerNorm2d(hidden_dim)
        
        # 前馈网络 (FFN) 部分，通常 Mamba 块后会接一个 MLP 或 ConvFFN
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, 1)
        ) # 使用 1x1 Conv 模拟 MLP

    def forward(self, x):
        # Input: [B, C, H, W]
        input_x = x
        
        # --- Part 1: Mamba Mixer ---
        x = self.ln_1(x)
        
        # 维度变换: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Mamba 处理 (Sequence Modeling)
        x_mamba = self.self_attention(x_flat)
        
        # 维度还原: [B, H*W, C] -> [B, C, H, W]
        x = rearrange(x_mamba, 'b (h w) c -> b c h w', h=H, w=W)
        
        # 残差连接 1
        x = input_x + x
        
        # --- Part 2: FFN ---
        # 残差连接 2
        x = x + self.mlp(self.ln_2(x))
        
        return x

if __name__ == "__main__":
    # 测试代码
    print("Testing VSSBlock...")
    block = VSSBlock(hidden_dim=64).cuda()
    dummy_input = torch.randn(2, 64, 128, 128).cuda()
    output = block(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == dummy_input.shape
    print("Test Passed!")