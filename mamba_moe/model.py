import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vmamba import VSSBlock
except ImportError as exc:
    raise ImportError(
        "VSSBlock is required. Add the project-compatible VMamba implementation "
        "as mamba_moe/vmamba.py before instantiating Mamba-MoE."
    ) from exc


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class MambaIRBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.vssm = VSSBlock(hidden_dim=dim)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.ca = ChannelAttention(dim)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x_norm = self.norm(x)
        x_mamba = self.vssm(x_norm)
        x_local = self.ca(self.local_conv(x_norm))
        return self.proj(x_mamba + self.alpha * x_local) + identity


class RIN(nn.Module):
    def __init__(self, in_channels: int = 1, dict_size: int = 16, instr_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, dict_size, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.dictionary = nn.Parameter(torch.randn(dict_size, instr_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.softmax(self.encoder(x).flatten(1))
        instruction = weights @ self.dictionary
        return instruction, weights


class TokenExpert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SRM(nn.Module):
    def __init__(self, dim: int, instr_dim: int = 256, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList([TokenExpert(dim) for _ in range(num_experts)])
        self.router_fc_x = nn.Linear(dim, 64)
        self.router_fc_instr = nn.Linear(instr_dim, 64)
        self.router_classifier = nn.Linear(128, num_experts)

    def forward(self, x: torch.Tensor, instruction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, c = x.shape
        x_flat = x.view(b, h * w, c)
        instr_proj = self.router_fc_instr(instruction).unsqueeze(1).expand(-1, h * w, -1)
        router_input = torch.cat([self.router_fc_x(x_flat), instr_proj], dim=-1)
        logits = self.router_classifier(router_input)
        topk_weights, topk_indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = topk_indices[:, :, k]
            weight = topk_weights[:, :, k].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = (idx == expert_id).unsqueeze(-1)
                if mask.any():
                    out = out + mask * weight * expert(x_flat)
        return out.view(b, h, w, c), logits


class CRM(nn.Module):
    def __init__(self, dim: int, instr_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(instr_dim, dim // 4),
            nn.ReLU(True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        return x * self.fc(instruction).unsqueeze(-1).unsqueeze(-1)


class AMIRProMambaSharedHead(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, dim: int = 48) -> None:
        super().__init__()
        self.rin = RIN(in_channels=in_channels)
        self.shallow_conv = nn.Conv2d(in_channels, dim, 3, 1, 1)

        self.srm1 = SRM(dim)
        self.enc1 = nn.Sequential(*[MambaIRBlock(dim) for _ in range(2)])
        self.down1 = nn.Conv2d(dim, dim * 2, 2, 2)

        self.srm2 = SRM(dim * 2)
        self.enc2 = nn.Sequential(*[MambaIRBlock(dim * 2) for _ in range(2)])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 2, 2)

        self.crm_bottle = CRM(dim * 4)
        self.bottleneck = nn.Sequential(*[MambaIRBlock(dim * 4) for _ in range(4)])

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, 2)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.crm2 = CRM(dim * 2)
        self.dec2 = nn.Sequential(*[MambaIRBlock(dim * 2) for _ in range(2)])

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 2, 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1)
        self.crm1 = CRM(dim)
        self.dec1 = nn.Sequential(*[MambaIRBlock(dim) for _ in range(2)])

        self.output_head = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, task_id_batch: Optional[torch.Tensor] = None):
        del task_id_batch
        instr, _ = self.rin(x)
        feat = self.shallow_conv(x)

        feat_srm, logits1 = self.srm1(feat.permute(0, 2, 3, 1), instr)
        feat = feat + feat_srm.permute(0, 3, 1, 2)
        feat_l1 = self.enc1(feat)
        feat = self.down1(feat_l1)

        feat_srm, logits2 = self.srm2(feat.permute(0, 2, 3, 1), instr)
        feat = feat + feat_srm.permute(0, 3, 1, 2)
        feat_l2 = self.enc2(feat)
        feat = self.down2(feat_l2)

        feat = self.bottleneck(self.crm_bottle(feat, instr))

        feat = self.up2(feat)
        feat = self.reduce2(torch.cat([feat, feat_l2], dim=1))
        feat = self.dec2(self.crm2(feat, instr))

        feat = self.up1(feat)
        feat = self.reduce1(torch.cat([feat, feat_l1], dim=1))
        feat = self.dec1(self.crm1(feat, instr))

        return self.output_head(feat) + x, [logits1, logits2]


class SpatialExpert(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.act = nn.GELU()
        self.spatial = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False)
        self.up = nn.Conv2d(rank, out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.spatial(self.act(self.down(x))))


class ChannelExpert(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rank: int, stride) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, rank, 1, stride=stride, bias=False),
            nn.Conv2d(rank, out_channels, 1, bias=False),
        )
        nn.init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.net[1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicHMoE(nn.Module):
    CURRENT_TASK: Optional[str] = None

    def __init__(self, base_conv: nn.Conv2d, rank_dict: Dict[str, int]) -> None:
        super().__init__()
        self.base_conv = base_conv
        for param in self.base_conv.parameters():
            param.requires_grad = False

        self.experts = nn.ModuleDict()
        if rank_dict.get("MRI", 0) > 0:
            if base_conv.stride in (1, (1, 1)):
                self.experts["MRI_Spatial"] = SpatialExpert(base_conv.in_channels, base_conv.out_channels, rank_dict["MRI"])
            else:
                self.experts["MRI_Channel"] = ChannelExpert(
                    base_conv.in_channels, base_conv.out_channels, rank_dict["MRI"], base_conv.stride
                )
        if rank_dict.get("CT", 0) > 0:
            self.experts["CT"] = ChannelExpert(base_conv.in_channels, base_conv.out_channels, rank_dict["CT"], base_conv.stride)
        if rank_dict.get("PET", 0) > 0:
            self.experts["PET"] = ChannelExpert(
                base_conv.in_channels, base_conv.out_channels, rank_dict["PET"], base_conv.stride
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_conv(x)
        task = DeterministicHMoE.CURRENT_TASK
        key = None
        if task == "MRI":
            key = "MRI_Spatial" if "MRI_Spatial" in self.experts else "MRI_Channel"
        elif task == "CT":
            key = "CT"
        elif task == "PET":
            key = "PET"
        return out + self.experts[key](x) if key in self.experts else out


def inject_hmoe_layers(model: nn.Module, rank_config: Optional[Dict[str, int]] = None) -> int:
    rank_config = rank_config or {"MRI": 128, "CT": 64, "PET": 32}
    injected = 0

    def replace(module: nn.Module) -> None:
        nonlocal injected
        for name, child in list(module.named_children()):
            if name in {"rin", "output_head", "head", "tail", "tails", "exit"}:
                continue
            if isinstance(child, nn.Conv2d) and child.in_channels >= 32:
                setattr(module, name, DeterministicHMoE(child, rank_config))
                injected += 1
            else:
                replace(child)

    replace(model)
    return injected


def build_mamba_moe_sharedhead(device: str = "cpu", base_ckpt: Optional[str] = None) -> AMIRProMambaSharedHead:
    model = AMIRProMambaSharedHead().to(device)
    inject_hmoe_layers(model)
    if base_ckpt:
        checkpoint = torch.load(base_ckpt, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state_dict, dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
    return model

