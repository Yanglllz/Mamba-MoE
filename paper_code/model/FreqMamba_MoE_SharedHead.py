"""Shared-head Mamba-MoE model entry point used by the public release.

This module keeps the manuscript naming convention while delegating the
implementation to the installable ``mamba_moe`` package code in this repository.
The model exposes deterministic known-modality H-MoE routing and one shared
reconstruction head.
"""

from mamba_moe.model import (  # noqa: F401
    AMIRProMambaSharedHead,
    ChannelExpert,
    DeterministicHMoE,
    SpatialExpert,
    build_mamba_moe_sharedhead,
    inject_hmoe_layers,
)


def build_mustwin_sharedhead_model(device: str = "cpu", checkpoint_path: str | None = None):
    """Compatibility wrapper for manuscript-era experiment scripts."""

    return build_mamba_moe_sharedhead(device=device, base_ckpt=checkpoint_path)
