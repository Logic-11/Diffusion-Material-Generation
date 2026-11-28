"""Assemble a CSP model from the shared denoiser and continuous corruptions."""

from collections.abc import Mapping

from torch import Tensor

from ..backbone.denoiser import Denoiser
from ..diffusion.corruption import LatticeCorruption, PositionCorruption
from ..diffusion.scheduler import LatticeVPScheduler, PositionVEScheduler
from .diffusion import CSPDiffusion


def build_csp_model(
    *,
    lattice_config: Mapping[str, float],
    denoiser_config: Mapping[str, object] | None = None,
    position_config: Mapping[str, float] | None = None,
    graph_config: Mapping[str, object] | None = None,
    denoiser_state_dict: Mapping[str, Tensor] | None = None,
) -> CSPDiffusion:
    """Build a model with a configured lattice density.

    To initialize from joint diffusion, pass the matching denoiser settings
    and the denoiser's own state dict, with keys relative to the denoiser.
    """
    denoiser = Denoiser(**(denoiser_config or {}))
    if denoiser_state_dict is not None:
        denoiser.load_state_dict(denoiser_state_dict, strict=True)
    return CSPDiffusion(
        denoiser,
        PositionCorruption(PositionVEScheduler(**(position_config or {}))),
        LatticeCorruption(LatticeVPScheduler(**lattice_config)),
        **(graph_config or {}),
    )
