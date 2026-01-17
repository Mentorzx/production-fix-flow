"""DSLFM-KGC core facade (Adapter/Facade).

Provides stable, generic import paths for the DSLFM-KGC model stack without
changing the underlying behavior. Keeps the public surface small and generic
(`core`, `config`, `manager`) while reusing the existing implementations.
"""

from __future__ import annotations

from .dslfm_kgc import (
    DSLFMModel,
    DSLFMKGCConfig,
    DSLFMKGCModel,
    create_dslfm_kgc_model,
)  # noqa: F401
from .vae import DSLFMVAEEncoder, IndianBuffetProcessPrior  # noqa: F401
from .sbm_decoder import StochasticBlockmodelDecoder, LowRankSBMDecoder  # noqa: F401
from .neg_sampling import (  # noqa: F401
    BaseNegativeSampler,
    DegreeBasedSampler,
    SamplerType,
    SamplerConfig,
    get_negative_sampler,
)
from .logic_layer import DifferentiableRuleEncoder, RuleDefinition  # noqa: F401

__all__ = [
    "DSLFMModel",
    "DSLFMKGCModel",
    "DSLFMKGCConfig",
    "create_dslfm_kgc_model",
    "DSLFMVAEEncoder",
    "IndianBuffetProcessPrior",
    "StochasticBlockmodelDecoder",
    "LowRankSBMDecoder",
    "BaseNegativeSampler",
    "DegreeBasedSampler",
    "SamplerType",
    "SamplerConfig",
    "get_negative_sampler",
    "DifferentiableRuleEncoder",
    "RuleDefinition",
]
