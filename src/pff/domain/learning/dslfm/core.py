"""DSLFM-KGC core facade (Adapter/Facade).

Provides stable, generic import paths for the DSLFM-KGC model stack without
changing the underlying behavior. Keeps the public surface small and generic
(`core`, `config`, `manager`) while reusing the existing implementations.
"""

from __future__ import annotations

from .dslfm_kgc import (
    DSLFMKGCConfig,
    DSLFMKGCModel,
    DSLFMModel,
    create_dslfm_kgc_model,
)
from .logic_layer import DifferentiableRuleEncoder, RuleDefinition
from .neg_sampling import (
    BaseNegativeSampler,
    DegreeBasedSampler,
    SamplerConfig,
    SamplerType,
    get_negative_sampler,
)
from .sbm_decoder import LowRankSBMDecoder, StochasticBlockmodelDecoder
from .vae import DSLFMVAEEncoder, IndianBuffetProcessPrior

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
