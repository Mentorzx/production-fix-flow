"""DSLFM-KGC package exports.

DSLFM-KGC: Deep Sparse Latent Feature Model for Knowledge Graph Completion.
Implements BERT encoder for relations, VAE with Indian Buffet Process,
and Stochastic Blockmodel decoder.
"""

from .logic_layer import DifferentiableRuleEncoder, RuleDefinition
from .config import DSLFMConfig, load_dslfm_config

from .dslfm_kgc import DSLFMKGCModel, DSLFMKGCConfig, create_dslfm_kgc_model
from .kgc_manager import DSLFMKGCManager, KGCTrainingConfig, train_dslfm_kgc
from .core import DSLFMKGCModel as DSLFMCoreModel, DSLFMKGCConfig as DSLFMCoreConfig
from .manager import (
    DSLFMKGCManager as DSLFMManagerFacade,
    KGCTrainingConfig as KGCTrainingConfigFacade,
)
from .validator import DSLFMValidator
from .metrics import DSLFMMetricsReporter
from .vae import DSLFMVAEEncoder, IndianBuffetProcessPrior
from .sbm_decoder import StochasticBlockmodelDecoder, LowRankSBMDecoder
from .neg_sampling import (
    BaseNegativeSampler,
    DegreeBasedSampler,
    SamplerType,
    SamplerConfig,
    get_negative_sampler,
)
from .bert_encoder import (
    RelationTextEncoder,
    LightweightRelationEncoder,
    create_relation_encoder,
    TRANSFORMERS_AVAILABLE,
)
from .evaluation import (  # noqa: E402
    ApproximateEvaluator,  # noqa: F401  # Not exported: used internally only
    ExactEvaluator,  # noqa: F401  # Not exported: used internally only
    EvaluatorConfig,  # noqa: F401
    IndexType,  # noqa: F401
    get_evaluator,  # noqa: F401
)

__all__ = [
    "DifferentiableRuleEncoder",
    "RuleDefinition",
    "DSLFMConfig",
    "load_dslfm_config",
    "DSLFMKGCModel",
    "DSLFMKGCConfig",
    "DSLFMCoreModel",
    "DSLFMCoreConfig",
    "create_dslfm_kgc_model",
    "DSLFMKGCManager",
    "KGCTrainingConfig",
    "DSLFMManagerFacade",
    "KGCTrainingConfigFacade",
    "DSLFMValidator",
    "DSLFMMetricsReporter",
    "train_dslfm_kgc",
    "DSLFMVAEEncoder",
    "IndianBuffetProcessPrior",
    "StochasticBlockmodelDecoder",
    "LowRankSBMDecoder",
    "BaseNegativeSampler",
    "DegreeBasedSampler",
    "SamplerType",
    "SamplerConfig",
    "get_negative_sampler",
    "RelationTextEncoder",
    "LightweightRelationEncoder",
    "create_relation_encoder",
    "TRANSFORMERS_AVAILABLE",
]
