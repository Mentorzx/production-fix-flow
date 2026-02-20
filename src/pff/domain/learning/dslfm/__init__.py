"""DSLFM-KGC package exports.

DSLFM-KGC: Deep Sparse Latent Feature Model for Knowledge Graph Completion.
Implements BERT encoder for relations, VAE with Indian Buffet Process,
and Stochastic Blockmodel decoder.
"""

from .bert_encoder import (
    TRANSFORMERS_AVAILABLE,
    LightweightRelationEncoder,
    RelationTextEncoder,
    create_relation_encoder,
)
from .dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel, create_dslfm_kgc_model
from .dslfm_kgc import DSLFMKGCConfig as DSLFMCoreConfig
from .dslfm_kgc import DSLFMKGCModel as DSLFMCoreModel
from .evaluation import (
    ApproximateEvaluator,
    EvaluatorConfig,
    ExactEvaluator,
    IndexType,
    get_evaluator,
)
from .kgc_manager import DSLFMKGCManager, KGCTrainingConfig, train_dslfm_kgc
from .logic_layer import DifferentiableRuleEncoder, RuleDefinition
from .kgc_manager import DSLFMKGCManager as DSLFMManagerFacade
from .kgc_manager import KGCTrainingConfig as KGCTrainingConfigFacade
from .metrics_reporter import DSLFMMetricsReporter
from .neg_sampling import (
    BaseNegativeSampler,
    DegreeBasedSampler,
    SamplerConfig,
    SamplerType,
    get_negative_sampler,
)
from .sbm_decoder import LowRankSBMDecoder, StochasticBlockmodelDecoder
from .vae import DSLFMVAEEncoder, IndianBuffetProcessPrior
from .validator import DSLFMValidator

__all__ = [
    "DifferentiableRuleEncoder",
    "RuleDefinition",
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
    "ApproximateEvaluator",
    "ExactEvaluator",
    "EvaluatorConfig",
    "IndexType",
    "get_evaluator",
]
