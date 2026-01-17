"""
Probabilistic Circuit Variant 2 (PC2) package for Knowledge Graph Completion.

This package implements PC2 (Probabilistic Circuit Variant 2) from Patil et al.'s
work on Knowledge Graph Completion. PC2 is the second inference strategy that
computes **exact probabilities** using tractable circuit properties.

Key Distinction:
- **PC2 (this package)**: Probabilistic Circuit Variant 2 - exact inference
  strategy from KGC research.
- **PC² (NOT this)**: "Probabilistic Circuits Squared" - a theoretical concept
  in circuit complexity involving structured-decomposability and vtrees.

Key Components:
- NeuralProbabilisticCircuit (PC2): Main PC2 implementation with HCLT backbone
- ProbabilisticCircuitStrategy: Strategy for rule confidence aggregation
- RuleToCircuitCompiler: Minimal compiler with fallback to Noisy-OR
- PCInferenceEngine: Inference engine for compiled circuits

PC2 Properties (for exact inference):
- Smooth: All sum nodes have children with identical scopes
- Decomposable: All product nodes have children with disjoint scopes
- Exact inference: No approximations in marginal computation
"""

from .strategy import ProbabilisticCircuitStrategy
from .compiler import RuleToCircuitCompiler
from .inference import PCInferenceEngine
from .npc import (
    NeuralProbabilisticCircuit,
    PC2,
    CircuitProperties,
)

__all__ = [
    "ProbabilisticCircuitStrategy",
    "RuleToCircuitCompiler",
    "PCInferenceEngine",
    "NeuralProbabilisticCircuit",
    "PC2",
    "CircuitProperties",
]
