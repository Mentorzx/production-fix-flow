"""
Probabilistic Circuits (PC) minimal package.

TODO[Fase1-PC]: Implementar compilador completo; versão atual usa fallback seguro.
"""

from .strategy import ProbabilisticCircuitStrategy
from .compiler import RuleToCircuitCompiler
from .inference import PCInferenceEngine

__all__ = [
    "ProbabilisticCircuitStrategy",
    "RuleToCircuitCompiler",
    "PCInferenceEngine",
]
