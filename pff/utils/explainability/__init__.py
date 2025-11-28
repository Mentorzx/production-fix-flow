"""
Explainability utilities (SHAP).

Design Patterns:
- Factory Pattern: SHAP explainer selection based on model type.
- Template Method: Standardized explain/save workflow with hooks for sampling and persistence.
"""

from .shap_explainer import ShapExplainerService, ShapExplainerConfig

__all__ = ["ShapExplainerService", "ShapExplainerConfig"]
