"""
Test for symbolic features fix (Sprint 23-24).

Validates that symbolic features have >1% sparsity (non-zero) 
and model balance is 40-60% between hybrid and symbolic.
"""
import pytest
from pathlib import Path


def test_model_balance_between_hybrid_and_symbolic():
    """
    Test that model balance is between 40-60% for both hybrid and symbolic.
    
    Before the fix, balance was 93.59% hybrid vs 6.41% symbolic (UNBALANCED).
    After the fix, it should be ~50/50 (BALANCED).
    """
    # Load last metrics report if exists
    metrics_path = Path("outputs/ensemble/metrics_all.json")
    
    if not metrics_path.exists():
        pytest.skip("No metrics file found - run 'pff learn ensemble' first")
    
    import json
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Get feature importance contributions
    balance = metrics.get("Feature_Balance", {})
    hybrid_contrib_str = balance.get("contribution_ratio", {}).get("hybrid", "0%")
    symbolic_contrib_str = balance.get("contribution_ratio", {}).get("symbolic", "0%")
    
    # Parse percentages
    hybrid_contrib = float(hybrid_contrib_str.rstrip('%'))
    symbolic_contrib = float(symbolic_contrib_str.rstrip('%'))
    
    # Assert balanced (40-60% range)
    assert 40 <= hybrid_contrib <= 60, (
        f"Hybrid contribution {hybrid_contrib:.2f}% is outside 40-60% range. "
        f"Model is UNBALANCED (too much hybrid)."
    )
    
    assert 40 <= symbolic_contrib <= 60, (
        f"Symbolic contribution {symbolic_contrib:.2f}% is outside 40-60% range. "
        f"Model is UNBALANCED (too much symbolic)."
    )
    
    print(f"✅ Model balance: Hybrid {hybrid_contrib:.2f}% vs Symbolic {symbolic_contrib:.2f}%")


def test_f1_score_improvement_after_fix():
    """
    Test that F1-Score improved after the symbolic features fix.
    
    Before fix: 0.5871
    After fix: >0.60 (expected)
    """
    metrics_path = Path("outputs/ensemble/metrics_all.json")
    
    if not metrics_path.exists():
        pytest.skip("No metrics file found - run 'pff learn ensemble' first")
    
    import json
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Get F1-Score from Ensemble_Final
    f1_score = metrics.get("Ensemble_Final", {}).get("f1_score", 0)
    
    # Assert improvement (before: 0.5871, after: >0.60)
    assert f1_score > 0.60, (
        f"F1-Score {f1_score:.4f} is below 0.60 threshold. "
        f"Expected improvement after symbolic features fix."
    )
    
    print(f"✅ F1-Score: {f1_score:.4f} (>0.60 threshold)")


def test_symbolic_features_sparsity_greater_than_zero():
    """
    Test that symbolic features have >0% sparsity (non-zero elements).
    
    Before the fix, sparsity was 0% due to Numba matching bugs.
    After the fix, it should be >1% (ideally ~1.2%).
    """
    metrics_path = Path("outputs/ensemble/metrics_all.json")
    
    if not metrics_path.exists():
        pytest.skip("No metrics file found - run 'pff learn ensemble' first")
    
    import json
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Check symbolic contribution
    balance = metrics.get("Feature_Balance", {})
    symbolic_contrib_str = balance.get("contribution_ratio", {}).get("symbolic", "0%")
    symbolic_contrib = float(symbolic_contrib_str.rstrip('%'))
    
    # If symbolic > 0%, then sparsity > 0%
    assert symbolic_contrib > 0, (
        f"Symbolic contribution is 0%! This means symbolic features are all zeros. "
        f"Numba matching is completely broken."
    )
    
    # Assert reasonable sparsity (>40%)
    assert symbolic_contrib > 40, (
        f"Symbolic contribution {symbolic_contrib:.2f}% is too low (<40%). "
        f"Expected balanced contribution after fix."
    )
    
    print(f"✅ Symbolic contribution: {symbolic_contrib:.2f}% (>40% threshold)")


@pytest.mark.slow
def test_full_ensemble_training_with_symbolic_features():
    """
    Full integration test: Train ensemble and validate symbolic features work.
    
    This is a slow test that runs the complete ensemble pipeline.
    """
    from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
    from pff.services.data_service import DataService
    
    # Load data
    data_service = DataService()
    ensemble_data = data_service.load_ensemble_data()
    
    # Train ensemble
    trainer = AdvancedEnsembleTrainer(
        output_dir="outputs/ensemble_test",
        enable_model_balancing=True,
    )
    
    result = trainer.train(
        train_data=ensemble_data["train"],
        test_data=ensemble_data["test"],
    )
    
    # Validate results
    assert result["f1_score"] > 0.60, f"F1-Score too low: {result['f1_score']}"
    assert result["symbolic_contribution"] > 40, (
        f"Symbolic contribution too low: {result['symbolic_contribution']}%"
    )
    
    print(f"✅ Full pipeline: F1={result['f1_score']:.4f}, "
          f"Symbolic={result['symbolic_contribution']:.2f}%")
