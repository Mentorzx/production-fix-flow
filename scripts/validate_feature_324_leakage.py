"""
Script to validate Feature 324 (TransE score) for potential feature leakage.

Feature 324 is the TransE score: -||h + r - t||
This script checks if this feature is overly correlated with the label,
which would indicate feature leakage (the model learns to copy TransE scores
instead of learning new patterns).

Usage:
    python scripts/validate_feature_324_leakage.py

Expected behavior:
    - If correlation > 0.85: HIGH RISK of leakage
    - If correlation 0.70-0.85: MODERATE RISK
    - If correlation < 0.70: OK (reasonable)

    - If AUC drops > 0.10 without Feature 324: CONFIRMED leakage
    - If AUC drops < 0.05: Feature 324 is not critical

Author: PFF Team
Date: 2025-10-31
"""

from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from pff import settings
from pff.utils import logger
from pff.validators.transe.lightgbm_trainer import TransELightGBMTrainer


def load_validation_data():
    """Load validation dataset and extract features."""
    logger.info("📂 Loading validation data...")

    # Load TransE manager (assumes checkpoint exists)
    from pff.validators.transe.core import TransEManager

    config_path = settings.CONFIG_DIR / "transe.yaml"
    transe_manager = TransEManager(config_path)
    transe_manager._load_checkpoint()

    # Create trainer
    trainer = TransELightGBMTrainer(transe_manager)

    # Load validation data
    val_path = settings.DATA_DIR / "models" / "correct" / "val_optimized.parquet"
    if not val_path.exists():
        logger.error(f"❌ Validation data not found: {val_path}")
        return None, None, None

    X_val, y_val, meta = trainer.create_lightgbm_dataset(val_path)

    logger.success(f"✅ Loaded {len(X_val)} validation samples with {X_val.shape[1]} features")

    return X_val, y_val, trainer


def analyze_feature_324_correlation(X_val, y_val):
    """Calculate correlation between Feature 324 and label."""
    logger.info("\n" + "="*80)
    logger.info("🔍 ANALYSIS 1: Feature 324 Correlation with Label")
    logger.info("="*80)

    feature_324 = X_val[:, 324]

    # Pearson correlation
    correlation = np.corrcoef(feature_324, y_val)[0, 1]

    logger.info(f"📊 Feature 324 (TransE score) correlation with label: {correlation:.4f}")

    # Interpretation
    if abs(correlation) > 0.85:
        logger.warning("🔴 HIGH RISK: Correlation > 0.85 indicates strong feature leakage!")
        logger.warning("   The model is likely just copying the TransE score.")
    elif abs(correlation) > 0.70:
        logger.warning("🟠 MODERATE RISK: Correlation 0.70-0.85 suggests some leakage.")
        logger.warning("   The model may be over-relying on TransE scores.")
    else:
        logger.info("✅ OK: Correlation < 0.70 is acceptable.")

    # Additional statistics
    logger.info(f"\n📈 Feature 324 statistics:")
    logger.info(f"   Min: {feature_324.min():.4f}")
    logger.info(f"   Max: {feature_324.max():.4f}")
    logger.info(f"   Mean: {feature_324.mean():.4f}")
    logger.info(f"   Std: {feature_324.std():.4f}")

    logger.info(f"\n📈 Label distribution:")
    logger.info(f"   Positive samples (y=1): {np.sum(y_val == 1)} ({np.mean(y_val)*100:.1f}%)")
    logger.info(f"   Negative samples (y=0): {np.sum(y_val == 0)} ({(1-np.mean(y_val))*100:.1f}%)")

    # Mean feature value per class
    logger.info(f"\n📈 Feature 324 by class:")
    logger.info(f"   Mean for y=1: {feature_324[y_val == 1].mean():.4f}")
    logger.info(f"   Mean for y=0: {feature_324[y_val == 0].mean():.4f}")
    logger.info(f"   Difference: {feature_324[y_val == 1].mean() - feature_324[y_val == 0].mean():.4f}")

    return correlation


def train_lightgbm_with_and_without_feature_324(X_train, y_train, X_val, y_val):
    """Train LightGBM with and without Feature 324 to measure impact."""
    logger.info("\n" + "="*80)
    logger.info("🔍 ANALYSIS 2: LightGBM Performance With/Without Feature 324")
    logger.info("="*80)

    # Load LightGBM config
    from pff.utils import FileManager
    config = FileManager().read(settings.CONFIG_DIR / "transe.yaml")
    lgb_config = config.get("lightgbm", {})
    lgb_params = lgb_config.get("params", {})
    training_config = lgb_config.get("training", {})

    # Prepare params
    params = {
        **lgb_params,
        "verbose": -1,
    }

    num_boost_round = training_config.get("num_boost_round", 50)
    early_stopping_rounds = training_config.get("early_stopping_rounds", 5)

    # Model WITH Feature 324
    logger.info("\n🔧 Training LightGBM WITH Feature 324...")
    train_data_with = lgb.Dataset(X_train, label=y_train)
    val_data_with = lgb.Dataset(X_val, label=y_val, reference=train_data_with)

    model_with = lgb.train(
        params,
        train_data_with,
        num_boost_round=num_boost_round,
        valid_sets=[val_data_with],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    y_pred_with = model_with.predict(X_val)
    auc_with = roc_auc_score(y_val, y_pred_with)
    acc_with = accuracy_score(y_val, (y_pred_with > 0.5).astype(int))
    f1_with = f1_score(y_val, (y_pred_with > 0.5).astype(int))

    logger.success(f"✅ WITH Feature 324: AUC={auc_with:.4f}, Acc={acc_with:.4f}, F1={f1_with:.4f}")

    # Model WITHOUT Feature 324 (remove column 324)
    logger.info("\n🔧 Training LightGBM WITHOUT Feature 324...")
    X_train_no324 = np.delete(X_train, 324, axis=1)
    X_val_no324 = np.delete(X_val, 324, axis=1)

    train_data_without = lgb.Dataset(X_train_no324, label=y_train)
    val_data_without = lgb.Dataset(X_val_no324, label=y_val, reference=train_data_without)

    model_without = lgb.train(
        params,
        train_data_without,
        num_boost_round=num_boost_round,
        valid_sets=[val_data_without],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    y_pred_without = model_without.predict(X_val_no324)
    auc_without = roc_auc_score(y_val, y_pred_without)
    acc_without = accuracy_score(y_val, (y_pred_without > 0.5).astype(int))
    f1_without = f1_score(y_val, (y_pred_without > 0.5).astype(int))

    logger.success(f"✅ WITHOUT Feature 324: AUC={auc_without:.4f}, Acc={acc_without:.4f}, F1={f1_without:.4f}")

    # Comparison
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARISON:")
    logger.info("="*80)

    auc_drop = auc_with - auc_without
    acc_drop = acc_with - acc_without
    f1_drop = f1_with - f1_without

    logger.info(f"AUC drop: {auc_drop:.4f} ({auc_drop/auc_with*100:.1f}%)")
    logger.info(f"Accuracy drop: {acc_drop:.4f} ({acc_drop/acc_with*100:.1f}%)")
    logger.info(f"F1 drop: {f1_drop:.4f} ({f1_drop/f1_with*100:.1f}%)")

    # Interpretation
    logger.info("\n" + "="*80)
    logger.info("🎯 INTERPRETATION:")
    logger.info("="*80)

    if auc_drop > 0.10:
        logger.warning("🔴 CONFIRMED LEAKAGE: AUC drops >10% without Feature 324")
        logger.warning("   The model is heavily dependent on the TransE score.")
        logger.warning("   Recommendation: Remove Feature 324 or use TransE score differently.")
    elif auc_drop > 0.05:
        logger.warning("🟠 MODERATE DEPENDENCY: AUC drops 5-10% without Feature 324")
        logger.warning("   The model relies significantly on TransE, but also uses other features.")
        logger.warning("   Recommendation: Monitor for overfitting, consider feature engineering.")
    else:
        logger.info("✅ HEALTHY: AUC drops <5% without Feature 324")
        logger.info("   The model uses Feature 324 but is not overly dependent on it.")

    return {
        "auc_with": auc_with,
        "auc_without": auc_without,
        "auc_drop": auc_drop,
        "acc_with": acc_with,
        "acc_without": acc_without,
        "f1_with": f1_with,
        "f1_without": f1_without,
    }


def analyze_feature_importance(trainer):
    """Analyze feature importance from existing LightGBM model."""
    logger.info("\n" + "="*80)
    logger.info("🔍 ANALYSIS 3: Feature Importance Analysis")
    logger.info("="*80)

    try:
        # Load existing model
        model_path = settings.OUTPUTS_DIR / "transe" / "lightgbm_model.txt"
        if not model_path.exists():
            logger.warning("⚠️ Existing LightGBM model not found, skipping importance analysis")
            return

        model = lgb.Booster(model_file=str(model_path))
        importance = model.feature_importance(importance_type='gain')

        # Get top 20 features
        top_indices = np.argsort(importance)[-20:][::-1]

        logger.info("\n📊 Top 20 Features by Importance:")
        for i, idx in enumerate(top_indices, 1):
            logger.info(f"   {i:2d}. Feature {idx:3d}: {importance[idx]:.2f}")

            if idx == 324:
                logger.warning(f"       ⚠️ Feature 324 (TransE score) ranked #{i}")

        # Feature 324 rank
        feature_324_rank = np.where(np.argsort(importance)[::-1] == 324)[0][0] + 1
        feature_324_importance = importance[324]
        total_importance = importance.sum()
        feature_324_pct = (feature_324_importance / total_importance) * 100

        logger.info(f"\n🎯 Feature 324 Statistics:")
        logger.info(f"   Rank: #{feature_324_rank} / {len(importance)}")
        logger.info(f"   Importance: {feature_324_importance:.2f}")
        logger.info(f"   Percentage of total: {feature_324_pct:.2f}%")

        if feature_324_rank <= 5:
            logger.warning(f"🔴 Feature 324 is in TOP 5 (rank #{feature_324_rank})")
            logger.warning("   This confirms it's one of the most important features.")
        elif feature_324_rank <= 20:
            logger.warning(f"🟠 Feature 324 is in TOP 20 (rank #{feature_324_rank})")
        else:
            logger.info(f"✅ Feature 324 rank is acceptable (#{feature_324_rank})")

    except Exception as e:
        logger.error(f"❌ Error analyzing feature importance: {e}")


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("🚀 Feature 324 Leakage Validation Script")
    logger.info("="*80)

    # Load data
    X_val, y_val, trainer = load_validation_data()

    if X_val is None:
        logger.error("❌ Failed to load validation data. Exiting.")
        return

    # Analysis 1: Correlation
    correlation = analyze_feature_324_correlation(X_val, y_val)

    # Analysis 2: Train with/without Feature 324
    # Split val into train/test for this experiment
    X_train, X_test, y_train, y_test = train_test_split(
        X_val, y_val, test_size=0.3, random_state=42, stratify=y_val
    )

    logger.info(f"\n📊 Split for experiment: {len(X_train)} train, {len(X_test)} test")

    results = train_lightgbm_with_and_without_feature_324(X_train, y_train, X_test, y_test)

    # Analysis 3: Feature importance from existing model
    analyze_feature_importance(trainer)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📝 FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Correlation with label: {correlation:.4f}")
    logger.info(f"✅ AUC with Feature 324: {results['auc_with']:.4f}")
    logger.info(f"✅ AUC without Feature 324: {results['auc_without']:.4f}")
    logger.info(f"✅ AUC drop: {results['auc_drop']:.4f} ({results['auc_drop']/results['auc_with']*100:.1f}%)")

    logger.info("\n" + "="*80)
    logger.info("🎯 RECOMMENDATION:")
    logger.info("="*80)

    if abs(correlation) > 0.85 and results['auc_drop'] > 0.10:
        logger.warning("🔴 STRONG EVIDENCE OF FEATURE LEAKAGE")
        logger.warning("   Action: Remove Feature 324 from LightGBM training")
        logger.warning("   Location: pff/validators/transe/lightgbm_trainer.py:94")
        logger.warning("   Change: Comment out the 'score' feature in feat_vec concatenation")
    elif abs(correlation) > 0.70 or results['auc_drop'] > 0.05:
        logger.warning("🟠 MODERATE RISK OF OVERFITTING")
        logger.warning("   Action: Monitor model performance on new data")
        logger.warning("   Consider: Feature engineering to reduce dependency")
    else:
        logger.info("✅ NO SIGNIFICANT LEAKAGE DETECTED")
        logger.info("   Feature 324 is useful but not causing leakage")

    logger.info("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
