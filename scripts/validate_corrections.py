#!/usr/bin/env python3
"""
Validate Corrections Implementation
Validates that all LOGS_ANALYSIS.md corrections have been properly implemented.
"""

import ast
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CorrectionValidator:
    """Validates implementation of LOGS_ANALYSIS.md corrections."""

    def __init__(self):
        self.base_path = Path(".")
        self.results = {
            "correction_1_overfitting": {"status": "NOT_CHECKED", "details": []},
            "correction_2_feature_324": {"status": "NOT_CHECKED", "details": []},
            "correction_3_fallback_logic": {"status": "NOT_CHECKED", "details": []},
            "correction_4_model_balance": {"status": "NOT_CHECKED", "details": []},
        }

    def check_overfitting_correction(self) -> Dict[str, Any]:
        """Check Correction 1: Overfitting fixes in transformers.py."""
        logger.info("🔍 Checking Correction 1: Overfitting Prevention...")

        result = {"status": "NOT_FOUND", "details": []}

        try:
            # Check transformers.py for overfitting prevention
            transformers_file = self.base_path / "pff/validators/ensembles/ensemble_wrappers/transformers.py"

            if not transformers_file.exists():
                result["details"].append("❌ transformers.py not found")
                return result

            with open(transformers_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for min_confidence_threshold increase
            if 'min_confidence_threshold: float = 0.05' in content:
                result["details"].append("✅ min_confidence_threshold set to 0.05")
                result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ min_confidence_threshold not found or not set to 0.05")

            # Check for violation percentage monitoring
            if 'violation_percentage' in content and 'max_violation_percentage' in content:
                result["details"].append("✅ Violation percentage monitoring implemented")
                if result["status"] != "NOT_FOUND":
                    result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ Violation percentage monitoring not found")

            # Check for overfitting alerts
            if 'violation_percentage > max_violation_percentage' in content:
                result["details"].append("✅ Overfitting alerts implemented")
                result["status"] = "COMPLETE"
            else:
                result["details"].append("❌ Overfitting alerts not found")

            # Check for validation warnings
            if 'logger.warning' in content and 'percentual de violações' in content:
                result["details"].append("✅ Portuguese validation warnings found")
            else:
                result["details"].append("⚠️ Portuguese validation warnings not found (optional)")

        except Exception as e:
            result["details"].append(f"❌ Error checking transformers.py: {e}")
            result["status"] = "ERROR"

        return result

    def check_feature_324_correction(self) -> Dict[str, Any]:
        """Check Correction 2: Feature 324 bug fix."""
        logger.info("🔍 Checking Correction 2: Feature 324 Bug Fix...")

        result = {"status": "NOT_FOUND", "details": []}

        try:
            transformers_file = self.base_path / "pff/validators/ensembles/ensemble_wrappers/transformers.py"

            with open(transformers_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for get_feature_names_out method
            if 'def get_feature_names_out(self' in content:
                result["details"].append("✅ get_feature_names_out method implemented")
                result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ get_feature_names_out method not found")

            # Check for feature 324 detection
            if 'analyze_feature_distribution' in content or '324' in content:
                result["details"].append("✅ Feature 324 detection logic found")
                if result["status"] != "NOT_FOUND":
                    result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ Feature 324 detection logic not found")

            # Check for feature importance logging
            if 'feature_names' in content and 'importance' in content:
                result["details"].append("✅ Feature importance logging implemented")
                result["status"] = "COMPLETE"
            else:
                result["details"].append("❌ Feature importance logging not found")

        except Exception as e:
            result["details"].append(f"❌ Error checking feature 324 fix: {e}")
            result["status"] = "ERROR"

        return result

    def check_fallback_logic_correction(self) -> Dict[str, Any]:
        """Check Correction 3: Fallback logic fix."""
        logger.info("🔍 Checking Correction 3: Fallback Logic Fix...")

        result = {"status": "NOT_FOUND", "details": []}

        try:
            transformers_file = self.base_path / "pff/validators/ensembles/ensemble_wrappers/transformers.py"

            with open(transformers_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for corrected fallback message (should be commented out or improved)
            if '# logger.info(f"🔄 Usando fallback: calculando violations manualmente' in content:
                result["details"].append("✅ Incorrect fallback message commented out")
                result["status"] = "PARTIAL"
            elif '🔄 Usando fallback: calculando violations manualmente' not in content:
                result["details"].append("✅ Incorrect fallback message removed")
                result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ Incorrect fallback message still present")

            # Check for proper Numba success logging
            if 'Numba: batch-parallel succeeded' in content or '⚡ Using Numba JIT acceleration' in content:
                result["details"].append("✅ Proper Numba success logging implemented")
                if result["status"] != "NOT_FOUND":
                    result["status"] = "COMPLETE"
            else:
                result["details"].append("❌ Proper Numba logging not found")

            # Check for consistent logging
            if 'logger.info' in content and 'Numba' in content:
                result["details"].append("✅ Numba logging consistency found")
            else:
                result["details"].append("⚠️ Numba logging consistency could be improved")

        except Exception as e:
            result["details"].append(f"❌ Error checking fallback logic: {e}")
            result["status"] = "ERROR"

        return result

    def check_model_balance_correction(self) -> Dict[str, Any]:
        """Check Correction 4: Model balance fixes in advanced_trainer.py."""
        logger.info("🔍 Checking Correction 4: Model Balance Fix...")

        result = {"status": "NOT_FOUND", "details": []}

        try:
            trainer_file = self.base_path / "pff/validators/ensembles/advanced_trainer.py"

            if not trainer_file.exists():
                result["details"].append("❌ advanced_trainer.py not found")
                return result

            with open(trainer_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for balanced XGBoost parameters
            if 'balanced_meta_params' in content:
                result["details"].append("✅ Balanced meta parameters section found")
                result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ Balanced meta parameters not found")

            # Check for specific parameter changes
            xgb_params_found = 0
            if '"n_estimators": 400' in content:
                result["details"].append("✅ n_estimators increased to 400")
                xgb_params_found += 1
            else:
                result["details"].append("❌ n_estimators not set to 400")

            if '"max_depth": 4' in content:
                result["details"].append("✅ max_depth increased to 4")
                xgb_params_found += 1
            else:
                result["details"].append("❌ max_depth not set to 4")

            if '"colsample_bytree": 0.4' in content:
                result["details"].append("✅ colsample_bytree increased to 0.4")
                xgb_params_found += 1
            else:
                result["details"].append("❌ colsample_bytree not set to 0.4")

            if '"scale_pos_weight": 1.0' in content:
                result["details"].append("✅ scale_pos_weight set for class balancing")
                xgb_params_found += 1
            else:
                result["details"].append("❌ scale_pos_weight not set to 1.0")

            if xgb_params_found >= 3:
                if result["status"] != "NOT_FOUND":
                    result["status"] = "COMPLETE"
            elif xgb_params_found >= 2:
                if result["status"] != "NOT_FOUND":
                    result["status"] = "PARTIAL"

            # Check for num_rules fix
            if "getattr(self, 'n_rules', 0)" in content:
                result["details"].append("✅ num_rules NameError fix implemented")
            else:
                result["details"].append("❌ num_rules NameError fix not found")

        except Exception as e:
            result["details"].append(f"❌ Error checking model balance fix: {e}")
            result["status"] = "ERROR"

        return result

    def check_accelerator_improvements(self) -> Dict[str, Any]:
        """Check for accelerator framework improvements."""
        logger.info("🔍 Checking Accelerator Framework Improvements...")

        result = {"status": "NOT_FOUND", "details": []}

        try:
            # Check for loop_accelerator.py
            loop_accelerator = self.base_path / "pff/utils/acceleration/loop_accelerator.py"
            if loop_accelerator.exists():
                result["details"].append("✅ loop_accelerator.py exists")
                result["status"] = "PARTIAL"

                with open(loop_accelerator, 'r') as f:
                    content = f.read()
                    if 'ParallelStrategy' in content and '@staticmethod' in content:
                        result["details"].append("✅ ParallelStrategy static method fix implemented")
                        result["status"] = "COMPLETE"
            else:
                result["details"].append("❌ loop_accelerator.py not found")

            # Check for symbolic_rule_accelerator.py
            symbolic_accelerator = self.base_path / "pff/utils/acceleration/symbolic_rule_accelerator.py"
            if symbolic_accelerator.exists():
                result["details"].append("✅ symbolic_rule_accelerator.py exists")

                with open(symbolic_accelerator, 'r') as f:
                    content = f.read()
                    if 'check_violations_vectorized' in content:
                        result["details"].append("✅ Vectorized processing implemented")
                    if 'adaptive strategy selection' in content:
                        result["details"].append("✅ Adaptive strategy selection implemented")
            else:
                result["details"].append("❌ symbolic_rule_accelerator.py not found")

        except Exception as e:
            result["details"].append(f"❌ Error checking accelerator improvements: {e}")
            result["status"] = "ERROR"

        return result

    def check_design_patterns(self) -> Dict[str, Any]:
        """Check for design patterns implementation."""
        logger.info("🔍 Checking Design Patterns Implementation...")

        result = {"status": "NOT_FOUND", "details": []}

        try:
            # Check for processors directory
            processors_dir = self.base_path / "processors"
            if processors_dir.exists() and processors_dir.is_dir():
                result["details"].append("✅ processors directory exists")
                result["status"] = "PARTIAL"

                # Check for key pattern files
                pattern_files = [
                    "base.py",
                    "strategies.py",
                    "factory.py",
                    "config.py",
                    "builder.py"
                ]

                files_found = 0
                for file in pattern_files:
                    file_path = processors_dir / file
                    if file_path.exists():
                        result["details"].append(f"✅ {file} exists")
                        files_found += 1
                    else:
                        result["details"].append(f"❌ {file} not found")

                if files_found >= 4:
                    result["status"] = "COMPLETE"
                elif files_found >= 2:
                    if result["status"] != "NOT_FOUND":
                        result["status"] = "PARTIAL"
            else:
                result["details"].append("❌ processors directory not found")

        except Exception as e:
            result["details"].append(f"❌ Error checking design patterns: {e}")
            result["status"] = "ERROR"

        return result

    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation of all corrections."""
        logger.info("🚀 Starting Corrections Validation...")

        # Run all checks
        self.results["correction_1_overfitting"] = self.check_overfitting_correction()
        self.results["correction_2_feature_324"] = self.check_feature_324_correction()
        self.results["correction_3_fallback_logic"] = self.check_fallback_logic_correction()
        self.results["correction_4_model_balance"] = self.check_model_balance_correction()

        # Additional checks
        accelerator_results = self.check_accelerator_improvements()
        patterns_results = self.check_design_patterns()

        # Calculate overall status
        total_corrections = len(self.results)
        complete_corrections = sum(1 for r in self.results.values() if r["status"] == "COMPLETE")
        partial_corrections = sum(1 for r in self.results.values() if r["status"] == "PARTIAL")
        error_corrections = sum(1 for r in self.results.values() if r["status"] == "ERROR")

        overall_status = "COMPLETE"
        if complete_corrections == total_corrections:
            overall_status = "COMPLETE"
        elif partial_corrections + complete_corrections >= total_corrections * 0.75:
            overall_status = "MOSTLY_COMPLETE"
        elif error_corrections > 0:
            overall_status = "ERROR"
        else:
            overall_status = "PARTIAL"

        summary = {
            "overall_status": overall_status,
            "corrections_summary": {
                "total": total_corrections,
                "complete": complete_corrections,
                "partial": partial_corrections,
                "errors": error_corrections
            },
            "corrections": self.results,
            "additional_improvements": {
                "accelerator_framework": accelerator_results,
                "design_patterns": patterns_results
            }
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "="*80)
        print("🔍 CORRECTIONS VALIDATION SUMMARY")
        print("="*80)

        # Overall status
        status_emoji = {
            "COMPLETE": "✅",
            "MOSTLY_COMPLETE": "🟡",
            "PARTIAL": "⚠️",
            "ERROR": "❌"
        }

        overall_status = summary["overall_status"]
        print(f"Overall Status: {status_emoji[overall_status]} {overall_status}")

        corr_summary = summary["corrections_summary"]
        print(f"Corrections: {corr_summary['complete']}/{corr_summary['total']} complete, "
              f"{corr_summary['partial']} partial, {corr_summary['errors']} errors")

        print(f"\n📋 CORRECTIONS STATUS:")
        print("-" * 40)

        correction_names = {
            "correction_1_overfitting": "Correction 1: Overfitting Prevention",
            "correction_2_feature_324": "Correction 2: Feature 324 Bug Fix",
            "correction_3_fallback_logic": "Correction 3: Fallback Logic Fix",
            "correction_4_model_balance": "Correction 4: Model Balance Fix"
        }

        for key, result in summary["corrections"].items():
            status_emoji_local = {
                "COMPLETE": "✅",
                "PARTIAL": "🟡",
                "ERROR": "❌",
                "NOT_FOUND": "❌"
            }

            name = correction_names.get(key, key)
            status = result["status"]
            print(f"{status_emoji_local.get(status, '❓')} {name:30} - {status}")

            # Show key details
            for detail in result["details"][:2]:  # Show first 2 details
                print(f"    {detail}")
            if len(result["details"]) > 2:
                print(f"    ... and {len(result['details']) - 2} more details")

        # Additional improvements
        print(f"\n🚀 ADDITIONAL IMPROVEMENTS:")
        print("-" * 40)

        improvements = summary["additional_improvements"]

        accel_status = improvements["accelerator_framework"]["status"]
        print(f"{status_emoji.get(accel_status, '❓')} Accelerator Framework     - {accel_status}")

        patterns_status = improvements["design_patterns"]["status"]
        print(f"{status_emoji.get(patterns_status, '❓')} Design Patterns          - {patterns_status}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)

        if overall_status == "COMPLETE":
            print("✅ All corrections successfully implemented!")
            print("🔄 Run pipeline with new corrections to validate effectiveness")
            print("📊 Monitor pipeline health with: python scripts/pipeline_health_monitor.py")
        elif overall_status == "MOSTLY_COMPLETE":
            print("🟡 Most corrections implemented, some may need attention")
            print("🔧 Review partial corrections and complete remaining items")
        else:
            print("⚠️ Several corrections need attention")
            print("🔧 Review error corrections and fix implementation issues")

        print("\n🎯 NEXT STEPS:")
        print("1. Test pipeline with: pff learn kg")
        print("2. Monitor health with: python scripts/pipeline_health_monitor.py")
        print("3. Check for improved metrics (violation % < 200, symbolic % < 85)")

        print("\n" + "="*80)

def main():
    """Main execution function."""
    validator = CorrectionValidator()

    try:
        summary = validator.run_validation()
        validator.print_summary(summary)

        # Exit codes based on status
        status = summary["overall_status"]
        if status == "COMPLETE":
            sys.exit(0)
        elif status == "MOSTLY_COMPLETE":
            sys.exit(0)
        elif status == "ERROR":
            sys.exit(2)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()