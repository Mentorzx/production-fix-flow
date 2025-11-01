#!/usr/bin/env python3
"""
Advanced Optimization Visualization with Anti-Overfitting Validation

Features:
- Interactive plots with ideal threshold lines
- Overfitting detection (train vs validation gap)
- Parameter importance analysis
- Pareto frontier visualization
- Historical convergence tracking

Author: PFF Team
Date: 2025-11-01
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from pff import settings
from pff.utils import logger
from pff.utils.core.file_manager import FileManager

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ═══════════════════════════════════════════════════════════════════════════
# Configuration & Thresholds
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptimalThresholds:
    """
    Ideal thresholds based on previous sprints and AGENTS.md analysis.
    
    Source: AGENTS.md - Sprint 23, LOGS_ANALYSIS.md
    """
    # Performance targets
    min_f1_score: float = 0.75
    max_f1_score: float = 0.85
    target_f1_score: float = 0.80
    
    # Overfitting prevention (CRITICAL)
    max_train_val_gap: float = 0.05  # Max 5% gap
    max_violation_percentage: float = 150.0  # Max 150% violations
    min_violation_percentage: float = 80.0   # Min 80% (below = underfitting)
    
    # Model balance (from LOGS_ANALYSIS.md)
    target_symbolic_ratio: float = 0.70  # Target 70% symbolic
    min_symbolic_ratio: float = 0.50     # Min 50%
    max_symbolic_ratio: float = 0.85     # Max 85%
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.02  # Optimal from Sprint 23
    max_confidence_threshold: float = 0.15  # Too high = underfitting
    
    # Sparsity (symbolic features)
    min_sparsity: float = 0.01  # Min 1% non-zero
    max_sparsity: float = 0.20  # Max 20% (above = too dense)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    output_dir: Path = settings.OUTPUTS_DIR / "hyperopt" / "plots"
    dpi: int = 300
    format: str = "png"
    show_plots: bool = False
    thresholds: OptimalThresholds = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = OptimalThresholds()
        if self.output_dir is None:
            self.output_dir = settings.OUTPUTS_DIR / "hyperopt" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Overfitting Detection
# ═══════════════════════════════════════════════════════════════════════════

class OverfittingDetector:
    """
    Detect overfitting in optimization results.
    
    Based on AGENTS.md - Sprint 23 fixes:
    - Train-validation gap > 5%
    - Violation percentage > 150%
    - Symbolic ratio > 85%
    """
    
    def __init__(self, thresholds: OptimalThresholds = None):
        self.thresholds = thresholds or OptimalThresholds()
    
    def detect(self, trial_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Detect overfitting signals.
        
        Returns:
            Dict with boolean flags for each overfitting signal
        """
        signals = {
            'train_val_gap_high': False,
            'violations_excessive': False,
            'symbolic_ratio_high': False,
            'confidence_too_low': False,
            'overall_overfitting': False,
        }
        
        # 1. Train-validation gap
        train_score = trial_data.get('train_score', 0)
        val_score = trial_data.get('val_score', 0)
        gap = abs(train_score - val_score)
        signals['train_val_gap_high'] = gap > self.thresholds.max_train_val_gap
        
        # 2. Excessive violations
        violations = trial_data.get('violation_percentage', 0)
        signals['violations_excessive'] = violations > self.thresholds.max_violation_percentage
        
        # 3. Symbolic ratio too high
        symbolic_ratio = trial_data.get('symbolic_ratio', 0)
        signals['symbolic_ratio_high'] = symbolic_ratio > self.thresholds.max_symbolic_ratio
        
        # 4. Confidence threshold too low (overfitting to training data)
        min_conf = trial_data.get('min_confidence_threshold', 0.1)
        signals['confidence_too_low'] = min_conf < 0.01
        
        # Overall: ANY signal triggered
        signals['overall_overfitting'] = any([
            signals['train_val_gap_high'],
            signals['violations_excessive'],
            signals['symbolic_ratio_high'],
            signals['confidence_too_low'],
        ])
        
        return signals
    
    def score_health(self, trial_data: Dict[str, Any]) -> float:
        """
        Calculate health score (0-1, higher is better).
        
        Penalizes:
        - High train-val gap
        - Excessive violations
        - Extreme symbolic ratios
        """
        score = 1.0
        
        # Penalty 1: Train-val gap
        train_score = trial_data.get('train_score', 0)
        val_score = trial_data.get('val_score', 0)
        gap = abs(train_score - val_score)
        if gap > self.thresholds.max_train_val_gap:
            score -= 0.3 * (gap / self.thresholds.max_train_val_gap - 1)
        
        # Penalty 2: Violations
        violations = trial_data.get('violation_percentage', 100)
        if violations > self.thresholds.max_violation_percentage:
            score -= 0.3 * (violations / self.thresholds.max_violation_percentage - 1)
        elif violations < self.thresholds.min_violation_percentage:
            score -= 0.2 * (self.thresholds.min_violation_percentage / violations - 1)
        
        # Penalty 3: Symbolic ratio
        symbolic_ratio = trial_data.get('symbolic_ratio', 0.7)
        if symbolic_ratio > self.thresholds.max_symbolic_ratio:
            score -= 0.2 * (symbolic_ratio - self.thresholds.max_symbolic_ratio)
        elif symbolic_ratio < self.thresholds.min_symbolic_ratio:
            score -= 0.2 * (self.thresholds.min_symbolic_ratio - symbolic_ratio)
        
        return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════════
# Visualization Generator
# ═══════════════════════════════════════════════════════════════════════════

class OptimizationVisualizer:
    """
    Generate comprehensive visualizations for hyperparameter optimization.
    
    Includes:
    - Convergence plots with ideal thresholds
    - Overfitting detection
    - Parameter importance
    - Pareto frontier
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.file_manager = FileManager()
        self.detector = OverfittingDetector(self.config.thresholds)
    
    def load_results(self, result_file: str | Path) -> Dict[str, Any]:
        """Load optimization results from JSON."""
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def plot_convergence_with_thresholds(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Plot optimization convergence with ideal threshold lines.
        
        Shows:
        - Best score per trial (line)
        - F1 score threshold (green zone)
        - Overfitting threshold (red zone)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Extract trial data
        history = results.get('optimization_history', [])
        if not history:
            logger.warning("No optimization history found")
            return None
        
        trials = [h['trial_number'] for h in history]
        scores = [h['scores']['combined'] for h in history]
        
        # Plot 1: Convergence with thresholds
        ax1.plot(trials, scores, 'o-', linewidth=2, markersize=6, 
                label='Trial Score', color='#2E86AB')
        
        # Best score line
        best_scores = np.maximum.accumulate(scores)
        ax1.plot(trials, best_scores, '--', linewidth=2, 
                label='Best Score', color='#A23B72')
        
        # Threshold zones
        th = self.config.thresholds
        
        # GREEN ZONE: Target performance
        ax1.axhspan(th.target_f1_score, th.max_f1_score, 
                   alpha=0.2, color='green', label='Target Zone')
        ax1.axhline(th.target_f1_score, color='green', linestyle='--', 
                   linewidth=1.5, label=f'Target F1: {th.target_f1_score}')
        
        # YELLOW ZONE: Acceptable performance
        ax1.axhspan(th.min_f1_score, th.target_f1_score, 
                   alpha=0.15, color='yellow')
        ax1.axhline(th.min_f1_score, color='orange', linestyle='--', 
                   linewidth=1, label=f'Min F1: {th.min_f1_score}')
        
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('F1 Score', fontsize=12)
        ax1.set_title('Optimization Convergence with Target Thresholds', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overfitting indicators
        violations = [h.get('violation_percentage', 100) for h in history]
        train_val_gaps = [
            abs(h.get('train_score', 0) - h.get('val_score', 0)) * 100
            for h in history
        ]
        
        ax2_twin = ax2.twinx()
        
        # Violations
        ax2.plot(trials, violations, 's-', linewidth=2, markersize=5,
                label='Violation %', color='#E63946')
        ax2.axhline(th.max_violation_percentage, color='red', linestyle='--',
                   linewidth=2, label=f'Max Violations: {th.max_violation_percentage}%')
        ax2.axhline(th.min_violation_percentage, color='blue', linestyle='--',
                   linewidth=1, label=f'Min Violations: {th.min_violation_percentage}%')
        
        # Train-val gap
        ax2_twin.plot(trials, train_val_gaps, '^-', linewidth=2, markersize=5,
                     label='Train-Val Gap %', color='#F77F00')
        ax2_twin.axhline(th.max_train_val_gap * 100, color='orange', 
                        linestyle=':', linewidth=2, 
                        label=f'Max Gap: {th.max_train_val_gap*100}%')
        
        ax2.set_xlabel('Trial Number', fontsize=12)
        ax2.set_ylabel('Violation Percentage', fontsize=12, color='#E63946')
        ax2_twin.set_ylabel('Train-Val Gap (%)', fontsize=12, color='#F77F00')
        ax2.set_title('Overfitting Indicators', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.output_dir / f"convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.success(f"📊 Convergence plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_parameter_importance(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Plot parameter importance with threshold indicators.
        
        Shows which parameters most affect F1 score.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract parameter correlations with score
        history = results.get('optimization_history', [])
        if not history:
            return None
        
        # Calculate correlation of each param with score
        param_importance = {}
        all_params = history[0]['params'].keys()
        
        for param in all_params:
            values = [h['params'].get(param, 0) for h in history]
            scores = [h['scores']['combined'] for h in history]
            
            # Handle categorical params
            if all(isinstance(v, (int, float)) for v in values):
                correlation = np.corrcoef(values, scores)[0, 1]
                param_importance[param] = abs(correlation)
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), 
                              key=lambda x: x[1], reverse=True)[:15]
        
        params = [p[0] for p in sorted_params]
        importances = [p[1] for p in sorted_params]
        
        # Color by parameter category
        colors = []
        for param in params:
            if 'xgb' in param:
                colors.append('#2E86AB')  # Blue for XGBoost
            elif 'anyburl' in param:
                colors.append('#A23B72')  # Purple for AnyBURL
            elif 'transe' in param:
                colors.append('#F77F00')  # Orange for TransE
            elif 'lgbm' in param:
                colors.append('#06A77D')  # Green for LightGBM
            else:
                colors.append('#E63946')  # Red for symbolic
        
        bars = ax.barh(params, importances, color=colors, alpha=0.7)
        
        # Add threshold line for significant importance
        ax.axvline(0.3, color='red', linestyle='--', linewidth=2,
                  label='Significance Threshold (0.3)')
        
        ax.set_xlabel('Absolute Correlation with F1 Score', fontsize=12)
        ax.set_ylabel('Hyperparameter', fontsize=12)
        ax.set_title('Parameter Importance Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importances)):
            ax.text(val + 0.01, i, f'{val:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.output_dir / f"param_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.success(f"📊 Parameter importance plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_overfitting_analysis(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Plot comprehensive overfitting analysis.
        
        Shows:
        - Health score per trial
        - Overfitting signals heatmap
        - Safe zone indicators
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        history = results.get('optimization_history', [])
        if not history:
            return None
        
        # Calculate health scores
        trials = [h['trial_number'] for h in history]
        health_scores = [self.detector.score_health(h) for h in history]
        f1_scores = [h['scores']['combined'] for h in history]
        
        # Plot 1: Health score vs F1 score
        scatter = ax1.scatter(f1_scores, health_scores, 
                            c=trials, cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black')
        
        # Safe zone (high F1 + high health)
        th = self.config.thresholds
        safe_zone = Rectangle(
            (th.min_f1_score, 0.7), 
            th.max_f1_score - th.min_f1_score, 
            0.3,
            alpha=0.2, facecolor='green', edgecolor='green',
            linewidth=2, linestyle='--',
            label='Safe Zone'
        )
        ax1.add_patch(safe_zone)
        
        # Overfitting zone (high F1 + low health)
        danger_zone = Rectangle(
            (th.target_f1_score, 0.0),
            th.max_f1_score - th.target_f1_score,
            0.5,
            alpha=0.2, facecolor='red', edgecolor='red',
            linewidth=2, linestyle='--',
            label='Danger Zone (Overfitting)'
        )
        ax1.add_patch(danger_zone)
        
        ax1.set_xlabel('F1 Score', fontsize=12)
        ax1.set_ylabel('Health Score (0-1)', fontsize=12)
        ax1.set_title('Overfitting Detection: F1 vs Health', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Trial Number', fontsize=10)
        
        # Plot 2: Overfitting signals heatmap
        signals_matrix = []
        signal_names = ['Train-Val Gap', 'Violations', 'Symbolic Ratio', 
                       'Low Confidence', 'Overall']
        
        for h in history:
            signals = self.detector.detect(h)
            signals_matrix.append([
                int(signals['train_val_gap_high']),
                int(signals['violations_excessive']),
                int(signals['symbolic_ratio_high']),
                int(signals['confidence_too_low']),
                int(signals['overall_overfitting']),
            ])
        
        signals_array = np.array(signals_matrix).T
        
        sns.heatmap(signals_array, 
                   ax=ax2,
                   cmap=['#90EE90', '#FF6B6B'],  # Green/Red
                   cbar_kws={'label': 'Signal Triggered'},
                   yticklabels=signal_names,
                   xticklabels=[f'T{i}' for i in trials[::max(1, len(trials)//20)]],
                   linewidths=0.5,
                   linecolor='gray')
        
        ax2.set_xlabel('Trial', fontsize=12)
        ax2.set_ylabel('Overfitting Signal', fontsize=12)
        ax2.set_title('Overfitting Signals Heatmap', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.output_dir / f"overfitting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.success(f"📊 Overfitting analysis plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_pareto_frontier(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Plot Pareto frontier: F1 score vs model complexity.
        
        Helps identify best trade-off between performance and simplicity.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        history = results.get('optimization_history', [])
        if not history:
            return None
        
        # Extract data
        f1_scores = [h['scores']['combined'] for h in history]
        complexities = [
            h['params'].get('xgb_n_estimators', 100) + 
            h['params'].get('xgb_max_depth', 3) * 10 +
            h['params'].get('lgbm_num_leaves', 5) * 5
            for h in history
        ]
        health_scores = [self.detector.score_health(h) for h in history]
        
        # Color by health
        scatter = ax.scatter(complexities, f1_scores, 
                           c=health_scores, cmap='RdYlGn',
                           s=150, alpha=0.6, edgecolors='black',
                           vmin=0, vmax=1)
        
        # Find Pareto frontier
        pareto_indices = []
        for i, (c, f) in enumerate(zip(complexities, f1_scores)):
            is_pareto = True
            for j, (c2, f2) in enumerate(zip(complexities, f1_scores)):
                if i != j and f2 >= f and c2 <= c and (f2 > f or c2 < c):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_indices.append(i)
        
        # Plot Pareto frontier
        pareto_c = [complexities[i] for i in pareto_indices]
        pareto_f = [f1_scores[i] for i in pareto_indices]
        
        # Sort for line
        sorted_pairs = sorted(zip(pareto_c, pareto_f))
        pareto_c_sorted = [p[0] for p in sorted_pairs]
        pareto_f_sorted = [p[1] for p in sorted_pairs]
        
        ax.plot(pareto_c_sorted, pareto_f_sorted, 'r--', 
               linewidth=2, label='Pareto Frontier', zorder=10)
        ax.scatter(pareto_c, pareto_f, 
                  s=200, facecolors='none', edgecolors='red',
                  linewidths=3, zorder=11)
        
        # Ideal zone (low complexity, high F1)
        th = self.config.thresholds
        ax.axhline(th.target_f1_score, color='green', linestyle='--',
                  linewidth=2, label=f'Target F1: {th.target_f1_score}')
        
        ax.set_xlabel('Model Complexity (Estimators + Depth + Leaves)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Pareto Frontier: Performance vs Complexity', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Health Score', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.output_dir / f"pareto_frontier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.success(f"📊 Pareto frontier plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def generate_all_plots(self, result_file: str | Path) -> List[Path]:
        """
        Generate all visualization plots.
        
        Returns:
            List of saved plot paths
        """
        logger.info(f"📊 Generating visualizations for {result_file}")
        
        results = self.load_results(result_file)
        
        plots = []
        plots.append(self.plot_convergence_with_thresholds(results))
        plots.append(self.plot_parameter_importance(results))
        plots.append(self.plot_overfitting_analysis(results))
        plots.append(self.plot_pareto_frontier(results))
        
        logger.success(f"✅ Generated {len(plots)} visualization plots")
        return [p for p in plots if p is not None]


# ═══════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate optimization visualizations with anti-overfitting validation'
    )
    parser.add_argument(
        'result_file',
        type=str,
        help='Path to optimization result JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plots instead of saving'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = VisualizationConfig(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        show_plots=args.show,
    )
    
    # Generate visualizations
    visualizer = OptimizationVisualizer(config)
    plots = visualizer.generate_all_plots(args.result_file)
    
    logger.success(f"✅ Generated {len(plots)} plots:")
    for plot in plots:
        logger.info(f"   📊 {plot}")


if __name__ == "__main__":
    main()
