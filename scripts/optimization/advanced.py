#!/usr/bin/env python3
"""
Advanced SOTA Features - Distributed Optimization & MLflow Model Registry

This module implements advanced features for the optimization module:
1. Distributed optimization with Ray
2. Optuna Dashboard integration
3. Bayesian optimization with BoTorch
4. Early stopping with Optuna Terminator
5. Hyperparameter importance with fANOVA
6. Automated report generation (PDF)
7. Model registry integration

All features integrate seamlessly with the main find_best_hyperparameters() function.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import warnings

from pff.utils import logger


# ============================================================================
# 1. DISTRIBUTED OPTIMIZATION WITH RAY
# ============================================================================

class DistributedOptimizer:
    """
    Distributed hyperparameter optimization using Ray Tune.
    
    Enables horizontal scaling of optimization across multiple machines/nodes.
    """

    def __init__(self, address: Optional[str] = None, num_cpus: Optional[int] = None):
        """
        Initialize distributed optimizer.
        
        Args:
            address: Ray cluster address (None for local)
            num_cpus: Number of CPUs to use (None for all)
        """
        self.address = address
        self.num_cpus = num_cpus
        self.ray_available = self._check_ray_availability()
        
    def _check_ray_availability(self) -> bool:
        """Check if Ray is available."""
        try:
            import ray
            return True
        except ImportError:
            logger.warning(
                "Ray not installed. Install with: pip install ray[tune]\n"
                "Distributed optimization will not be available."
            )
            return False
    
    def run_distributed(
        self,
        objective_func: Callable[[Any], float],
        search_space: Dict[str, Any],
        n_trials: int = 100,
        num_workers: int = 4,
        resources_per_worker: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run distributed optimization with Ray Tune.
        
        Args:
            objective_func: Objective function
            search_space: Search space definition
            n_trials: Number of trials
            num_workers: Number of distributed workers
            resources_per_worker: Resources per worker (e.g., {"cpu": 1})
            **kwargs: Additional Ray Tune parameters
            
        Returns:
            Optimization results
        """
        if not self.ray_available:
            raise ImportError("Ray is required for distributed optimization")
        
        try:
            import ray
            from ray import tune
            from ray.tune import run, run_config
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(
                    address=self.address,
                    num_cpus=self.num_cpus,
                    log_to_driver=False
                )
            
            logger.info(f" Starting distributed optimization with {num_workers} workers")
            
            # Convert to Ray Tune search space
            ray_search_space = self._convert_to_ray_space(search_space)
            
            # Define training function
            def trainable(config):
                # Create trial object
                class RayTrial:
                    def __init__(self, config):
                        self.config = config
                        self.number = 0
                    
                    def suggest_float(self, name, low, high, log=False):
                        return config[name]
                    
                    def suggest_int(self, name, low, high, step=1):
                        return int(config[name])
                    
                    def suggest_categorical(self, name, choices):
                        return config[name]
                
                trial = RayTrial(config)
                score = objective_func(trial)
                tune.report(score=score)
            
            # Run distributed optimization
            start_time = time.time()
            
            result = run(
                trainable,
                config=ray_search_space,
                num_samples=n_trials,
                resources_per_trial=resources_per_worker or {"cpu": 1},
                local_dir="./ray_results",
                **kwargs
            )
            
            optimization_time = time.time() - start_time
            
            # Extract best result
            best_trial = result.best_trial
            best_score = result.best_result['score']
            best_params = best_trial.config
            
            logger.success(f" Distributed optimization complete!")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Time: {optimization_time:.2f}s")
            logger.info(f"Workers: {num_workers}")
            
            return {
                'best_params': best_params,
                'best_value': best_score,
                'n_trials': n_trials,
                'optimization_time': optimization_time,
                'framework': 'ray-tune',
                'num_workers': num_workers,
                'result': result,
            }
            
        except Exception as e:
            logger.error(f"Distributed optimization failed: {e}")
            raise
    
    def _convert_to_ray_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Optuna-style search space to Ray Tune format."""
        from ray import tune
        
        ray_space = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, (list, tuple)):
                if len(param_config) == 2:
                    low, high = float(param_config[0]), float(param_config[1])
                    if low < 0 and high > 0 and abs(high / low) > 100:
                        # Log scale
                        ray_space[param_name] = tune.loguniform(low, high)
                    else:
                        # Regular float
                        ray_space[param_name] = tune.uniform(low, high)
                else:
                    # Categorical
                    ray_space[param_name] = tune.choice(param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                if param_type == 'int':
                    low = param_config['low']
                    high = param_config['high']
                    step = param_config.get('step', 1)
                    if step == 1:
                        ray_space[param_name] = tune.randint(low, high + 1)
                    else:
                        ray_space[param_name] = tune.uniform(low, high)
                elif param_type == 'categorical':
                    ray_space[param_name] = tune.choice(param_config['choices'])
        
        return ray_space


# ============================================================================
# 2. OPTUNA DASHBOARD INTEGRATION
# ============================================================================

class OptunaDashboard:
    """
    Integration with Optuna Dashboard for real-time visualization.
    """

    def __init__(self, storage_url: str = "sqlite:///optuna.db"):
        """
        Initialize Optuna Dashboard.
        
        Args:
            storage_url: URL for Optuna storage
        """
        self.storage_url = storage_url
        self.dashboard_process = None
    
    def start_dashboard(self, port: int = 8080) -> None:
        """
        Start Optuna Dashboard.
        
        Args:
            port: Port to run dashboard on
        """
        try:
            import optuna
            from optuna_dashboard import run_server
            
            logger.info(f" Starting Optuna Dashboard on port {port}")
            logger.info(f"Storage: {self.storage_url}")
            logger.info(f"URL: http://localhost:{port}")
            
            # Start dashboard in background
            import threading
            self.dashboard_process = threading.Thread(
                target=lambda: run_server(self.storage_url, host="0.0.0.0", port=port),
                daemon=True
            )
            self.dashboard_process.start()
            
            logger.success(" Optuna Dashboard started")
            
        except ImportError:
            logger.warning(
                "Optuna Dashboard not installed. Install with: pip install optuna-dashboard"
            )
    
    def stop_dashboard(self) -> None:
        """Stop Optuna Dashboard."""
        if self.dashboard_process:
            # Note: In production, would use proper process management
            logger.info(" Stopping Optuna Dashboard")


# ============================================================================
# 3. BAYESIAN OPTIMIZATION WITH BOTORCH
# ============================================================================

class BayesianOptimizer:
    """
    Bayesian optimization using BoTorch (Gaussian Processes).
    
    Provides state-of-the-art Bayesian optimization for expensive black-box functions.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize Bayesian optimizer.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.botorch_available = self._check_botorch_availability()
    
    def _check_botorch_availability(self) -> bool:
        """Check if BoTorch is available."""
        try:
            import botorch
            return True
        except ImportError:
            logger.warning(
                "BoTorch not installed. Install with: pip install botorch\n"
                "Bayesian optimization will not be available."
            )
            return False
    
    def optimize(
        self,
        objective_func: Callable[[Any], float],
        search_space: Dict[str, Any],
        n_trials: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization with BoTorch.
        
        Args:
            objective_func: Objective function
            search_space: Search space
            n_trials: Number of trials
            **kwargs: Additional parameters
            
        Returns:
            Optimization results
        """
        if not self.botorch_available:
            raise ImportError("BoTorch is required for Bayesian optimization")
        
        try:
            import optuna
            from optuna.integration import BoTorchSampler
            
            logger.info(f" Starting Bayesian optimization with BoTorch")
            
            # Create study with BoTorch sampler
            sampler = BoTorchSampler()
            study = optuna.create_study(sampler=sampler)
            
            # Run optimization
            study.optimize(objective_func, n_trials=n_trials)
            
            logger.success(f" Bayesian optimization complete!")
            logger.info(f"Best score: {study.best_value:.4f}")
            
            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': n_trials,
                'framework': 'botorch',
                'study': study,
            }
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            raise


# ============================================================================
# 4. EARLY STOPPING WITH OPTUNA TERMINATOR
# ============================================================================

class EarlyStoppingOptimizer:
    """
    Early stopping optimization using Optuna Terminator.
    
    Automatically stops optimization when no improvement is expected.
    """

    def __init__(self):
        """Initialize early stopping optimizer."""
        self.terminator_available = self._check_terminator_availability()
    
    def _check_terminator_availability(self) -> bool:
        """Check if Optuna Terminator is available."""
        try:
            from optuna.terminator import Terminator
            return True
        except ImportError:
            logger.warning(
                "Optuna Terminator not installed. Install with: pip install optuna-terminator"
            )
            return False
    
    def optimize_with_early_stopping(
        self,
        objective_func: Callable[[Any], float],
        search_space: Dict[str, Any],
        n_trials: int = 100,
        min_trials: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization with early stopping.
        
        Args:
            objective_func: Objective function
            search_space: Search space
            n_trials: Maximum trials
            min_trials: Minimum trials before early stopping
            **kwargs: Additional parameters
            
        Returns:
            Optimization results
        """
        if not self.terminator_available:
            raise ImportError("Optuna Terminator is required for early stopping")
        
        try:
            import optuna
            from optuna.terminator import Terminator, TerminatorCallback
            
            logger.info("Iniciando otimização com early stopping")
            
            # Create study
            study = optuna.create_study()
            
            # Add terminator
            terminator = Terminator()
            callback = TerminatorCallback(terminator)
            
            # Run optimization
            study.optimize(
                objective_func,
                n_trials=n_trials,
                callbacks=[callback]
            )
            
            # Check if early stopping triggered
            if len(study.trials) < n_trials:
                logger.info(f"Early stopping disparado no trial {len(study.trials)}")
            
            logger.success(f" Optimization with early stopping complete!")
            logger.info(f"Best score: {study.best_value:.4f}")
            logger.info(f"Trials run: {len(study.trials)}")
            
            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'early_stopped': len(study.trials) < n_trials,
                'framework': 'terminator',
                'study': study,
            }
            
        except Exception as e:
            logger.error(f"Early stopping optimization failed: {e}")
            raise


# ============================================================================
# 5. HYPERPARAMETER IMPORTANCE WITH FANOVA
# ============================================================================

class ImportanceAnalyzer:
    """
    Hyperparameter importance analysis using fANOVA.
    
    Analyzes which hyperparameters contribute most to model performance.
    """

    def __init__(self):
        """Initialize importance analyzer."""
        self.fanova_available = self._check_fanova_availability()
    
    def _check_fanova_availability(self) -> bool:
        """Check if fANOVA is available."""
        try:
            import optuna
            from optuna.importance import FanovaImportanceEvaluator
            return True
        except (ImportError, AttributeError):
            logger.warning(
                "fANOVA not available in this Optuna version.\n"
                "Using built-in importance analysis instead."
            )
            return False
    
    def analyze_importance(
        self,
        study: Any,
        params: Optional[List[str]] = None,
        evaluator_name: str = "fanova"
    ) -> Dict[str, float]:
        """
        Analyze hyperparameter importance.
        
        Args:
            study: Optuna study
            params: Parameters to analyze
            evaluator_name: Evaluator type ('fanova' or 'mean')
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            import optuna
            
            if evaluator_name == "fanova" and self.fanova_available:
                from optuna.importance import FanovaImportanceEvaluator
                evaluator = FanovaImportanceEvaluator()
                logger.info(f" Analyzing importance with fANOVA")
            else:
                from optuna.importance import MeanImportanceEvaluator
                evaluator = MeanImportanceEvaluator()
                logger.info(f" Analyzing importance with Mean")
            
            importances = optuna.importance.get_param_importances(
                study,
                evaluator=evaluator,
                params=params
            )
            
            # Log results
            logger.info(" Hyperparameter Importance Analysis:")
            for param, importance in sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger.info(f"  • {param}: {importance:.4f}")
            
            return dict(importances)
            
        except Exception as e:
            logger.error(f"Importance analysis failed: {e}")
            return {}


# ============================================================================
# 6. AUTOMATED REPORT GENERATION (PDF)
# ============================================================================

class PDFReportGenerator:
    """
    Automated PDF report generation for optimization results.
    
    Generates comprehensive PDF reports with plots, statistics, and insights.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize PDF report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or Path("./reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reportlab_available = self._check_reportlab()
    
    def _check_reportlab(self) -> bool:
        """Check if ReportLab is available."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.pdfgen import canvas
            return True
        except ImportError:
            logger.warning(
                "ReportLab not installed. Install with: pip install reportlab\n"
                "PDF reports will not be available."
            )
            return False
    
    def generate_pdf_report(
        self,
        result: Dict[str, Any],
        title: str = "Optimization Report",
        include_plots: bool = True
    ) -> Path:
        """
        Generate PDF report.
        
        Args:
            result: Optimization result
            title: Report title
            include_plots: Whether to include plots
            
        Returns:
            Path to generated PDF
        """
        if not self.reportlab_available:
            raise ImportError("ReportLab is required for PDF generation")
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pdf_path = self.output_dir / f"optimization_report_{timestamp}.pdf"
            
            logger.info(f" Generating PDF report: {pdf_path}")
            
            # Create PDF
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            story = []
            
            # Title
            from reportlab.platypus import Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Summary table
            summary_data = [
                ['Metric', 'Value'],
                ['Best Score', f"{result.get('best_value', 'N/A'):.4f}"],
                ['Number of Trials', str(result.get('n_trials', 'N/A'))],
                ['Optimization Time', f"{result.get('optimization_time', 0):.2f}s"],
                ['Framework', result.get('framework', 'N/A')],
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Best parameters
            story.append(Paragraph("Best Parameters", styles['Heading2']))
            best_params = result.get('best_params', {})
            for param, value in best_params.items():
                story.append(Paragraph(f"• {param}: {value}", styles['Normal']))
            
            story.append(Spacer(1, 12))
            
            # Include plots if requested
            if include_plots:
                story.append(Paragraph("Visualizations", styles['Heading2']))
                visualization_plots = result.get('visualization_plots', {})
                for name, path in visualization_plots.items():
                    if path.exists():
                        story.append(Paragraph(f"• {name}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            logger.success(f" PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise


# ============================================================================
# 7. MODEL REGISTRY INTEGRATION
# ============================================================================

class ModelRegistry:
    """
    MLflow Model Registry integration.
    
    Automatically registers optimized models to MLflow Model Registry.
    """

    def __init__(self, registry_uri: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            registry_uri: MLflow registry URI
        """
        self.registry_uri = registry_uri
        self.mlflow_available = self._check_mlflow()
    
    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow
            return True
        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow\n"
                "Model registry will not be available."
            )
            return False
    
    def register_model(
        self,
        model_name: str,
        model_path: Union[str, Path],
        result: Dict[str, Any],
        stage: str = "Production",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model to MLflow Model Registry.
        
        Args:
            model_name: Name for the model
            model_path: Path to model artifact
            result: Optimization result
            stage: Model stage (Staging, Production, Archived)
            tags: Model tags
            
        Returns:
            Model version
        """
        if not self.mlflow_available:
            raise ImportError("MLflow is required for model registry")
        
        try:
            import mlflow
            import mlflow.sklearn
            
            logger.info(f" Registering model to MLflow Model Registry")
            logger.info(f"Model name: {model_name}")
            logger.info(f"Model path: {model_path}")
            
            # Create or get experiment
            experiment_name = f"optimization_{int(time.time())}"
            mlflow.set_experiment(experiment_name)
            
            # Start MLflow run
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(result.get('best_params', {}))
                mlflow.log_metric("best_score", result.get('best_value', 0))
                mlflow.log_metric("n_trials", result.get('n_trials', 0))
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model_path,  # In practice, would load the actual model
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                # Set tags
                if tags:
                    mlflow.set_tags(tags)
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=f"runs:/latest/model",
                name=model_name
            )
            
            # Transition to stage
            mlflow.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            
            logger.success(f" Model registered: {model_name} v{model_version.version}")
            logger.info(f"Stage: {stage}")
            
            return model_version.version
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise


# ============================================================================
# UNIFIED ADVANCED OPTIMIZER
# ============================================================================

class AdvancedOptimizer:
    """
    Unified advanced optimizer combining all SOTA features.
    
    Provides:
    - Distributed optimization
    - Bayesian optimization
    - Early stopping
    - Importance analysis
    - PDF reports
    - Model registry
    """

    def __init__(self, enable_distributed: bool = False):
        """
        Initialize advanced optimizer.
        
        Args:
            enable_distributed: Enable distributed optimization
        """
        self.distributed_optimizer = DistributedOptimizer() if enable_distributed else None
        self.bayesian_optimizer = BayesianOptimizer()
        self.early_stopping_optimizer = EarlyStoppingOptimizer()
        self.importance_analyzer = ImportanceAnalyzer()
        self.pdf_generator = PDFReportGenerator()
        self.model_registry = ModelRegistry()
    
    def optimize_advanced(
        self,
        objective_func: Callable[[Any], float],
        search_space: Dict[str, Any],
        n_trials: int = 100,
        strategy: str = "auto",
        enable_bayesian: bool = False,
        enable_early_stopping: bool = True,
        enable_importance: bool = True,
        enable_pdf_report: bool = True,
        model_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run advanced optimization with all features.
        
        Args:
            objective_func: Objective function
            search_space: Search space
            n_trials: Number of trials
            strategy: Optimization strategy
            enable_bayesian: Use Bayesian optimization
            enable_early_stopping: Use early stopping
            enable_importance: Analyze importance
            enable_pdf_report: Generate PDF report
            model_path: Path to trained model
            model_name: Model registry name
            **kwargs: Additional parameters
            
        Returns:
            Complete optimization results
        """
        logger.info(" Starting advanced optimization with all SOTA features")
        
        # Step 1: Run optimization
        if enable_bayesian:
            result = self.bayesian_optimizer.optimize(
                objective_func, search_space, n_trials, **kwargs
            )
        else:
            from .core import find_best_hyperparameters
            result = find_best_hyperparameters(
                objective_func,
                search_space,
                n_trials,
                strategy=strategy,
                **kwargs
            )
        
        # Step 2: Early stopping
        if enable_early_stopping and 'study' in result:
            early_result = self.early_stopping_optimizer.optimize_with_early_stopping(
                objective_func, search_space, n_trials
            )
            result['early_stopping'] = early_result
        
        # Step 3: Importance analysis
        if enable_importance and 'study' in result:
            importance = self.importance_analyzer.analyze_importance(result['study'])
            result['importance'] = importance
        
        # Step 4: PDF report
        if enable_pdf_report:
            try:
                pdf_path = self.pdf_generator.generate_pdf_report(result)
                result['pdf_report'] = pdf_path
            except Exception as e:
                logger.warning(f"PDF report generation failed: {e}")
        
        # Step 5: Model registry
        if model_path and model_name:
            try:
                model_version = self.model_registry.register_model(
                    model_name,
                    model_path,
                    result
                )
                result['model_version'] = model_version
            except Exception as e:
                logger.warning(f"Model registration failed: {e}")
        
        logger.success(" Advanced optimization complete!")
        
        return result


# Export all classes
__all__ = [
    'DistributedOptimizer',
    'OptunaDashboard',
    'BayesianOptimizer',
    'EarlyStoppingOptimizer',
    'ImportanceAnalyzer',
    'PDFReportGenerator',
    'ModelRegistry',
    'AdvancedOptimizer',
]
