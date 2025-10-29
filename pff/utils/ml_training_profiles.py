"""
ML Training Profiles - Hardware-aware configurations for TransE, AnyBURL, and LightGBM.

This module provides safe training configurations based on detected hardware to prevent OOM errors.
Different profiles for low_spec, mid_spec, and high_spec machines.

Author: PFF Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Literal

from pff.utils.hardware_detector import HardwareDetector, HardwareProfile


@dataclass
class TransETrainingConfig:
    """TransE training configuration optimized for hardware."""

    embedding_dim: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    margin: float
    num_workers: int
    use_gpu: bool
    negative_samples: int
    max_entities: int | None  # Limit entities for low-spec machines


@dataclass
class AnyBURLConfig:
    """AnyBURL rule learning configuration optimized for hardware."""

    max_memory_gb: int
    num_threads: int
    max_rule_length: int
    max_rules: int
    timeout_seconds: int


@dataclass
class LightGBMConfig:
    """LightGBM training configuration optimized for hardware."""

    num_leaves: int
    max_depth: int
    num_threads: int
    max_bin: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float


@dataclass
class MLTrainingProfile:
    """Complete ML training profile for all models."""

    machine_name: str
    transe: TransETrainingConfig
    anyburl: AnyBURLConfig
    lightgbm: LightGBMConfig
    ray_num_cpus: int
    ray_object_store_memory_gb: int | None

    def get_warnings(self) -> list[str]:
        """Get warnings about training limitations for this profile."""
        warnings = []

        if self.machine_name == "low_spec":
            warnings.append("⚠️  LOW_SPEC: Treinamento limitado a 50k entidades máximo")
            warnings.append("⚠️  LOW_SPEC: AnyBURL limitado a 6GB RAM (evitar datasets grandes)")
            warnings.append("⚠️  LOW_SPEC: TransE em CPU apenas (sem GPU detectada)")
            warnings.append("⚠️  LOW_SPEC: Recomendado usar apenas para testes pequenos")

        elif self.machine_name == "mid_spec":
            warnings.append("ℹ️  MID_SPEC: Adequado para testes e desenvolvimento")
            warnings.append("ℹ️  MID_SPEC: Treinamento completo pode levar 2-4x mais tempo que high_spec")
            warnings.append("ℹ️  MID_SPEC: AnyBURL limitado a 10GB RAM")
            if not HardwareDetector.detect().has_gpu:
                warnings.append("⚠️  MID_SPEC: TransE em CPU (sem GPU detectada) - espere treinamento lento")

        elif self.machine_name == "high_spec":
            warnings.append("✅ HIGH_SPEC: Configuração completa para produção")
            warnings.append("✅ HIGH_SPEC: GPU detectada - treinamento TransE será 10-50x mais rápido")

        return warnings


class MLTrainingProfileGenerator:
    """Generate ML training profiles based on hardware detection."""

    @staticmethod
    def generate(profile: HardwareProfile) -> MLTrainingProfile:
        """
        Generate optimal ML training configuration for detected hardware.

        Args:
            profile: Hardware profile from HardwareDetector.

        Returns:
            MLTrainingProfile: Safe training configuration to prevent OOM.
        """
        if profile.machine_name == "high_spec":
            return MLTrainingProfileGenerator._generate_high_spec(profile)
        elif profile.machine_name == "mid_spec":
            return MLTrainingProfileGenerator._generate_mid_spec(profile)
        else:
            return MLTrainingProfileGenerator._generate_low_spec(profile)

    @staticmethod
    def _generate_low_spec(profile: HardwareProfile) -> MLTrainingProfile:
        """
        Low-spec profile: 8GB RAM or less, no GPU.
        SAFE for small tests only - will prevent OOM on large datasets.
        """
        return MLTrainingProfile(
            machine_name="low_spec",
            transe=TransETrainingConfig(
                embedding_dim=64,  # Reduced from 128 (4x less memory)
                batch_size=256,  # Small batch to prevent OOM
                num_epochs=20,  # Reduced from 100
                learning_rate=0.001,
                margin=1.0,
                num_workers=2,  # Limit parallel workers
                use_gpu=False,  # No GPU
                negative_samples=10,  # Reduced from 50
                max_entities=50_000,  # Hard limit to prevent OOM
            ),
            anyburl=AnyBURLConfig(
                max_memory_gb=6,  # Leave 2GB for OS
                num_threads=min(4, profile.cpu_threads),
                max_rule_length=2,  # Reduced from 3
                max_rules=500,  # Reduced from 1000
                timeout_seconds=1800,  # 30min (reduced from 1h)
            ),
            lightgbm=LightGBMConfig(
                num_leaves=31,  # Small trees
                max_depth=6,  # Shallow
                num_threads=min(4, profile.cpu_threads),
                max_bin=128,  # Reduced from 255
                min_data_in_leaf=50,
                feature_fraction=0.8,
                bagging_fraction=0.8,
            ),
            ray_num_cpus=min(4, profile.cpu_threads),
            ray_object_store_memory_gb=2,  # 2GB object store
        )

    @staticmethod
    def _generate_mid_spec(profile: HardwareProfile) -> MLTrainingProfile:
        """
        Mid-spec profile: 12-16GB RAM, 12 threads, possibly no GPU (WSL).
        SUITABLE for development and testing - reasonable performance.
        """
        return MLTrainingProfile(
            machine_name="mid_spec",
            transe=TransETrainingConfig(
                embedding_dim=128,  # Full dimension
                batch_size=512,  # Moderate batch size
                num_epochs=50,  # Reduced from 100 for faster iteration
                learning_rate=0.001,
                margin=1.0,
                num_workers=4,  # Moderate parallelism
                use_gpu=profile.has_gpu,  # Use GPU if available
                negative_samples=25,  # Moderate sampling
                max_entities=200_000,  # Limit for safety
            ),
            anyburl=AnyBURLConfig(
                max_memory_gb=10,  # Leave 2-6GB for OS/other processes
                num_threads=min(8, profile.cpu_threads),
                max_rule_length=3,
                max_rules=1000,
                timeout_seconds=3600,  # 1h
            ),
            lightgbm=LightGBMConfig(
                num_leaves=63,  # Moderate trees
                max_depth=8,
                num_threads=min(8, profile.cpu_threads),
                max_bin=255,
                min_data_in_leaf=20,
                feature_fraction=0.9,
                bagging_fraction=0.9,
            ),
            ray_num_cpus=min(8, profile.cpu_threads),
            ray_object_store_memory_gb=4,  # 4GB object store
        )

    @staticmethod
    def _generate_high_spec(profile: HardwareProfile) -> MLTrainingProfile:
        """
        High-spec profile: 32GB RAM, 8-16 cores, RTX 3070 Ti (8GB VRAM).
        PRODUCTION configuration - full training capability.
        """
        return MLTrainingProfile(
            machine_name="high_spec",
            transe=TransETrainingConfig(
                embedding_dim=256,  # Large embeddings for better quality
                batch_size=2048,  # Large batch for GPU efficiency
                num_epochs=100,  # Full training
                learning_rate=0.001,
                margin=1.0,
                num_workers=8,  # Full parallelism
                use_gpu=True,  # GPU required for high_spec
                negative_samples=50,  # Full sampling
                max_entities=None,  # No limit
            ),
            anyburl=AnyBURLConfig(
                max_memory_gb=20,  # Use most available RAM (leave 12GB for OS)
                num_threads=min(16, profile.cpu_threads),
                max_rule_length=3,
                max_rules=2000,
                timeout_seconds=7200,  # 2h
            ),
            lightgbm=LightGBMConfig(
                num_leaves=127,  # Large trees
                max_depth=10,
                num_threads=min(16, profile.cpu_threads),
                max_bin=255,
                min_data_in_leaf=10,
                feature_fraction=0.9,
                bagging_fraction=0.9,
            ),
            ray_num_cpus=min(16, profile.cpu_threads),
            ray_object_store_memory_gb=8,  # 8GB object store
        )


def get_ml_training_profile() -> MLTrainingProfile:
    """
    Convenience function to get ML training profile for current hardware.

    Returns:
        MLTrainingProfile: Safe training configuration.
    """
    hardware_profile = HardwareDetector.detect()
    return MLTrainingProfileGenerator.generate(hardware_profile)


def print_ml_training_info():
    """Print ML training configuration (for debugging/info)."""
    hardware_profile = HardwareDetector.detect()
    ml_profile = MLTrainingProfileGenerator.generate(hardware_profile)

    print("🤖 ML Training Profile")
    print("─" * 60)
    print(f"Machine Type: {ml_profile.machine_name.upper()}")
    print(f"Hardware: {hardware_profile.total_ram_gb:.1f} GB RAM, "
          f"{hardware_profile.cpu_threads} threads, "
          f"GPU: {'Yes' if hardware_profile.has_gpu else 'No'}")

    print("\n⚙️  TransE Configuration")
    print("─" * 60)
    transe = ml_profile.transe
    print(f"Embedding Dimension:     {transe.embedding_dim}")
    print(f"Batch Size:              {transe.batch_size}")
    print(f"Num Epochs:              {transe.num_epochs}")
    print(f"Learning Rate:           {transe.learning_rate}")
    print(f"Negative Samples:        {transe.negative_samples}")
    print(f"Max Entities:            {transe.max_entities if transe.max_entities else 'Unlimited'}")
    print(f"Use GPU:                 {transe.use_gpu}")
    print(f"Num Workers:             {transe.num_workers}")

    print("\n⚙️  AnyBURL Configuration")
    print("─" * 60)
    anyburl = ml_profile.anyburl
    print(f"Max Memory:              {anyburl.max_memory_gb} GB")
    print(f"Num Threads:             {anyburl.num_threads}")
    print(f"Max Rule Length:         {anyburl.max_rule_length}")
    print(f"Max Rules:               {anyburl.max_rules}")
    print(f"Timeout:                 {anyburl.timeout_seconds}s ({anyburl.timeout_seconds // 60}min)")

    print("\n⚙️  LightGBM Configuration")
    print("─" * 60)
    lgbm = ml_profile.lightgbm
    print(f"Num Leaves:              {lgbm.num_leaves}")
    print(f"Max Depth:               {lgbm.max_depth}")
    print(f"Num Threads:             {lgbm.num_threads}")
    print(f"Max Bin:                 {lgbm.max_bin}")

    print("\n⚙️  Ray Configuration")
    print("─" * 60)
    print(f"Num CPUs:                {ml_profile.ray_num_cpus}")
    print(f"Object Store Memory:     {ml_profile.ray_object_store_memory_gb} GB")

    # Print warnings
    warnings = ml_profile.get_warnings()
    if warnings:
        print("\n⚠️  Training Warnings")
        print("─" * 60)
        for warning in warnings:
            print(warning)


if __name__ == "__main__":
    print_ml_training_info()
