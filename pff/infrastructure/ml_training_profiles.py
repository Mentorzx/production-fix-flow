"""ML Training Profiles - Hardware-aware configurations for DSLFM."""

from dataclasses import dataclass

from pff.shared import logger

from pff.infrastructure.hardware_detector import HardwareDetector, HardwareProfile


@dataclass
class DSLFMTrainingConfig:
    """DSLFM training configuration optimized for hardware."""

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
class MLTrainingProfile:
    """Complete ML training profile for all models."""

    machine_name: str
    dslfm: DSLFMTrainingConfig
    ray_num_cpus: int
    ray_object_store_memory_gb: int | None

    def get_warnings(self) -> list[str]:
        """Get warnings about training limitations for this profile."""
        warnings = []

        if self.machine_name == "low_spec":
            warnings.append("  LOW_SPEC: Treinamento limitado a 50k entidades máximo")
            warnings.append("  LOW_SPEC: DSLFM em CPU apenas (sem GPU detectada)")
            warnings.append("  LOW_SPEC: Recomendado usar apenas para testes pequenos")

        elif self.machine_name == "mid_spec":
            warnings.append("MID_SPEC: adequado para testes e desenvolvimento")
            warnings.append(
                "MID_SPEC: treinamento completo pode levar 2-4x mais tempo que high_spec"
            )
            if not HardwareDetector.detect().has_gpu:
                warnings.append(
                    "  MID_SPEC: DSLFM em CPU (sem GPU detectada) - espere treinamento lento"
                )

        elif self.machine_name == "high_spec":
            warnings.append(" HIGH_SPEC: Configuração completa para produção")
            warnings.append(
                " HIGH_SPEC: GPU detectada - treinamento DSLFM será 10-50x mais rápido"
            )

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
            dslfm=DSLFMTrainingConfig(
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
            dslfm=DSLFMTrainingConfig(
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
            dslfm=DSLFMTrainingConfig(
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

    # All config details should be debug level
    logger.debug("ML Training Profile")
    logger.debug(f"Machine Type: {ml_profile.machine_name.upper()}")
    logger.debug(
        f"Hardware: {hardware_profile.total_ram_gb:.1f} GB RAM, "
        f"{hardware_profile.cpu_threads} threads, "
        f"GPU: {'Yes' if hardware_profile.has_gpu else 'No'}"
    )

    dslfm = ml_profile.dslfm
    logger.debug(
        f"DSLFM config: dim={dslfm.embedding_dim}, batch={dslfm.batch_size}, "
        f"epochs={dslfm.num_epochs}, lr={dslfm.learning_rate}"
    )

    logger.debug(
        f"Ray config: cpus={ml_profile.ray_num_cpus}, object_store={ml_profile.ray_object_store_memory_gb}GB"
    )

    # Print warnings at warning level
    warnings = ml_profile.get_warnings()
    for warning in warnings:
        logger.warning(warning)

    # Summary at info level
    logger.info(f"Perfil ML configurado: {ml_profile.machine_name.upper()}")


if __name__ == "__main__":
    print_ml_training_info()
