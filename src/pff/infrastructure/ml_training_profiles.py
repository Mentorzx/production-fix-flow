"""ML Training Profiles - Hardware-aware configurations for DSLFM."""

from dataclasses import dataclass

from pff.shared.system.resource_manager import HardwareDetector, HardwareProfile
from pff.shared import logger


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
    max_entities: int | None


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
            warnings.append("LOW_SPEC: Training limited to 50k entities max")
            warnings.append("LOW_SPEC: DSLFM on CPU only (no GPU detected)")
            warnings.append("LOW_SPEC: Recommended for small tests only")

        elif self.machine_name == "mid_spec":
            warnings.append("MID_SPEC: Suitable for testing and development")
            warnings.append("MID_SPEC: Full training may take 2-4x longer than high_spec")
            if not HardwareDetector.detect().has_gpu:
                warnings.append("MID_SPEC: DSLFM on CPU (no GPU detected) - expect slow training")

        elif self.machine_name == "high_spec":
            warnings.append("HIGH_SPEC: Full production configuration")
            warnings.append("HIGH_SPEC: GPU detected - DSLFM training will be 10-50x faster")

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
        machine_name = getattr(profile, "machine_name", "low_spec")
        if machine_name == "high_spec":
            return MLTrainingProfileGenerator._generate_high_spec(profile)
        elif machine_name == "mid_spec":
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
                embedding_dim=64,
                batch_size=256,
                num_epochs=20,
                learning_rate=0.001,
                margin=1.0,
                num_workers=2,
                use_gpu=False,
                negative_samples=10,
                max_entities=50_000,
            ),
            ray_num_cpus=min(4, profile.cpu_threads),
            ray_object_store_memory_gb=2,
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
                embedding_dim=128,
                batch_size=512,
                num_epochs=50,
                learning_rate=0.001,
                margin=1.0,
                num_workers=4,
                use_gpu=profile.has_gpu,
                negative_samples=25,
                max_entities=200_000,
            ),
            ray_num_cpus=min(8, profile.cpu_threads),
            ray_object_store_memory_gb=4,
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
                embedding_dim=256,
                batch_size=2048,
                num_epochs=100,
                learning_rate=0.001,
                margin=1.0,
                num_workers=8,
                use_gpu=True,
                negative_samples=50,
                max_entities=None,
            ),
            ray_num_cpus=min(16, profile.cpu_threads),
            ray_object_store_memory_gb=8,
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

    warnings = ml_profile.get_warnings()
    for warning in warnings:
        logger.warning(warning)

    logger.info(f"Perfil ML configurado: {ml_profile.machine_name.upper()}")


if __name__ == "__main__":
    print_ml_training_info()
