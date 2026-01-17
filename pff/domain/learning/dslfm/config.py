from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.shared.core.config import MODELS_DIR
from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.shared.core.logger import logger

DSLFM_CONFIG_PATH = MODELS_DIR / "dslfm.yaml"


@dataclass
class DSLFMConfig:
    """Legacy DSLFM config (pre-KGC). Note: DSLFM-KGC uses DSLFMKGCConfig."""

    embedding_dim: int = 128
    lambda_logic: float = 0.0
    t_norm: str = "product"
    attr_hidden_dim: int = 64
    use_rule_support: bool = False
    pruning_threshold: float = 0.01
    grow_noise: float = 0.01
    lambda_pc: float = 0.0
    rebuild_every: int = 0
    smoothing_epsilon: float = 1e-6
    max_circuit_depth: int | None = None
    migration_mode: str = "late_fusion"


class DSLFMConfigBuilder:
    """Fluent builder for DSLFMConfig."""

    def __init__(self, config: DSLFMConfig | None = None) -> None:
        self._config = config or DSLFMConfig()

    def with_embedding_dim(self, value: int) -> DSLFMConfigBuilder:
        self._config.embedding_dim = int(value)
        return self

    def with_lambda_logic(self, value: float) -> DSLFMConfigBuilder:
        self._config.lambda_logic = float(value)
        return self

    def with_t_norm(self, value: str) -> DSLFMConfigBuilder:
        self._config.t_norm = str(value)
        return self

    def with_attr_hidden_dim(self, value: int) -> DSLFMConfigBuilder:
        self._config.attr_hidden_dim = int(value)
        return self

    def with_pc_settings(
        self,
        *,
        pruning_threshold: float | None = None,
        grow_noise: float | None = None,
        lambda_pc: float | None = None,
        rebuild_every: int | None = None,
        max_circuit_depth: int | None = None,
    ) -> DSLFMConfigBuilder:
        if pruning_threshold is not None:
            self._config.pruning_threshold = float(pruning_threshold)
        if grow_noise is not None:
            self._config.grow_noise = float(grow_noise)
        if lambda_pc is not None:
            self._config.lambda_pc = float(lambda_pc)
        if rebuild_every is not None:
            self._config.rebuild_every = int(rebuild_every)
        if max_circuit_depth is not None:
            self._config.max_circuit_depth = int(max_circuit_depth)
        return self

    def with_smoothing_epsilon(self, value: float) -> DSLFMConfigBuilder:
        self._config.smoothing_epsilon = float(value)
        return self

    def with_migration_mode(self, value: str) -> DSLFMConfigBuilder:
        self._config.migration_mode = str(value)
        return self

    def build(self) -> DSLFMConfig:
        return self._config

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> DSLFMConfigBuilder:
        return cls(load_dslfm_config(path))


def load_dslfm_config(path: Path | None = None) -> DSLFMConfig:
    cfg_path = path or DSLFM_CONFIG_PATH
    file_manager = FileManager()
    raw: dict[str, Any] = {}
    if file_manager.exists(cfg_path):
        try:
            payload = file_manager.read(cfg_path)
            raw = (
                payload.to_native()
                if isinstance(payload, ParquetBundle)
                else payload or {}
            )
            logger.debug(f"DSLFM config loaded from {cfg_path}")
        except Exception as exc:
            logger.warning(f"Failed to load DSLFM config: {exc}")
    else:
        logger.debug(f"DSLFM config not found at {cfg_path}; using defaults")

    model_raw = raw.get("model", {})
    pc_raw = raw.get("pc", {})
    logic_raw = raw.get("logic", {})
    migration_raw = raw.get("migration", {})

    return DSLFMConfig(
        embedding_dim=int(model_raw.get("embedding_dim", 128)),
        lambda_logic=float(
            logic_raw.get("lambda_logic", model_raw.get("lambda_logic", 0.0))
        ),
        t_norm=str(logic_raw.get("t_norm", "product")),
        attr_hidden_dim=int(model_raw.get("attr_hidden_dim", 64)),
        use_rule_support=bool(model_raw.get("use_rule_support", False)),
        pruning_threshold=float(pc_raw.get("pruning_threshold", 0.01)),
        grow_noise=float(pc_raw.get("grow_noise", 0.01)),
        lambda_pc=float(pc_raw.get("lambda_pc", 0.0)),
        rebuild_every=int(pc_raw.get("rebuild_every", 0)),
        smoothing_epsilon=float(logic_raw.get("smoothing_epsilon", 1e-6)),
        max_circuit_depth=(
            int(pc_raw["max_circuit_depth"]) if "max_circuit_depth" in pc_raw else None
        ),
        migration_mode=migration_raw.get("mode", "late_fusion"),
    )
