"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kg_preprocessor_no_fallback.py

"""

from __future__ import annotations

from pathlib import Path

import pytest

from pff.domain.kg.config import ConfigurationInterface
from pff.domain.kg.preprocess import KGPreprocessor


class _StubConfig(ConfigurationInterface):
    def __init__(self, base_dir: Path) -> None:
        """Execute init.



        Args:

            base_dir: Input value used by this callable.

        """

        self._base_dir = base_dir

    def validate(self) -> bool:
        """Execute validate.



        Returns:

            Return value produced by the callable.

        """

        return True

    def get_split_path(self, split_name: str) -> Path:
        """Execute get split path.



        Args:

            split_name: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return self._base_dir / f"{split_name}.parquet"

    def get_preprocessing_parameters(self) -> dict[str, float | int | bool]:
        """Execute get preprocessing parameters.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return {"use_centralized_preprocessing": True}

    def get_entity_map_path(self) -> Path:
        """Execute get entity map path.



        Returns:

            Return value produced by the callable.

        """

        return self._base_dir / "entity_map.json"

    def get_relation_map_path(self) -> Path:
        """Execute get relation map path.



        Returns:

            Return value produced by the callable.

        """

        return self._base_dir / "relation_map.json"

    def get_max_chunk_size(self) -> int:
        """Execute get max chunk size.



        Returns:

            Return value produced by the callable.

        """

        return 1

    def get_mappings_directory(self) -> Path:
        """Execute get mappings directory.



        Returns:

            Return value produced by the callable.

        """

        return self._base_dir

    def get_calibration_config(self) -> dict:
        """Execute get calibration config.



        Returns:

            Return value produced by the callable.

        """

        return {}

    def get_dask_configuration(self) -> dict:
        """Execute get dask configuration.



        Returns:

            Return value produced by the callable.

        """

        return {}


def test_preprocessor_raises_when_centralized_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute test preprocessor raises when centralized fails.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = _StubConfig(tmp_path)
    preprocessor = KGPreprocessor(config)

    monkeypatch.setattr(preprocessor, "_run_centralized_preprocessing", lambda: False)

    with pytest.raises(RuntimeError, match="Centralized preprocessing failed"):
        preprocessor.run()
