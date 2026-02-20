"""Dependency injection contracts for KG factory components."""

from __future__ import annotations

from pathlib import Path

from pff.domain.kg.factory import KGComponentFactory


class _DummyCache:
    def get(self, _key: str):
        """Execute get.



        Args:

            _key: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return None

    def set(self, _key: str, _value) -> None:
        """Execute set.



        Args:

            _key: Input value used by this callable.

            _value: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return None

    def disk(self, ttl=None):
        """Execute disk.



        Args:

            ttl: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        def _decorator(fn):
            return fn

        return _decorator


class _DummyConfig:
    graph_directory = Path("outputs/kg/graph")

    def get_builder_config(self) -> dict[str, object]:
        """Execute get builder config.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return {
            "source_path": "outputs/kg/input.parquet",
            "max_members": None,
            "parallel": False,
            "disk_cache": False,
            "workers": 1,
        }

    def get_preprocessing_parameters(self) -> dict[str, object]:
        """Execute get preprocessing parameters.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return {}


class _DummyFileManager:
    def ensure_dir(self, _path: Path) -> None:
        """Execute ensure dir.



        Args:

            _path: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return None


def test_create_builder_accepts_injected_managers() -> None:
    """Execute test create builder accepts injected managers.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    factory = KGComponentFactory()
    cfg = _DummyConfig()
    file_manager = _DummyFileManager()
    cache_manager = _DummyCache()

    builder = factory.create_builder(
        cfg,
        file_manager=file_manager,
        cache_manager=cache_manager,
    )

    assert builder.fm is file_manager
    assert builder._cache_manager is cache_manager


def test_create_preprocessor_accepts_injected_cache_manager() -> None:
    """Execute test create preprocessor accepts injected cache manager.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    factory = KGComponentFactory()
    cfg = _DummyConfig()
    file_manager = _DummyFileManager()
    cache_manager = _DummyCache()

    preprocessor = factory.create_preprocessor(
        cfg,
        file_manager=file_manager,
        cache_manager=cache_manager,
    )

    assert preprocessor.file_manager is file_manager
    assert preprocessor.cache_manager is cache_manager
    assert preprocessor.indexer.cache_manager is cache_manager
