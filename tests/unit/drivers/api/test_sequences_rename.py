"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/drivers/api/test_sequences_rename.py

"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from pff.drivers.api.routers import sequences


class _FakeFileManager:
    def __init__(self, data: dict) -> None:
        """Execute init.



        Args:

            data: Input value used by this callable.

        """

        self._data = data
        self.saved: dict | None = None

    def read(self, _path, return_native: bool = True):  # noqa: ARG002
        """Execute read.



        Args:

            _path: Input value used by this callable.

            return_native: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        return self._data

    def save(self, data, _path):  # noqa: ANN001
        """Execute save.



        Args:

            data: Input value used by this callable.

            _path: Input value used by this callable.

        """

        self.saved = data
        self._data = data


def test_rename_sequence_rejects_empty_name() -> None:
    """Execute test rename sequence rejects empty name.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    with pytest.raises(HTTPException) as exc_info:
        sequences.rename_sequence("old_name", "  ")
    assert exc_info.value.status_code == 400


def test_rename_sequence_returns_404_for_missing_sequence(monkeypatch) -> None:
    """Execute test rename sequence returns 404 for missing sequence.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    monkeypatch.setattr(sequences, "file_manager", _FakeFileManager({}))
    monkeypatch.setattr(sequences, "cache_manager", {})
    with pytest.raises(HTTPException) as exc_info:
        sequences.rename_sequence("missing", "new")
    assert exc_info.value.status_code == 404


def test_rename_sequence_updates_references_and_cache(monkeypatch) -> None:
    """Execute test rename sequence updates references and cache.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    initial = {
        "old": [{"method": "noop"}],
        "caller": [{"next_sequence": "old"}],
    }
    fake_file_manager = _FakeFileManager(initial)
    fake_cache = {"sequences:list": ["cached"], "sequence:old": [{"cached": True}]}
    monkeypatch.setattr(sequences, "file_manager", fake_file_manager)
    monkeypatch.setattr(sequences, "cache_manager", fake_cache)

    result = sequences.rename_sequence("old", "new")

    assert result["updated_references"] == 1
    assert "old" not in fake_file_manager.saved
    assert "new" in fake_file_manager.saved
    caller_sequence = fake_file_manager.saved.get("caller")
    assert caller_sequence is not None
    assert caller_sequence[0]["next_sequence"] == "new"
    assert "sequences:list" not in fake_cache
    assert "sequence:old" not in fake_cache
