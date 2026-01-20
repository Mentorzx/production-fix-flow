from unittest.mock import patch

import polars as pl
import pytest

from pff.infrastructure.hpo.trials.data_loader import load_real_kg_data
from pff.shared.core.file_manager import FileManager


@pytest.fixture
def temp_kg_data(tmp_path):
    """Create temporary KG data layout."""
    # Setup directories
    models_dir = tmp_path / "data" / "models" / "kg"
    outputs_dir = tmp_path / "outputs" / "kg"
    models_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    # Create dummy data
    train_df = pl.DataFrame({"s": [1, 2, 3], "p": [1, 1, 1], "o": [2, 3, 1]})
    valid_df = pl.DataFrame({"s": [4, 5], "p": [2, 2], "o": [5, 4]})

    # Save as parquet (simulating raw models)
    train_path = models_dir / "train.parquet"
    valid_path = models_dir / "valid.parquet"

    train_df.write_parquet(train_path)
    valid_df.write_parquet(valid_path)

    return {
        "root": tmp_path,
        "models_dir": models_dir,
        "train_path": train_path,
        "valid_path": valid_path,
        "train_df": train_df,
    }


def test_sidecar_creation_and_usage(temp_kg_data, monkeypatch):
    """Verify that load_real_kg_data creates and uses Arrow sidecars."""

    # Mock settings to point to our temp dir
    monkeypatch.setattr("pff.settings.DATA_DIR", temp_kg_data["root"] / "data")
    monkeypatch.setattr("pff.settings.OUTPUTS_DIR", temp_kg_data["root"] / "outputs")

    # Mock _get_kg_paths to return our temp paths explicitly
    # (Since the real one checks config files which we didn't mock fully)
    with patch("pff.infrastructure.hpo.trials.data_loader._get_kg_paths") as mock_paths:
        mock_paths.return_value = (temp_kg_data["train_path"], temp_kg_data["valid_path"])

        # 1. First Load (Should create sidecar)
        fm = FileManager()
        train, valid, info = load_real_kg_data(fm)

        assert len(train) == 3
        assert len(valid) == 2

        # Check sidecars exist
        train_arrow = temp_kg_data["train_path"].with_suffix(".arrow")
        valid_arrow = temp_kg_data["valid_path"].with_suffix(".arrow")

        assert train_arrow.exists(), "Train Arrow sidecar not created"
        assert valid_arrow.exists(), "Valid Arrow sidecar not created"

        # 2. Verify Sidecar Content matches
        train_arrow_df = pl.read_ipc(train_arrow)
        assert train_arrow_df.equals(temp_kg_data["train_df"])

        # 3. Test Acceleration (Corrupt the parquet to prove we read Arrow)
        # Overwrite parquet with garbage
        with open(temp_kg_data["train_path"], "wb") as f:
            f.write(b"CORRUPTED_PARQUET_HEADER_GARBAGE")

        # This load should succeed because it uses the sidecar
        train_2, valid_2, _ = load_real_kg_data(fm)

        assert len(train_2) == 3
        assert train_2.equals(train)

        # 4. Verify Fallback (Delete sidecar, read corrupted parquet -> Should Fail)
        train_arrow.unlink()

        with pytest.raises(Exception):
            # Should fail reading the corrupted parquet
            load_real_kg_data(fm)
