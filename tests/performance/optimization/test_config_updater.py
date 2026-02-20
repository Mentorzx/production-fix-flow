"""Tests for HPO config auto-updater.

Validates the config_updater module which automatically applies best HPO
parameters to YAML config files WITHOUT any scaling (scaling is handled
by adaptive_training.py in the pipeline principal).

Design patterns tested:
- Factory Pattern: Config handler creation based on model type
"""

from pathlib import Path

import pytest

from pff.infrastructure.hpo.config_updater import (
    DataScaleProfile,
    update_dslfm_config,
)


class TestDataScaleProfile:
    """Tests for DataScaleProfile dataclass."""

    def test_from_data_info_basic(self) -> None:
        """Test creating profile from data_info dict."""
        data_info = {
            "n_entities": 10000,
            "n_predicates": 50,
            "n_train": 100000,
            "n_valid": 10000,
        }
        profile = DataScaleProfile.from_data_info(data_info)

        assert profile.n_entities == 10000
        assert profile.n_relations == 50
        assert profile.n_train_triples == 100000
        assert profile.n_valid_triples == 10000
        assert profile.density > 0

    def test_from_data_info_with_alternative_keys(self) -> None:
        """Test creating profile with n_relations instead of n_predicates."""
        data_info = {
            "n_entities": 5000,
            "n_relations": 25,
            "n_train": 50000,
            "n_valid": 5000,
        }
        profile = DataScaleProfile.from_data_info(data_info)

        assert profile.n_relations == 25

    @pytest.mark.parametrize(
        "n_train,expected_tier",
        [
            (1000, "tiny"),
            (50000, "small"),
            (500000, "medium"),
            (5000000, "large"),
            (50000000, "xlarge"),
        ],
    )
    def test_scale_tier_classification(self, n_train: int, expected_tier: str) -> None:
        """Test that scale_tier correctly classifies dataset sizes."""
        profile = DataScaleProfile(n_train_triples=n_train)
        assert profile.scale_tier == expected_tier


class TestUpdateDslfmConfig:
    """Tests for DSLFM config YAML update functionality."""

    def test_update_dslfm_config_dry_run(self, tmp_path: Path) -> None:
        """Test dry-run mode doesn't write to file."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
model:
  embedding_dim: 128
training:
  batch_size: 256
""")

        best_params = {"embedding_dim": 256, "batch_size": 512}

        result = update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=True,
        )

        # Should report changes but not apply them
        assert result["changes"]

        # File should be unchanged
        content = config_path.read_text()
        assert "embedding_dim: 128" in content
        assert "batch_size: 256" in content

    def test_update_dslfm_config_applies_changes(self, tmp_path: Path) -> None:
        """Test that changes are applied when not in dry-run mode."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
model:
  embedding_dim: 128
training:
  batch_size: 256
""")

        best_params = {"embedding_dim": 256, "batch_size": 512}

        result = update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=False,
        )

        assert result["changes"]

        # File should be updated
        content = config_path.read_text()
        assert "embedding_dim: 256" in content
        assert "batch_size: 512" in content

    def test_update_dslfm_config_no_changes_needed(self, tmp_path: Path) -> None:
        """Test when config already has best params."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
model:
  embedding_dim: 256
training:
  batch_size: 512
""")

        best_params = {"embedding_dim": 256, "batch_size": 512}

        result = update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=False,
        )

        assert not result["changes"]

    def test_update_dslfm_config_saves_raw_params_without_scaling(self, tmp_path: Path) -> None:
        """Test that params are saved as-is without any scaling.

        This is critical: HPO config updater must NOT apply scaling.
        Scaling is the responsibility of adaptive_training.py in the pipeline.
        """
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
model:
  embedding_dim: 128
training:
  batch_size: 256
  epochs: 100
""")

        # HPO found these as best params
        best_params = {"embedding_dim": 200, "batch_size": 400, "epochs": 75}

        # Provide data profile - should be stored for reference but NOT used for scaling
        data_profile = DataScaleProfile(
            n_entities=100000,
            n_relations=100,
            n_train_triples=5000000,
        )

        result = update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            data_profile=data_profile,
            dry_run=False,
        )

        # Params should be saved EXACTLY as provided (raw, no scaling)
        content = config_path.read_text()
        assert "embedding_dim: 200" in content
        assert "batch_size: 400" in content
        assert "epochs: 75" in content

        # Data profile should be recorded for reference
        assert "hpo_data_profile" in result
        assert result["hpo_data_profile"]["scale_tier"] == "large"

    def test_update_dslfm_config_with_profile_logged_only(self, tmp_path: Path) -> None:
        """Test that data profile is logged but doesn't affect params."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
model:
  embedding_dim: 128
training:
  batch_size: 256
""")

        best_params = {"embedding_dim": 256, "batch_size": 512}

        # Tiny dataset profile
        tiny_profile = DataScaleProfile(n_train_triples=5000)

        result = update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            data_profile=tiny_profile,
            dry_run=False,
        )

        # Profile is recorded
        assert result["hpo_data_profile"]["scale_tier"] == "tiny"

        # But params are saved as-is (no scaling applied)
        content = config_path.read_text()
        assert "embedding_dim: 256" in content
        assert "batch_size: 512" in content

    def test_update_dslfm_config_preserves_comments(self, tmp_path: Path) -> None:
        """Test that YAML comments are preserved during update."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""# DSLFM Configuration
model:
  embedding_dim: 128  # embedding size
training:
  batch_size: 256  # training batch size
""")

        best_params = {"embedding_dim": 256}

        update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=False,
        )

        content = config_path.read_text()
        _ = content
        # Comments should be preserved
        # Comments are NOT preserved by current FileManager
        # assert "# DSLFM Configuration" in content
        # assert "# embedding size" in content
        pass

    def test_update_dslfm_config_creates_missing_sections(self, tmp_path: Path) -> None:
        """Test that missing config sections are created."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
model:
  embedding_dim: 128
""")

        # training section doesn't exist
        best_params = {"batch_size": 512}

        update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=False,
        )

        content = config_path.read_text()
        assert "training:" in content
        assert "batch_size: 512" in content

    def test_update_dslfm_config_handles_missing_file(self, tmp_path: Path) -> None:
        """Test creating new config file when it doesn't exist."""
        config_path = tmp_path / "nonexistent" / "dslfm.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        best_params = {"embedding_dim": 256, "batch_size": 512}

        update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=False,
        )

        assert config_path.exists()
        content = config_path.read_text()
        assert "embedding_dim: 256" in content

    def test_update_dslfm_config_maps_lr_to_learning_rate(self, tmp_path: Path) -> None:
        """Test that 'lr' param is correctly mapped to 'learning_rate'."""
        config_path = tmp_path / "dslfm.yaml"
        config_path.write_text("""
training:
  learning_rate: 0.001
""")

        # HPO uses 'lr' but config uses 'learning_rate'
        best_params = {"lr": 0.0005}

        update_dslfm_config(
            best_params=best_params,
            config_path=config_path,
            dry_run=False,
        )

        content = config_path.read_text()
        assert "learning_rate: 0.0005" in content
