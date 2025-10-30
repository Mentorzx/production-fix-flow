import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pff.cli import learn_command
from pff.validators.kg.config import KGConfig


@pytest.fixture
def temp_config_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    kg_config_content = f"""
output_dir: {tmp_path / "outputs"}
checkpoint_dir: {tmp_path / "checkpoints"}
train_path: {tmp_path / "train.parquet"}
valid_path: {tmp_path / "valid.parquet"}
test_path: {tmp_path / "test.parquet"}
rules_path: {tmp_path / "rules.tsv"}
max_epochs: 1
num_workers: 2
backend: dask
"""

    kg_config_file = config_dir / "kg.yaml"
    kg_config_file.write_text(kg_config_content)

    return kg_config_file


@pytest.fixture
def mock_kg_pipeline():
    with patch("pff.validators.kg.pipeline.KGPipeline") as mock_pipeline_class:
        mock_instance = Mock()
        mock_instance.run_build_and_preprocess = AsyncMock(return_value=None)
        mock_instance.run_learn_rules = AsyncMock(return_value=None)
        mock_instance.run_ranking = AsyncMock(return_value=None)
        mock_pipeline_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_transe_pipeline():
    with patch("pff.validators.transe.transe_pipeline.TransEPipeline") as mock_pipeline_class:
        mock_instance = Mock()
        mock_instance.train_transe = AsyncMock(return_value=None)
        mock_instance.rank_and_evaluate_transe = Mock(return_value=None)
        mock_pipeline_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_ensemble_pipeline():
    with patch("pff.validators.ensembles.advanced_trainer.run_standalone_ensemble_pipeline", new_callable=AsyncMock) as mock_func:
        mock_func.return_value = {"status": "success"}
        yield mock_func


@pytest.fixture
def mock_autofeeding():
    with patch("pff.utils.autofeeding.apply_autofeeding_rules", new_callable=AsyncMock) as mock_func:
        mock_func.return_value = None
        yield mock_func


class TestLearnCommandKG:

    @pytest.mark.asyncio
    async def test_learn_kg_only(self, temp_config_dir, mock_kg_pipeline):
        with patch("pff.cli.check_interruption", return_value=None):
            await learn_command(model="kg", config_path=temp_config_dir)

        mock_kg_pipeline.run_build_and_preprocess.assert_called_once()
        mock_kg_pipeline.run_learn_rules.assert_called_once()
        mock_kg_pipeline.run_ranking.assert_called_once()

    @pytest.mark.asyncio
    async def test_learn_kg_builds_config_correctly(self, temp_config_dir):
        with patch("pff.validators.kg.pipeline.KGPipeline") as mock_pipeline_class:
            mock_instance = Mock()
            mock_instance.run_build_and_preprocess = AsyncMock()
            mock_instance.run_learn_rules = AsyncMock()
            mock_instance.run_ranking = AsyncMock()
            mock_pipeline_class.return_value = mock_instance

            with patch("pff.cli.check_interruption"):
                await learn_command(model="kg", config_path=temp_config_dir)

            mock_pipeline_class.assert_called_once()
            call_args = mock_pipeline_class.call_args[0]
            assert isinstance(call_args[0], KGConfig)


class TestLearnCommandTransE:

    @pytest.mark.asyncio
    async def test_learn_transe_only(self, temp_config_dir, mock_transe_pipeline):
        with patch("pff.cli.check_interruption", return_value=None):
            await learn_command(model="transe", config_path=temp_config_dir)

        mock_transe_pipeline.train_transe.assert_called_once()
        mock_transe_pipeline.rank_and_evaluate_transe.assert_called_once()

    @pytest.mark.asyncio
    async def test_learn_transe_builds_pipeline_correctly(self, temp_config_dir):
        with patch("pff.validators.transe.transe_pipeline.TransEPipeline") as mock_pipeline_class:
            mock_instance = Mock()
            mock_instance.train_transe = AsyncMock()
            mock_instance.rank_and_evaluate_transe = Mock()
            mock_pipeline_class.return_value = mock_instance

            with patch("pff.cli.check_interruption"):
                await learn_command(model="transe", config_path=temp_config_dir)

            mock_pipeline_class.assert_called_once()
            assert mock_pipeline_class.call_args[0][0] == temp_config_dir


@pytest.mark.slow
class TestLearnCommandEnsemble:

    @pytest.mark.asyncio
    async def test_learn_ensemble_only(self, mock_ensemble_pipeline):
        with patch("pff.cli.check_interruption", return_value=None):
            await learn_command(model="ensemble", config_path=None)

        mock_ensemble_pipeline.assert_called_once()


@pytest.mark.slow
class TestLearnCommandAll:

    @pytest.mark.asyncio
    async def test_learn_all_runs_complete_pipeline(
        self, temp_config_dir, mock_kg_pipeline, mock_transe_pipeline,
        mock_ensemble_pipeline, mock_autofeeding
    ):
        with patch("pff.cli.check_interruption", return_value=None):
            await learn_command(model="all", config_path=temp_config_dir)

        mock_kg_pipeline.run_build_and_preprocess.assert_called_once()
        mock_kg_pipeline.run_learn_rules.assert_called_once()

        mock_transe_pipeline.train_transe.assert_called_once()
        mock_transe_pipeline.rank_and_evaluate_transe.assert_called_once()

        mock_ensemble_pipeline.assert_called_once()
        mock_autofeeding.assert_called_once()

    @pytest.mark.asyncio
    async def test_learn_both_alias(
        self, temp_config_dir, mock_kg_pipeline, mock_transe_pipeline,
        mock_ensemble_pipeline, mock_autofeeding
    ):
        with patch("pff.cli.check_interruption", return_value=None):
            await learn_command(model="both", config_path=temp_config_dir)

        mock_kg_pipeline.run_build_and_preprocess.assert_called_once()
        mock_transe_pipeline.train_transe.assert_called_once()
        mock_ensemble_pipeline.assert_called_once()
        mock_autofeeding.assert_called_once()


class TestLearnCommandInterruption:

    @pytest.mark.asyncio
    async def test_learn_handles_keyboard_interrupt(self, temp_config_dir):
        with patch("pff.validators.kg.pipeline.KGPipeline") as mock_pipeline_class:
            mock_instance = Mock()
            mock_instance.run_build_and_preprocess = AsyncMock(side_effect=KeyboardInterrupt())
            mock_pipeline_class.return_value = mock_instance

            with pytest.raises(SystemExit) as exc_info:
                await learn_command(model="kg", config_path=temp_config_dir)

            assert exc_info.value.code == 128

    @pytest.mark.asyncio
    async def test_learn_handles_interruption_check(self, temp_config_dir):
        interrupt_count = 0

        def mock_check_interruption():
            nonlocal interrupt_count
            interrupt_count += 1
            if interrupt_count > 1:
                raise KeyboardInterrupt("Test interruption")

        with patch("pff.cli.check_interruption", side_effect=mock_check_interruption):
            with patch("pff.validators.kg.pipeline.KGPipeline") as mock_pipeline_class:
                mock_instance = Mock()
                mock_instance.run_build_and_preprocess = AsyncMock()
                mock_instance.run_learn_rules = AsyncMock()
                mock_pipeline_class.return_value = mock_instance

                with pytest.raises(SystemExit) as exc_info:
                    await learn_command(model="kg", config_path=temp_config_dir)

                assert exc_info.value.code == 128


class TestLearnCommandErrorHandling:

    @pytest.mark.asyncio
    async def test_learn_handles_unknown_model(self, temp_config_dir):
        with patch("pff.cli.check_interruption"):
            with pytest.raises(SystemExit) as exc_info:
                await learn_command(model="unknown", config_path=temp_config_dir)

            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_learn_handles_pipeline_exception(self, temp_config_dir):
        with patch("pff.validators.kg.pipeline.KGPipeline") as mock_pipeline_class:
            mock_instance = Mock()
            mock_instance.run_build_and_preprocess = AsyncMock(
                side_effect=RuntimeError("Test error")
            )
            mock_pipeline_class.return_value = mock_instance

            with patch("pff.cli.check_interruption"):
                with pytest.raises(SystemExit) as exc_info:
                    await learn_command(model="kg", config_path=temp_config_dir)

                assert exc_info.value.code == 1


class TestLearnCommandDefaultConfig:

    @pytest.mark.asyncio
    async def test_learn_uses_default_config_when_none_provided(self, mock_kg_pipeline):
        with patch("pff.cli.settings") as mock_settings:
            mock_settings.CONFIG_DIR = Path("/tmp/test_config")
            default_config = Path("/tmp/test_config/kg.yaml")

            default_config.parent.mkdir(parents=True, exist_ok=True)
            default_config.write_text("output_dir: /tmp/test")

            with patch("pff.cli.check_interruption"):
                await learn_command(model="kg", config_path=None)

            mock_kg_pipeline.run_build_and_preprocess.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
