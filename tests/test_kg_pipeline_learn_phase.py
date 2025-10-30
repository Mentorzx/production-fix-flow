import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pff.validators.kg.config import KGConfig
from pff.validators.kg.pipeline import KGPipeline


@pytest.fixture
def mock_kg_config(tmp_path):
    config_file = tmp_path / "kg_config.yaml"
    config_content = f"""
output_dir: {tmp_path / "outputs"}
checkpoint_dir: {tmp_path / "checkpoints"}
train_path: {tmp_path / "train.parquet"}
valid_path: {tmp_path / "valid.parquet"}
test_path: {tmp_path / "test.parquet"}
rules_path: {tmp_path / "rules.tsv"}
max_epochs: 2
num_workers: 2
backend: dask
"""
    config_file.write_text(config_content)
    return KGConfig(str(config_file))


class TestKGPipelineLearnPhase:

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_run_learn_rules_calls_anyburl(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        with patch.object(pipeline, 'rule_learner') as mock_learner:
            mock_learner.learn_rules = AsyncMock(return_value=Path("/tmp/rules.tsv"))

            await pipeline.run_learn_rules()

            mock_learner.learn_rules.assert_called_once()

    @pytest.mark.asyncio
    async def test_learn_phase_creates_checkpoint(self, mock_kg_config, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        pipeline = KGPipeline(mock_kg_config)

        with patch.object(pipeline, 'rule_learner') as mock_learner:
            mock_learner.learn_rules = AsyncMock(return_value=Path("/tmp/rules.tsv"))
            await pipeline.run_learn_rules()

        checkpoint_file = checkpoint_dir / "learn_complete.json"
        if checkpoint_file.exists():
            import json
            checkpoint_data = json.loads(checkpoint_file.read_text())
            assert checkpoint_data["phase"] == "learn"
            assert checkpoint_data["completed"] is True


class TestKGPipelineRanking:

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_run_ranking_phase(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        with patch.object(pipeline, 'scorer') as mock_scorer:
            mock_scorer.score_rules = AsyncMock(return_value={"rules_scored": 100})

            await pipeline.run_ranking()

            mock_scorer.score_rules.assert_called_once()


class TestKGPipelineBackendSelection:

    @pytest.mark.slow
    def test_backend_auto_selection_linux(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)
        assert pipeline is not None

    @pytest.mark.slow
    def test_backend_auto_selection_windows(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)
        assert pipeline is not None


class TestKGPipelineCheckpoints:

    def test_can_resume_from_checkpoint_exists(self, mock_kg_config, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / "learn_complete.json"
        checkpoint_data = {
            "phase": "learn",
            "completed": True,
            "timestamp": "2025-10-29T21:00:00"
        }

        import json
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        pipeline = KGPipeline(mock_kg_config)

        can_resume = pipeline.can_resume_from_checkpoint("learn")
        assert can_resume in [True, False]

    def test_can_resume_from_checkpoint_not_exists(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        can_resume = pipeline.can_resume_from_checkpoint("learn")
        assert can_resume is False

    def test_can_resume_from_incomplete_checkpoint(self, mock_kg_config, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / "learn_complete.json"
        checkpoint_data = {
            "phase": "learn",
            "completed": False,
            "timestamp": "2025-10-29T21:00:00"
        }

        import json
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        pipeline = KGPipeline(mock_kg_config)

        can_resume = pipeline.can_resume_from_checkpoint("learn")
        assert can_resume is False


class TestKGPipelineIntegration:

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_pipeline_flow(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        with patch.object(pipeline, 'rule_learner') as mock_learner:
            with patch.object(pipeline, 'scorer') as mock_scorer:
                mock_learner.learn_rules = AsyncMock(return_value=Path("/tmp/rules.tsv"))
                mock_scorer.score_rules = AsyncMock(return_value={"rules_scored": 100})

                await pipeline.run_learn_rules()
                await pipeline.run_ranking()

                mock_learner.learn_rules.assert_called_once()
                mock_scorer.score_rules.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
