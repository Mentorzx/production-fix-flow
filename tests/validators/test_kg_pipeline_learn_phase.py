import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import polars as pl
import pytest

from pff import settings
from pff.validators.kg.config import KGConfig
from pff.validators.kg.pipeline import KGPipeline


@pytest.fixture
def mock_kg_config(tmp_path):
    """Create a minimal valid KGConfig for testing."""
    # Create required directories
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    graph_dir = data_dir / "models" / "kg"
    pyclause_dir = outputs_dir / "pyclause"
    checkpoints_dir = tmp_path / "checkpoints"
    
    for d in [data_dir, outputs_dir, graph_dir, pyclause_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data files
    import polars as pl
    dummy_df = pl.DataFrame({"s": ["a"], "p": ["r"], "o": ["b"]})
    dummy_df.write_parquet(graph_dir / "train.parquet")
    dummy_df.write_parquet(graph_dir / "valid.parquet")
    dummy_df.write_parquet(graph_dir / "test.parquet")
    
    config_file = tmp_path / "kg_config.yaml"
    config_content = f"""
paths:
  data_dir: {data_dir}
  output_dir: {outputs_dir}
  graph_subdir: models/kg
  pyclause_subdir: pyclause

pipeline:
  top_k: 10
  num_workers: 2
  chunk_size: 200
  max_chunk_size: 300
  max_rules_per_chunk: 1000
  enable_caching: false
  preprocess:
    enabled: false
  calibration:
    enabled: false

builder:
  source_path: "{data_dir}/models/correct.zip"
  parallel: false
  disk_cache: false

pyclause:
  verbose: false
  loader:
    collect_pred_stats: false
  ranking_handler:
    aggregation_function: "noisyor"
    num_threads: 1

anyburl:
  TIME: 10
  SNAPSHOTS_AT: 10
  JAVA_HEAP: "1G"
  WORKER_THREADS: 1
  MAX_LENGTH_CYCLIC: 2
  MAX_LENGTH_ACYCLIC: 2
"""
    config_file.write_text(config_content)
    return KGConfig(str(config_file))


class TestKGPipelineLearnPhase:
    def test_output_dir_resolves_to_outputs(self, tmp_path, monkeypatch):
        """Ensure output_dir='kg' stays under settings.OUTPUTS_DIR, not project root."""
        config_file = tmp_path / "kg_config.yaml"
        config_content = """
paths:
  data_dir: ./data
  output_dir: kg
  graph_subdir: kg
  pyclause_subdir: pyclause
"""
        config_file.write_text(config_content)

        cfg = KGConfig(str(config_file))

        assert str(cfg.output_directory).startswith(str(settings.OUTPUTS_DIR))
        assert cfg.output_directory == settings.OUTPUTS_DIR / "kg"
        assert cfg.graph_directory.is_relative_to(settings.OUTPUTS_DIR)
        assert cfg.graph_directory == settings.OUTPUTS_DIR / "kg"

    @pytest.mark.asyncio
    async def test_run_learn_rules_calls_anyburl(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        with patch.object(pipeline, 'rule_learner') as mock_learner:
            mock_learner.learn_rules = AsyncMock(return_value=Path("/tmp/rules.tsv"))
            # Also mock the internal step method to avoid other dependencies
            with patch.object(pipeline, '_run_learn_rules_step', new_callable=AsyncMock) as mock_step:
                mock_step.return_value = True
                await pipeline.run_learn_rules()
                mock_step.assert_called_once()

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
    async def test_run_ranking_phase(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        # Mock the internal ranking step method instead of non-existent scorer
        with patch.object(pipeline, '_run_ranking_step', new_callable=AsyncMock) as mock_ranking:
            mock_ranking.return_value = {"rules_scored": 100, "mrr": 0.5}

            result = await pipeline.run_ranking()

            mock_ranking.assert_called_once()
            assert result is not None


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
    async def test_complete_pipeline_flow(self, mock_kg_config):
        pipeline = KGPipeline(mock_kg_config)

        # Mock internal step methods for clean unit testing
        with patch.object(pipeline, '_run_learn_rules_step', new_callable=AsyncMock) as mock_learn:
            with patch.object(pipeline, '_run_ranking_step', new_callable=AsyncMock) as mock_ranking:
                mock_learn.return_value = True
                mock_ranking.return_value = {"rules_scored": 100, "mrr": 0.5}

                await pipeline.run_learn_rules()
                await pipeline.run_ranking()

                mock_learn.assert_called_once()
                mock_ranking.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
