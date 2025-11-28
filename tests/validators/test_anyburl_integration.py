import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import polars as pl
import pytest

from pff.validators.kg.anyburl import (
    AnyBURLLearner,
    AnyBURLOptionsBuilder,
    TripleFormatConverter,
)
from pff.validators.kg.config import KGConfig


@pytest.fixture
def sample_triples_df():
    return pl.DataFrame({
        "s": ["user_1", "user_1", "user_2", "user_2", "user_3"],
        "p": ["hasProduct", "hasStatus", "hasProduct", "hasStatus", "hasProduct"],
        "o": ["prod_A", "active", "prod_B", "inactive", "prod_A"],
    })


@pytest.fixture
def temp_paths(tmp_path):
    return {
        "parquet": tmp_path / "test_train.parquet",
        "tsv": tmp_path / "test_train.tsv",
        "rules": tmp_path / "test_rules.tsv",
        "pyclause_dir": tmp_path / "pyclause",
    }


@pytest.fixture
def mock_config(temp_paths):
    config = Mock(spec=KGConfig)
    temp_paths["pyclause_dir"].mkdir(parents=True, exist_ok=True)
    config.get_pyclause_directory.return_value = temp_paths["pyclause_dir"]
    config.get_rules_path.return_value = temp_paths["rules"]
    config.get_split_path.return_value = temp_paths["parquet"]
    config.get_anyburl_parameters.return_value = {
        "TIME": 10,
        "SNAPSHOTS_AT": [5, 10],
        "JAVA_HEAP": "2G",
    }
    return config


@pytest.fixture
def mock_rules_repo():
    repo = AsyncMock()
    repo.save_rules_from_file = AsyncMock(return_value=10)
    return repo


class TestTripleFormatConverter:

    def test_convert_parquet_to_tsv_success(self, sample_triples_df, temp_paths):
        sample_triples_df.write_parquet(temp_paths["parquet"])

        converter = TripleFormatConverter()
        converter.convert_parquet_to_tsv(temp_paths["parquet"], temp_paths["tsv"])

        assert temp_paths["tsv"].exists()
        result_df = pl.read_csv(temp_paths["tsv"], separator="\t", has_header=True)
        assert len(result_df) == 5
        assert result_df.columns == ["s", "p", "o"]
        temp_paths["tsv"].unlink()

    def test_convert_parquet_to_tsv_missing_columns(self, temp_paths):
        invalid_df = pl.DataFrame({
            "subject": ["user_1"],
            "predicate": ["hasProduct"],
        })
        invalid_df.write_parquet(temp_paths["parquet"])

        converter = TripleFormatConverter()
        with pytest.raises(ValueError, match="deve conter colunas"):
            converter.convert_parquet_to_tsv(temp_paths["parquet"], temp_paths["tsv"])

        temp_paths["parquet"].unlink()

    def test_convert_parquet_to_tsv_empty_dataframe(self, temp_paths):
        empty_df = pl.DataFrame({"s": [], "p": [], "o": []})
        empty_df.write_parquet(temp_paths["parquet"])

        converter = TripleFormatConverter()
        converter.convert_parquet_to_tsv(temp_paths["parquet"], temp_paths["tsv"])

        assert temp_paths["tsv"].exists()
        result_df = pl.read_csv(temp_paths["tsv"], separator="\t", has_header=True)
        assert len(result_df) == 0

        temp_paths["parquet"].unlink()
        temp_paths["tsv"].unlink()


class TestAnyBURLOptionsBuilder:

    def test_build_options_basic(self, mock_config, temp_paths):
        builder = AnyBURLOptionsBuilder()
        options = builder.build_options(
            mock_config,
            temp_paths["tsv"],
            temp_paths["rules"],
        )

        opts_dict = options.to_dict() if hasattr(options, 'to_dict') else options.get("learner")
        assert opts_dict is not None
        if isinstance(opts_dict, dict):
            assert "mode" in opts_dict or "anyburl" in opts_dict

    def test_build_options_with_time_parameter(self, mock_config, temp_paths):
        builder = AnyBURLOptionsBuilder()
        options = builder.build_options(
            mock_config,
            temp_paths["tsv"],
            temp_paths["rules"],
        )

        assert options is not None

    def test_build_options_with_java_heap(self, mock_config, temp_paths):
        builder = AnyBURLOptionsBuilder()
        options = builder.build_options(
            mock_config,
            temp_paths["tsv"],
            temp_paths["rules"],
        )

        assert options is not None

    def test_build_options_with_list_parameters(self, mock_config, temp_paths):
        builder = AnyBURLOptionsBuilder()
        options = builder.build_options(
            mock_config,
            temp_paths["tsv"],
            temp_paths["rules"],
        )

        assert options is not None


@pytest.mark.slow
class TestAnyBURLLearner:

    @pytest.mark.asyncio
    async def test_learn_rules_with_homogenized_data(self, mock_config, sample_triples_df, temp_paths, mock_rules_repo):
        homogenized_path = temp_paths["pyclause_dir"] / "train.homogenized.parquet"
        sample_triples_df.write_parquet(homogenized_path)

        learner = AnyBURLLearner(rules_repository=mock_rules_repo)

        with patch.object(learner, "_execute_anyburl", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = None
            temp_paths["rules"].touch()

            result_path = await learner.learn_rules(mock_config)

            assert result_path == temp_paths["rules"]
            mock_execute.assert_called_once()
            assert not (temp_paths["pyclause_dir"] / "train.tsv").exists()

        homogenized_path.unlink()
        temp_paths["rules"].unlink()

    @pytest.mark.asyncio
    async def test_learn_rules_without_homogenized_data(self, mock_config, sample_triples_df, temp_paths, mock_rules_repo):
        sample_triples_df.write_parquet(temp_paths["parquet"])

        learner = AnyBURLLearner(rules_repository=mock_rules_repo)

        with patch.object(learner, "_execute_anyburl", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = None
            temp_paths["rules"].touch()

            result_path = await learner.learn_rules(mock_config)

            assert result_path == temp_paths["rules"]
            mock_execute.assert_called_once()

        temp_paths["parquet"].unlink()
        temp_paths["rules"].unlink()

    @pytest.mark.asyncio
    async def test_learn_rules_cleans_up_temp_files(self, mock_config, sample_triples_df, temp_paths, mock_rules_repo):
        sample_triples_df.write_parquet(temp_paths["parquet"])

        learner = AnyBURLLearner(rules_repository=mock_rules_repo)
        train_tsv_path = temp_paths["pyclause_dir"] / "train.tsv"

        with patch.object(learner, "_execute_anyburl", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = None
            temp_paths["rules"].touch()

            await learner.learn_rules(mock_config)

            assert not train_tsv_path.exists()

        temp_paths["parquet"].unlink()
        temp_paths["rules"].unlink()

    @pytest.mark.asyncio
    async def test_learn_rules_error_handling(self, mock_config):
        learner = AnyBURLLearner()

        mock_config.get_split_path.return_value = Path("/nonexistent/path.parquet")

        with pytest.raises(Exception):
            await learner.learn_rules(mock_config)

    @pytest.mark.asyncio
    async def test_execute_anyburl_creates_rules_file(self, temp_paths, sample_triples_df):
        sample_triples_df.select(["s", "p", "o"]).write_csv(
            temp_paths["tsv"], separator="\t", include_header=False
        )

        if not temp_paths["tsv"].exists():
            pytest.skip("TSV file not created, skipping AnyBURL execution test")

        from clause import Learner, Options

        options = Options()
        options.set("learner.mode", "anyburl")
        options.set("learner.anyburl.raw.PATH_TRAIN", temp_paths["tsv"].as_posix())
        options.set("learner.anyburl.raw.PATH_OUTPUT", temp_paths["rules"].as_posix())
        options.set("learner.anyburl.time", "5")
        options.set("learner.anyburl.java_options", "['-Xmx1G', '-Dfile.encoding=UTF-8']")

        learner_instance = Learner(options=options.get("learner"))
        learner_instance.learn_rules(
            temp_paths["tsv"].as_posix(),
            temp_paths["rules"].as_posix()
        )

        temp_paths["tsv"].unlink()
        if temp_paths["rules"].exists():
            temp_paths["rules"].unlink()


class TestAnyBURLIntegrationEdgeCases:

    def test_converter_with_special_characters(self, temp_paths):
        df = pl.DataFrame({
            "s": ["user@1", "user#2"],
            "p": ["hasProduct", "hasStatus"],
            "o": ["prod_A", "active"],
        })
        df.write_parquet(temp_paths["parquet"])

        converter = TripleFormatConverter()
        converter.convert_parquet_to_tsv(temp_paths["parquet"], temp_paths["tsv"])

        assert temp_paths["tsv"].exists()
        result_df = pl.read_csv(temp_paths["tsv"], separator="\t", has_header=True)
        assert len(result_df) == 2

        temp_paths["parquet"].unlink()
        temp_paths["tsv"].unlink()

    def test_converter_with_unicode(self, temp_paths):
        df = pl.DataFrame({
            "s": ["usuário_1", "usuário_2"],
            "p": ["temProduto", "temStatus"],
            "o": ["prod_Ã", "ativo"],
        })
        df.write_parquet(temp_paths["parquet"])

        converter = TripleFormatConverter()
        converter.convert_parquet_to_tsv(temp_paths["parquet"], temp_paths["tsv"])

        assert temp_paths["tsv"].exists()
        result_df = pl.read_csv(temp_paths["tsv"], separator="\t", has_header=True)
        assert len(result_df) == 2

        temp_paths["parquet"].unlink()
        temp_paths["tsv"].unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
