import polars as pl
from pathlib import Path

from pff import settings
from pff.domain.kg.preprocessing.config import PreprocessingConfig
from pff.domain.kg.preprocessing.pipeline import KGPreprocessingPipeline
from pff.shared import FileManager


def test_id_mapping_consistency(tmp_path: Path) -> None:
    """Verify ID mapping consistency across splits using a synthetic KG."""
    output_dir = (
        settings.OUTPUTS_DIR / "temp" / "tests" / "audit_id_mapping" / tmp_path.name
    )
    fm = FileManager()

    try:
        pipeline = KGPreprocessingPipeline(
            PreprocessingConfig(output_dir=str(output_dir))
        )
        train_df = pl.DataFrame({"s": ["a", "a"], "p": ["r1", "r1"], "o": ["b", "c"]})
        valid_df = pl.DataFrame({"s": ["b", "c"], "p": ["r2", "r2"], "o": ["a", "b"]})

        mapped_train, mapped_valid, _ = pipeline._map_ids_for_splits(
            train_df, valid_df, None
        )

        train_entities = set(mapped_train["s"].to_list()) | set(
            mapped_train["o"].to_list()
        )
        valid_entities = set(mapped_valid["s"].to_list()) | set(
            mapped_valid["o"].to_list()
        )
        assert not (valid_entities - train_entities)

        assert mapped_train.schema["s"] in (pl.Int64, pl.Int32)
        assert mapped_valid.schema["s"] in (pl.Int64, pl.Int32)

        all_ids = train_entities | valid_entities
        assert max(all_ids) == len(all_ids) - 1
    finally:
        fm.delete_directory(output_dir, ignore_errors=True)
