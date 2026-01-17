from pathlib import Path
from pff.shared import FileManager, logger


class ParquetAdapter:
    def __init__(self, file_manager: FileManager):
        self.fm = file_manager

    def to_triples(self, path: Path) -> list[list[str]]:
        bundle = self.fm.read(path)
        df = bundle.lazyframe().collect() if hasattr(bundle, "lazyframe") else bundle

        required = ["s", "p", "o"]
        if not all(c in df.columns for c in required):
            # Try to find columns if not named s, p, o
            # This is a fallback for different parquet structures
            logger.warning(
                f"Columns {required} not found in {path}. Found: {df.columns}"
            )
            if len(df.columns) >= 3:
                cols = df.columns[:3]
                logger.info(f"Using first 3 columns: {cols}")
                df = df.select(cols)
                df.columns = required
            else:
                raise ValueError(
                    f"Parquet must have at least 3 columns for triples. Found: {df.columns}"
                )

        return [list(row) for row in df.select(required).iter_rows()]


class DataLoaderAdapterFactory:
    def __init__(self, file_manager: FileManager):
        self.fm = file_manager

    def create(self, path: Path):
        if path.suffix == ".parquet":
            return ParquetAdapter(self.fm)
        raise ValueError(f"Unsupported file format: {path.suffix}")
