import shutil
from pathlib import Path

import lancedb
import pyarrow as pa

from pff.shared import FileManager

DB_PATH = "data/lancedb"
TABLE_NAME = "kg_splits"


if Path(DB_PATH).exists():
    shutil.rmtree(DB_PATH)
Path(DB_PATH).mkdir(parents=True, exist_ok=True)


db = lancedb.connect(DB_PATH)
data = pa.Table.from_pydict(
    {
        "s": ["a", "b", "c"],
        "p": ["p1", "p2", "p3"],
        "o": ["x", "y", "z"],
        "split_name": ["train", "train", "valid"],
        "split_type": ["raw", "raw", "raw"],
    }
)

table = db.create_table(TABLE_NAME, data)
print(f"Created dummy LanceDB at {DB_PATH}")


Path("outputs").mkdir(exist_ok=True)
FileManager().save("A" * 1000, Path("outputs/test_clean.txt"))
print("Created dummy output file outputs/test_clean.txt")
