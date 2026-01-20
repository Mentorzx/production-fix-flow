from pathlib import Path

import pytest

from pff import settings
from pff.drivers.orchestrator import BufferedWriter
from pff.shared import FileManager


@pytest.mark.asyncio
async def test_buffered_writer_combines_parts(tmp_path: Path) -> None:
    dest = settings.OUTPUTS_DIR / "temp" / "tests" / "bw_test.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for leftover in dest.parent.glob(f"{dest.stem}__part*{dest.suffix}"):
        leftover.unlink(missing_ok=True)

    writer = BufferedWriter(dest, flush_rows=1, flush_secs=0.01)
    await writer.write({"col": 1})
    await writer.write({"col": 2})
    await writer.close()

    df = FileManager.read(dest, return_native=True)
    try:
        assert df.shape[0] == 2
        part_files = list(dest.parent.glob(f"{dest.stem}__part*{dest.suffix}"))
        assert not part_files
    finally:
        dest.unlink(missing_ok=True)
