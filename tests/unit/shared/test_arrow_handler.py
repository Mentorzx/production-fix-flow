"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/test_arrow_handler.py

"""

import polars as pl
import pyarrow as pa
import pytest

from pff.shared.core.file_manager.handlers.arrow_ipc import ArrowIPCHandler


@pytest.fixture
def handler():
    """Execute handler.



    Returns:

        Return value produced by the callable.

    """

    return ArrowIPCHandler()


@pytest.fixture
def arrow_file(tmp_path):
    """Execute arrow file.



    Args:

        tmp_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "test.arrow"
    df.write_ipc(path, compression="uncompressed")
    return path


def test_read_defaults(handler, arrow_file):
    # Should default to Polars DataFrame
    """Execute test read defaults.



    Args:

        handler: Input value used by this callable.

        arrow_file: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    df = handler.read(arrow_file)
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)
    assert df["a"].sum() == 6


def test_read_lazy(handler, arrow_file):
    """Execute test read lazy.



    Args:

        handler: Input value used by this callable.

        arrow_file: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    lf = handler.read(arrow_file, lazy=True)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().shape == (3, 2)


def test_read_pyarrow(handler, arrow_file):
    """Execute test read pyarrow.



    Args:

        handler: Input value used by this callable.

        arrow_file: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    tbl = handler.read(arrow_file, use_pyarrow=True)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3


def test_save_atomic(handler, tmp_path):
    """Execute test save atomic.



    Args:

        handler: Input value used by this callable.

        tmp_path: Input value used by this callable.

    """

    df = pl.DataFrame({"x": [10, 20]})
    path = tmp_path / "output.arrow"

    handler.save(df, path)
    assert path.exists()
    assert pl.read_ipc(path)["x"][0] == 10


def test_save_pyarrow_table(handler, tmp_path):
    """Execute test save pyarrow table.



    Args:

        handler: Input value used by this callable.

        tmp_path: Input value used by this callable.

    """

    tbl = pa.Table.from_pydict({"x": [100, 200]})
    path = tmp_path / "output_pa.arrow"

    handler.save(tbl, path)
    assert path.exists()
    assert pl.read_ipc(path)["x"][0] == 100


def test_compression_handling(handler, tmp_path):
    """Execute test compression handling.



    Args:

        handler: Input value used by this callable.

        tmp_path: Input value used by this callable.

    """

    df = pl.DataFrame({"x": [1]})
    path = tmp_path / "comp.arrow"

    # Save with LZ4
    handler.save(df, path, compression="lz4")

    # Read back
    df_read = handler.read(path)
    assert df_read["x"][0] == 1

    # Verify it is actually compressed? (Hard to do easily, but we trust the call)


def test_invalid_input(handler, tmp_path):
    """Execute test invalid input.



    Args:

        handler: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    path = tmp_path / "bad.arrow"
    with pytest.raises(TypeError):
        handler.save({"not": "a dataframe"}, path)


def test_missing_file_read(handler, tmp_path):
    """Execute test missing file read.



    Args:

        handler: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    path = tmp_path / "missing.arrow"
    with pytest.raises(Exception):
        handler.read(path)
