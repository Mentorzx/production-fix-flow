"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/hpo/test_distributed_sampling.py

"""

from pff.infrastructure.hpo.distributed import _sample_params


def test_sample_params_with_categorical_and_ranges() -> None:
    """Execute test sample params with categorical and ranges.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    search_space = {
        "choice": ["a", "b", "c"],
        "float_range": (0.1, 0.5),
        "int_range": {"type": "int", "low": 1, "high": 5},
        "missing_range": {"type": "int", "low": 1},
    }
    params = _sample_params(search_space, trial_number=2, seed=1337)
    assert params["choice"] == "c"
    assert 0.1 <= float(params["float_range"]) <= 0.5
    assert 1 <= int(params["int_range"]) <= 5
    assert "missing_range" not in params
