"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/kg/test_task_runner_no_fallback.py

"""

from __future__ import annotations

import pytest

from pff.domain.kg.task_runner import TaskRunnerFactory


@pytest.mark.asyncio
async def test_execute_with_fallback_rejects_multiple_backends() -> None:
    """Execute test execute with fallback rejects multiple backends.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    with pytest.raises(ValueError, match="Fallback execution is disabled"):
        await TaskRunnerFactory.execute_with_fallback(
            ["dask", "thread"],
            lambda x: x,
            [1],
            desc="test",
        )


@pytest.mark.asyncio
async def test_execute_with_fallback_requires_one_backend() -> None:
    """Execute test execute with fallback requires one backend.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    with pytest.raises(ValueError, match="At least one backend"):
        await TaskRunnerFactory.execute_with_fallback([], lambda x: x, [1], desc="test")


@pytest.mark.asyncio
async def test_execute_with_fallback_runs_selected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute test execute with fallback runs selected backend.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    class _Runner:
        async def execute(self, func, args, desc):
            """Execute execute.



            Args:

                func: Input value used by this callable.

                args: Input value used by this callable.

                desc: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            return [func(args[0]), desc]

    monkeypatch.setattr(
        TaskRunnerFactory,
        "get_specific_runner",
        lambda runner_type, config=None: _Runner(),
    )

    result = await TaskRunnerFactory.execute_with_fallback(
        ["thread"],
        lambda x: x + 1,
        [2],
        desc="strict",
        config_by_backend={"thread": {"max_workers": 1}},
    )

    assert result == [3, "strict"]
