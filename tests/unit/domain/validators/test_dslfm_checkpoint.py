"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_dslfm_checkpoint.py

"""

from __future__ import annotations

import torch
from torch import nn

from pff.domain.learning.dslfm.checkpoint_manager import DSLFMCheckpointManager
from pff.shared import FileManager
from pff.shared.core.config import settings


def test_checkpoint_manager_persists_extra_state(tmp_path) -> None:
    """Execute test checkpoint manager persists extra state.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    file_manager = FileManager()
    checkpoint_dir = settings.OUTPUTS_DIR / "temp" / "tests" / "dslfm_checkpoint" / tmp_path.name
    file_manager.ensure_dir(checkpoint_dir)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = DSLFMCheckpointManager(checkpoint_dir, file_manager=file_manager)

    extra_state = {
        "npc": {
            "state_dict": {"dummy": torch.tensor(1)},
            "parents": [0, -1],
            "version": 1,
        }
    }

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={"mrr": 0.1},
        is_best=True,
        extra_state=extra_state,
    )

    restored_model = nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)

    info = manager.load(
        model=restored_model,
        optimizer=restored_optimizer,
        path=checkpoint_path,
        device="cpu",
    )

    assert "extra_state" in info
    assert info["extra_state"].get("npc", {}).get("parents") == [0, -1]
    assert info["epoch"] == 3

    file_manager.delete_directory(checkpoint_dir, ignore_errors=True)
