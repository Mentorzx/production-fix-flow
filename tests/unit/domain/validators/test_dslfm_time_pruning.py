"""Integration tests for DSLFM Time Budget Pruning."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockClock:
    """Represent MockClock."""

    def __init__(self):
        """Execute init."""

        self.current_time = 0.0

    def __call__(self):
        return self.current_time

    def advance(self, seconds: float):
        """Execute advance.



        Args:

            seconds: Input value used by this callable.

        """

        self.current_time += seconds


@pytest.fixture
def mock_trial():
    """Execute mock trial.



    Returns:

        Return value produced by the callable.

    """

    trial = MagicMock()
    trial.should_prune.return_value = False
    return trial


@pytest.fixture
def mock_manager(tmp_path):
    """Execute mock manager.



    Args:

        tmp_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    model_config = DSLFMKGCConfig(
        num_entities=10, num_relations=2, entity_dim=4, feature_dim=4, hidden_dim=4
    )

    # Configure for Phase 1 Pruning
    time_budget_dict = {
        "enabled": True,
        "max_total_time_s": 100.0,
        "tolerance_start_s": 90.0,
        "tolerance_evals": 1,
    }

    training_config = KGCTrainingConfig(
        epochs=20,
        batch_size=2,
        checkpoint_dir=tmp_path,
        validate_every=2,
        time_budget=time_budget_dict,
    )

    class MockPersistence:
        """Represent MockPersistence."""

        def save_checkpoint(self, data, filename):
            """Execute save checkpoint.



            Args:

                data: Input value used by this callable.

                filename: Input value used by this callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            pass

        def load_checkpoint(self, filename, map_location=None):
            """Execute load checkpoint.



            Args:

                filename: Input value used by this callable.

                map_location: Optional input value.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            return None

    manager = DSLFMKGCManager(model_config, training_config, persistence_port=MockPersistence())
    return manager


def test_integration_time_pruning(mock_manager, mock_trial):
    """Test that manager prunes when time budget is exceeded."""

    # Mock dependencies
    mock_manager._train_epoch = MagicMock(return_value={"loss": 0.5})
    mock_manager._validate = MagicMock(return_value={"mrr": 0.1, "hits@10": 0.1})
    mock_manager._save_checkpoint = MagicMock()
    mock_manager._load_checkpoint = MagicMock()

    # Mock clock
    clock = MockClock()
    mock_manager.time_estimator.clock = clock
    mock_manager.time_estimator.start_time = clock()
    mock_manager.time_estimator._last_eval_end_time = clock()

    # Define side effects to simulate time passing
    # Epoch 0-1 (first eval interval): Takes 50s.
    # Speed = 50s / 2 eps = 25s/ep.
    # Halfway = 10 eps. Rem = 8.
    # Proj = 50 + (8*25) = 250s > 100s Limit.
    # Should PRUNE at epoch 1.

    def train_side_effect(*args, **kwargs):
        """Execute train side effect.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        clock.advance(20.0)
        return {"loss": 0.5}

    def validate_side_effect(*args, **kwargs):
        """Execute validate side effect.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        clock.advance(5.0)
        return {"mrr": 0.1, "hits@10": 0.1}

    mock_manager._train_epoch.side_effect = train_side_effect
    mock_manager._validate.side_effect = validate_side_effect

    triples = np.array([[0, 0, 1], [1, 1, 0]])

    with patch.dict("sys.modules", {"optuna": MagicMock()}):
        import optuna

        class MockTrialPruned(Exception):
            """Represent MockTrialPruned."""

            pass

        optuna.TrialPruned = MockTrialPruned

        with pytest.raises(MockTrialPruned, match="Time budget exceeded"):
            mock_manager.train(triples, triples, trial=mock_trial)

        # Should verify calls
        # Ep 0: Train (20)
        # Ep 1: Train (20). Validate (5). Check budget. -> Prune.
        assert mock_manager._train_epoch.call_count in [2, 3]
