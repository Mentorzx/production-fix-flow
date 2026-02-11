from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pff.domain.learning.dslfm.kgc_manager import (
    DSLFMKGCConfig,
    DSLFMKGCManager,
    KGCTrainingConfig,
)


class TestDSLFMGracefulShutdown:
    @patch("pff.domain.learning.dslfm.kgc_manager.should_stop")
    def test_train_stops_on_signal(self, mock_should_stop):
        """Test that train loop breaks when should_stop returns True."""

        # Scenario:
        # epoch 0: should_stop = False
        # epoch 1: should_stop = True
        # Expected: train returns after epoch 0 (or at start of epoch 1)

        # We configure 10 epochs
        config = DSLFMKGCConfig(num_entities=100, num_relations=10)
        train_config = KGCTrainingConfig(epochs=10, batch_size=10)

        class MockPersistence:
            def save_checkpoint(self, data, filename):
                pass

            def load_checkpoint(self, filename, map_location=None):
                return None

        manager = DSLFMKGCManager(
            config, train_config, persistence_port=MockPersistence()
        )

        # Mock dependencies to avoid real training overhead
        manager.model = MagicMock()
        manager.model.compute_loss.return_value = {
            "loss": torch.tensor(1.0, requires_grad=True)
        }
        manager._validate = MagicMock(return_value={})
        manager._save_checkpoint = MagicMock()
        manager.optimizer = MagicMock()
        manager.scheduler = MagicMock()
        manager.scaler = (
            None  # Disable scaler for this test to avoid inf check issues with mocks
        )

        # Mock dataset
        triples = np.random.randint(0, 10, (100, 3))

        # Mock should_stop to return False first, then True
        # NOTE: should_stop is called once per epoch in the loop
        # Iteration 0: should_stop() -> False (runs epoch 0)
        # Iteration 1: should_stop() -> True (breaks)
        mock_should_stop.side_effect = [False, True, True, True]

        stats = manager.train(triples, triples)

        # Assertions
        assert mock_should_stop.call_count >= 2
        # It should have trained fewer than 10 epochs.
        # Specifically, it completes epoch 0, then checks signal at start of epoch 1 and breaks.
        # So epochs_trained should be ~1 (depends on how stats is updated).
        # In code: self.current_epoch = epoch happens AFTER check?
        # No, let's check code:
        # for epoch in ...:
        #   if should_stop(): break
        #   self.current_epoch = epoch

        # So if it breaks at epoch 1, self.current_epoch remains 0.
        # stats["epochs_trained"] = self.current_epoch + 1 = 1.

        assert stats["epochs_trained"] == 1
        print("\nSuccessfully verified graceful shutdown logic!")


if __name__ == "__main__":
    t = TestDSLFMGracefulShutdown()
    # Manual setup for running verifying locally if needed
    pass
