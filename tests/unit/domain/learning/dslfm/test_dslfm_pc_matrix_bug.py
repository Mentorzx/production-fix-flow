import torch
import torch.nn as nn

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


class TestDSLFMProbeMatrixBug:
    def test_pc_log_prob_matrix_tensor_mismatch(self):
        """
        Reproduce the RuntimeError: Sizes of tensors must match except in dimension 0.

        Scenario:
        - num_heads = 10 (chunk size 10)
        - num_tails = 559 (chunk size 500)
        - tail chunks will be: 500, 59

        Current buggy implementation attempts to cat [(10, 500), (10, 59)] along dim=0.
        This fails because dim=1 (500 vs 59) doesn't match.
        """
        config = DSLFMKGCConfig(
            num_entities=1000,
            num_relations=10,
            entity_dim=16,
            feature_dim=16,
            lambda_pc=1.0,  # Enable PC logic
            max_communities=16,
            smoothing_epsilon=1e-6,
        )

        model = DSLFMKGCModel(config)

        # Mock the PC model using a dummy nn.Module
        class DummyPCModel(nn.Module):
            def log_prob(self, attr_probs, labels):
                return torch.zeros(labels.shape)

            def matrix_log_prob(self, z_heads, z_tails):
                return torch.zeros((z_heads.shape[0], z_tails.shape[0]))

        model.pc_model = DummyPCModel()

        # Prepare dummy embeddings
        # We need z_heads and all_z used in _pc_log_prob_matrix
        # z_h is (num_heads, dim)
        # chunk_z comes from all_z (num_tails, dim)

        num_heads = 10
        num_tails = 559
        dim = 16

        z_heads = torch.randn(num_heads, dim)
        z_tails = torch.randn(num_tails, dim)  # this acts as all_z in the method

        # Force CPU path where max_tails_chunk = 5000 is default, but we need to trigger split.
        # We can mock the ranges or monkeypatch, but better is to just call the method directly
        # and rely on its internal loop structure if we can influence chunk size.
        # The chunks are hardcoded:
        # if device == cuda: tails=500
        # else: tails=5000

        # To trigger the bug on CPU (tails=5000), we need > 5000 tails.
        # OR we can simply mock "torch.cat" or similar? No.
        # Best is to subclass or mock the chunk sizes inside the method...
        # But they are local variables.
        #
        # Let's try to run this test assuming we can simulate the condition.
        # If we run on CPU, we need num_tails = 5059 for example.
        # That's fine, it's just tensors.

        # NOTE: The bug report showed "Expected size 500 but got size 59".
        # This implies chunk size was 500. This implies CUDA path OR user changed code?
        # The user traceback shows:
        # File "/home/Alex/Development/PFF/pff/domain/learning/dslfm/dslfm_kgc.py", line 724
        # max_tails_chunk = 500
        # This suggests the user was on CUDA? Or the logic picked CUDA path.
        # The traceback shows `device=device` in `labels = ...`

        # We can force the "buggy loop" by calling the logic that sets variables.
        # Actually, let's just make the input large enough for CPU default (5000)
        # num_tails = 5500

        num_heads = 10
        num_tails = 5100  # 5000 + 100

        z_heads = torch.randn(num_heads, dim)
        z_tails = torch.randn(num_tails, dim)

        # The method is _pc_log_prob_matrix(z_h, chunk_z)
        # Wait, the traceback says:
        # evaluate -> _pc_log_prob_matrix(z_h, chunk_z)
        # In evaluate using triton/torch backend?
        # Let's look at `dslfm_kgc.py` evaluate method again.

        # It calls `self._pc_log_prob_matrix(z_h, chunk_z)`
        # `z_h` is heads, `chunk_z` is "chunk of entities" (tails).

        # If Evaluate breaks chunks itself, then `_pc_log_prob_matrix` receives the Whole Set of Tails?
        # "chunk_z" name suggests it's already a chunk?
        # But inside `_pc_log_prob_matrix` it iterates `all_z` (which is the 2nd arg).
        # So yes, it treats 2nd arg as "all candidate tails for this batch".

        # The fix should allow this to pass and return (num_heads, num_tails)
        result = model._pc_log_prob_matrix(z_heads, z_tails)

        assert result.shape == (num_heads, num_tails)
        print("\nSuccessfully verified fix for tensor shape bug!")


if __name__ == "__main__":
    # verification mode
    t = TestDSLFMProbeMatrixBug()
    t.test_pc_log_prob_matrix_tensor_mismatch()
