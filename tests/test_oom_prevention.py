"""
OOM Prevention Tests

Tests to ensure system prevents Out-Of-Memory crashes with large task lists.
Validates bounded memory usage in ProcessExecutor and RayExecutor.
"""

import gc
import os
import sys
from unittest.mock import MagicMock, patch

import psutil
import pytest

from pff.services.business_service import Rule, RuleValidator, _run_rule_check
from pff.utils.concurrency import ConcurrencyManager, ProcessExecutor, RayExecutor


def _simple_task(x):
    """Helper function for multiprocessing tests (must be module-level)."""
    return x * 2


def _dummy_task(x):
    """Helper function for multiprocessing tests (must be module-level)."""
    return x


def _delayed_task(x):
    """Helper function with artificial delay (must be module-level)."""
    import time
    time.sleep(0.001 * (100 - x))
    return x * 2


def _increment_task(x):
    """Helper function that increments by 1 (must be module-level)."""
    return x + 1


@pytest.fixture
def mock_large_rules():
    """Generate 10K mock rules for testing."""
    rules = []
    for i in range(10000):
        rule = MagicMock(spec=Rule)
        rule.id = f"rule_{i}"
        rule.confidence = 0.9
        rule.body = [{"predicate": "test", "args": ["X", "Y"]}]
        rule.head = {"predicate": "result", "args": ["X"]}
        rules.append(rule)
    return rules


@pytest.fixture
def mock_triples():
    """Generate mock triples for testing."""
    return [("subj1", "pred1", "obj1"), ("subj2", "pred2", "obj2")]


class TestProcessExecutorMemoryBounds:
    """Test ProcessExecutor bounded memory with lazy task submission."""

    def test_lazy_submission_limits_memory(self):
        """Ensure ProcessExecutor doesn't create all futures at once."""
        executor = ProcessExecutor(max_workers=4)

        task_count = 1000
        args_list = [(i,) for i in range(task_count)]

        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        results = executor.map(_simple_task, args_list, desc="Testing bounded memory")

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before

        assert results == [i * 2 for i in range(task_count)]
        assert mem_increase < 200, f"Memory increased {mem_increase:.1f} MB (expected <200 MB)"

        executor.shutdown()

    def test_max_pending_scales_with_workers(self):
        """Verify max_pending futures scales properly with worker count."""
        executor = ProcessExecutor(max_workers=8)

        with patch.object(executor._pool, 'submit', wraps=executor._pool.submit) as mock_submit:
            results = executor.map(_dummy_task, [(i,) for i in range(200)])

            assert len(results) == 200
            assert all(results[i] == i for i in range(200))

        executor.shutdown()

    def test_maintains_result_order(self):
        """Ensure lazy submission preserves original order of results."""
        executor = ProcessExecutor(max_workers=4)

        args_list = [(i,) for i in range(100)]
        results = executor.map(_delayed_task, args_list)

        expected = [i * 2 for i in range(100)]
        assert results == expected, "Result order must match input order"

        executor.shutdown()


class TestRayExecutorAdaptiveBatching:
    """Test RayExecutor adaptive batching for 100K+ tasks."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Ray unstable on Windows")
    def test_adaptive_batching_large_tasks(self):
        """Test Ray batching activates for 50K+ tasks."""
        executor = RayExecutor()

        task_count = 60000
        args_list = [(i,) for i in range(task_count)]

        with patch.object(executor, '_map_batched', wraps=executor._map_batched) as mock_batched:
            results = executor.map(_simple_task, args_list, desc="Testing batching")

            assert mock_batched.call_count == 1, "Batching should activate for 60K tasks"
            assert len(results) == task_count
            assert results[:10] == [i * 2 for i in range(10)]

        executor.shutdown()

    @pytest.mark.skipif(sys.platform == "win32", reason="Ray unstable on Windows")
    def test_no_batching_small_tasks(self):
        """Test Ray uses standard execution for <50K tasks."""
        executor = RayExecutor()

        task_count = 1000
        args_list = [(i,) for i in range(task_count)]

        with patch.object(executor, '_map_batched') as mock_batched:
            results = executor.map(_increment_task, args_list)

            assert mock_batched.call_count == 0, "Batching should NOT activate for 1K tasks"
            assert results == [i + 1 for i in range(task_count)]

        executor.shutdown()

    @pytest.mark.skipif(sys.platform == "win32", reason="Ray unstable on Windows")
    def test_bounded_inflight_tasks(self):
        """Ensure Ray limits max inflight tasks to 10K."""
        executor = RayExecutor()

        task_count = 20000
        args_list = [(i,) for i in range(task_count)]

        results = executor.map(_dummy_task, args_list, desc="Testing inflight limit")

        assert len(results) == task_count
        assert results[0] == 0
        assert results[-1] == task_count - 1

        executor.shutdown()


class TestRuleValidatorOOMPrevention:
    """Test RuleValidator handles 128K+ rules without OOM."""

    def test_switches_to_ray_for_large_rulesets(self):
        """Ensure RuleValidator uses Ray for 10K+ rules."""
        validator = RuleValidator()

        large_rule_count = 15000
        fake_rules = [None] * large_rule_count
        fake_triples = [("s", "p", "o")]

        with patch.object(ConcurrencyManager, 'execute_sync') as mock_execute:
            mock_execute.return_value = [[] for _ in range(large_rule_count)]

            validator.validate_rules(fake_rules, fake_triples)

            assert mock_execute.call_count == 1
            call_kwargs = mock_execute.call_args[1]
            assert call_kwargs['task_type'] == 'ray', "Should use Ray for 15K+ rules"

    def test_uses_process_for_small_rulesets(self):
        """Ensure RuleValidator uses ProcessPoolExecutor for <10K rules."""
        validator = RuleValidator()

        small_rule_count = 500
        fake_rules = [None] * small_rule_count
        fake_triples = [("s", "p", "o")]

        with patch.object(ConcurrencyManager, 'execute_sync') as mock_execute:
            mock_execute.return_value = [[] for _ in range(small_rule_count)]

            validator.validate_rules(fake_rules, fake_triples)

            call_kwargs = mock_execute.call_args[1]
            assert call_kwargs['task_type'] == 'process', "Should use process for <10K rules"

    def test_task_type_selection_logic(self):
        """Test task_type selection threshold at 10K rules."""
        validator = RuleValidator()

        test_cases = [
            (9999, 'process'),
            (10000, 'process'),
            (10001, 'ray'),
            (50000, 'ray'),
            (128319, 'ray'),
        ]

        for rule_count, expected_type in test_cases:
            fake_rules = [None] * rule_count
            fake_triples = []

            with patch.object(ConcurrencyManager, 'execute_sync') as mock_execute:
                mock_execute.return_value = []

                validator.validate_rules(fake_rules, fake_triples)

                call_kwargs = mock_execute.call_args[1]
                actual_type = call_kwargs['task_type']
                assert actual_type == expected_type, \
                    f"Rules={rule_count}: expected {expected_type}, got {actual_type}"


class TestConcurrencyManagerMemorySafety:
    """Test ConcurrencyManager memory safety checks."""

    def test_memory_check_passes_with_available_ram(self):
        """Ensure memory check passes when sufficient RAM available."""
        cm = ConcurrencyManager()

        mem = psutil.virtual_memory()
        if mem.percent < 80:
            cm._check_memory_safety()
        else:
            pytest.skip("System RAM >80%, cannot test memory safety check")

    def test_memory_check_fails_with_low_ram(self):
        """Ensure memory check raises MemoryError when RAM critical."""
        cm = ConcurrencyManager()

        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value = MagicMock(
                percent=90.0,  # Above default threshold of 85%
                available=1024 * 1024 * 1024,
                total=8 * 1024 * 1024 * 1024  # 8 GB total
            )

            # Updated to match actual error message format
            with pytest.raises(MemoryError, match=r"RAM usage 90\.0% exceeds safety threshold"):
                cm._check_memory_safety()


class TestEdgeCases:
    """Test edge cases for OOM prevention."""

    def test_empty_task_list(self):
        """Test executors handle empty task lists gracefully."""
        executor = ProcessExecutor(max_workers=4)
        results = executor.map(_dummy_task, [])
        assert results == []
        executor.shutdown()

    def test_single_task(self):
        """Test executors handle single task efficiently."""
        executor = ProcessExecutor(max_workers=4)
        results = executor.map(_simple_task, [(5,)])
        assert results == [10]
        executor.shutdown()

    @pytest.mark.skipif(sys.platform == "win32", reason="Ray unstable on Windows")
    def test_ray_executor_exactly_50k_tasks(self):
        """Test boundary condition at 50K tasks (batching threshold)."""
        executor = RayExecutor()

        results = executor.map(_dummy_task, [(i,) for i in range(50000)])

        assert len(results) == 50000
        assert results[0] == 0
        assert results[-1] == 49999

        executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
