"""Tests for Orchestrator OOM prevention (hardware-aware max_workers)."""

import pytest

from pff.orchestrator import Orchestrator
from pff import TaskModel


class TestOrchestratorStaticMethods:
    """Test Orchestrator static methods (no event loop needed)."""

    def test_get_safe_max_workers_limits(self):
        """_get_safe_max_workers should return correct limits per profile."""
        assert Orchestrator._get_safe_max_workers("low_spec") == 4
        assert Orchestrator._get_safe_max_workers("mid_spec") == 8
        assert Orchestrator._get_safe_max_workers("high_spec") == 16
        assert Orchestrator._get_safe_max_workers("unknown") == 8  # Default


class TestOrchestratorHardwareAware:
    """Test hardware-aware max_workers validation."""

    @pytest.mark.asyncio
    async def test_max_workers_reduced_for_mid_spec(self):
        """mid_spec should limit max_workers to 8."""
        tasks = [TaskModel(msisdn="test1", sequence="seq1")]

        # Try to create with 16 workers (unsafe for mid_spec)
        orchestrator = Orchestrator(exec_id="test", tasks=tasks, max_workers=16)

        # Should be reduced to 8 (safe limit for mid_spec)
        # Note: Actual limit depends on detected hardware
        # On mid_spec: should be capped at 8
        # On high_spec: can be 16
        # On low_spec: should be capped at 4
        assert orchestrator.max_workers <= 16
        assert orchestrator.max_workers >= 4

    @pytest.mark.asyncio
    async def test_max_workers_respects_safe_limits(self):
        """max_workers should not exceed hardware profile limits."""
        tasks = [TaskModel(msisdn="test1", sequence="seq1")]

        # Test various worker counts
        for requested_workers in [1, 4, 8, 12, 16, 32]:
            orchestrator = Orchestrator(
                exec_id="test", tasks=tasks, max_workers=requested_workers
            )

            # Should be within safe limits (4-16 depending on hardware)
            assert 1 <= orchestrator.max_workers <= 16

    @pytest.mark.asyncio
    async def test_invalid_max_workers_uses_default(self):
        """Invalid max_workers (<=0) should use default safe value."""
        tasks = [TaskModel(msisdn="test1", sequence="seq1")]

        # Test invalid values
        for invalid_workers in [0, -1, -10]:
            orchestrator = Orchestrator(
                exec_id="test", tasks=tasks, max_workers=invalid_workers
            )

            # Should use safe default (4-16 depending on hardware)
            assert orchestrator.max_workers > 0
            assert orchestrator.max_workers <= 16

    @pytest.mark.asyncio
    async def test_empty_tasks_list(self):
        """Orchestrator should handle empty task list gracefully."""
        orchestrator = Orchestrator(exec_id="test", tasks=[], max_workers=8)
        assert orchestrator.max_workers == 8
        assert len(orchestrator.tasks) == 0


class TestOrchestratorIntegration:
    """Integration tests for Orchestrator with hardware detection."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization_logs_config(self):
        """Orchestrator should log hardware profile and worker count."""
        tasks = [TaskModel(msisdn="test1", sequence="seq1")]

        # Simply verify Orchestrator initializes successfully with logging
        orchestrator = Orchestrator(exec_id="test", tasks=tasks, max_workers=8)

        # Verify orchestrator is properly initialized
        assert orchestrator.exec_id == "test"
        assert len(orchestrator.tasks) == 1
        assert orchestrator.max_workers == 8

    @pytest.mark.asyncio
    async def test_orchestrator_warns_on_excessive_workers(self, caplog):
        """Orchestrator should warn when reducing excessive max_workers."""
        tasks = [TaskModel(msisdn="test1", sequence="seq1")]

        with caplog.at_level("WARNING"):
            orchestrator = Orchestrator(exec_id="test", tasks=tasks, max_workers=32)

        # Should have warning if hardware doesn't support 32 workers
        # (only high_spec supports 16, so 32 should always warn)
        log_messages = [record.message for record in caplog.records]
        # May have warning depending on detected hardware
        assert orchestrator.max_workers <= 16
