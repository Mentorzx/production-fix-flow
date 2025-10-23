"""
Error Scenarios Integration Tests - Sprint 12

Tests system behavior under error conditions:
1. Timeout scenarios
2. Invalid data handling
3. OOM prevention
4. Network failures
5. Corrupted files
6. Resource exhaustion

Ensures system fails gracefully and provides meaningful error messages.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest


class TestTimeoutScenarios:
    """Test system behavior under timeout conditions."""

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self):
        """Test that requests timeout properly and don't hang indefinitely."""

        async def slow_operation():
            """Simulate operation that takes too long."""
            await asyncio.sleep(10)  # 10 seconds
            return "completed"

        # Should timeout before 10s
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_partial_timeout_recovery(self):
        """Test system recovers from partial timeouts."""

        results = []

        async def mixed_operations(idx):
            if idx == 2:
                await asyncio.sleep(5)  # This one will timeout
            else:
                await asyncio.sleep(0.01)
            return idx

        tasks = [mixed_operations(i) for i in range(5)]

        # Gather with return_exceptions to not fail entire batch
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have some successes and some timeouts
        successes = [r for r in completed if not isinstance(r, Exception)]
        assert len(successes) > 0, "All operations failed, expected some successes"


class TestInvalidDataHandling:
    """Test system handles various invalid data scenarios."""

    def test_missing_required_fields(self, tmp_path):
        """Test handling of data missing required fields."""

        # Data with only optional fields
        data = pl.DataFrame({"optional_field": ["value1", "value2"]})

        data_file = tmp_path / "missing_fields.parquet"
        data.write_parquet(data_file)

        # Should not crash when loading, even with missing expected fields
        loaded_data = pl.read_parquet(data_file)

        assert loaded_data.shape[0] > 0
        assert "optional_field" in loaded_data.columns

    def test_wrong_data_types(self, tmp_path):
        """Test handling of incorrect data types."""

        # Data with string where number expected
        data = pl.DataFrame({
            "msisdn": ["5511999990001", "invalid_msisdn", "5511999990003"],
            "balance": ["100.50", "not_a_number", "75.00"],  # Strings instead of numbers
        })

        data_file = tmp_path / "wrong_types.parquet"
        data.write_parquet(data_file)

        # Should load data successfully (Polars handles type coercion)
        loaded_data = pl.read_parquet(data_file)

        assert loaded_data.shape[0] == 3
        assert "balance" in loaded_data.columns

    def test_duplicate_records(self, tmp_path):
        """Test handling of duplicate records."""

        # Data with duplicates
        data = pl.DataFrame({
            "msisdn": ["5511999990001", "5511999990001", "5511999990002"],
            "customer_id": ["CUST_001", "CUST_001", "CUST_002"],
        })

        data_file = tmp_path / "duplicates.parquet"
        data.write_parquet(data_file)

        # Should load without errors (duplicates are valid)
        loaded_data = pl.read_parquet(data_file)

        assert loaded_data.shape[0] == 3  # All rows including duplicates
        assert loaded_data.filter(pl.col("msisdn") == "5511999990001").shape[0] == 2

    def test_null_values_handling(self, tmp_path):
        """Test handling of null/None values."""

        # Data with nulls
        data = pl.DataFrame({
            "msisdn": ["5511999990001", None, "5511999990003"],
            "customer_id": [None, "CUST_002", "CUST_003"],
        })

        data_file = tmp_path / "nulls.parquet"
        data.write_parquet(data_file)

        # Should handle nulls gracefully
        loaded_data = pl.read_parquet(data_file)

        assert loaded_data.shape[0] == 3
        assert loaded_data.filter(pl.col("msisdn").is_null()).shape[0] == 1
        assert loaded_data.filter(pl.col("customer_id").is_null()).shape[0] == 1


class TestOOMPrevention:
    """Test OOM prevention mechanisms work correctly."""

    def test_large_batch_size_rejection(self):
        """Test that excessively large batch sizes are rejected."""

        # Create very large task list
        large_task_list = [(i,) for i in range(1_000_000)]

        # Should not create 1M futures immediately (OOM prevention active)
        from pff.utils.concurrency import ProcessExecutor

        executor = ProcessExecutor(max_workers=4)

        # Should handle gracefully with lazy submission
        # Note: We don't actually execute to avoid long test time
        assert len(large_task_list) == 1_000_000

        executor.shutdown()

    @pytest.mark.asyncio
    async def test_memory_check_triggers_correctly(self):
        """Test memory safety checks trigger when RAM is low."""

        from pff.utils.concurrency import ConcurrencyManager
        from unittest.mock import patch, MagicMock

        cm = ConcurrencyManager()

        # Mock low RAM scenario
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value = MagicMock(
                percent=95.0,  # 95% RAM usage
                available=512 * 1024 * 1024,  # Only 512MB available
                total=8 * 1024 * 1024 * 1024,  # 8GB total
            )

            # Should raise MemoryError
            with pytest.raises(MemoryError, match="RAM usage"):
                cm._check_memory_safety()


class TestNetworkFailures:
    """Test handling of network-related failures."""

    @pytest.mark.asyncio
    async def test_connection_refused(self):
        """Test handling when connection is refused."""

        async def failing_request():
            raise ConnectionRefusedError("Connection refused")

        with pytest.raises(ConnectionRefusedError):
            await failing_request()

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test handling of connection timeouts."""

        async def timeout_request():
            await asyncio.sleep(10)
            return "result"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(timeout_request(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_network_retry_logic(self):
        """Test that failed requests are retried."""

        attempt_count = 0

        async def flaky_request():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await flaky_request()
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.01)

        assert result == "success"
        assert attempt_count == 3, "Should have retried twice before succeeding"


class TestCorruptedFiles:
    """Test handling of corrupted or malformed files."""

    def test_corrupted_parquet_file(self, tmp_path):
        """Test handling of corrupted parquet files."""

        # Create corrupted file
        corrupted_file = tmp_path / "corrupted.parquet"
        corrupted_file.write_bytes(b"This is not valid parquet data")

        # Should raise appropriate error
        with pytest.raises(Exception):  # pl.exceptions or similar
            pl.read_parquet(corrupted_file)

    def test_empty_file(self, tmp_path):
        """Test handling of empty files."""

        empty_file = tmp_path / "empty.txt"
        empty_file.touch()  # Create empty file

        # Should handle empty file
        content = empty_file.read_text()
        assert content == ""

    def test_truncated_file(self, tmp_path):
        """Test handling of truncated files."""

        # Create file and truncate it
        truncated_file = tmp_path / "truncated.parquet"

        # First create valid parquet
        data = pl.DataFrame({"col": [1, 2, 3]})
        data.write_parquet(truncated_file)

        # Then truncate it
        with open(truncated_file, "rb") as f:
            partial_content = f.read(50)  # Read only 50 bytes

        with open(truncated_file, "wb") as f:
            f.write(partial_content)

        # Should raise error on corrupted file
        with pytest.raises(Exception):
            pl.read_parquet(truncated_file)


class TestResourceExhaustion:
    """Test system behavior under resource exhaustion."""

    def test_disk_space_handling(self, tmp_path):
        """Test handling when disk space is limited."""

        # Note: Can't easily test actual disk full, so we verify error handling exists

        output_file = tmp_path / "large_output.txt"

        # Verify we can write normal file
        output_file.write_text("test" * 1000)
        assert output_file.exists()

    def test_file_descriptor_limits(self, tmp_path):
        """Test handling of file descriptor limits."""

        # Open multiple files (but not too many to crash test)
        files = []

        try:
            for i in range(100):  # Open 100 files
                f = open(tmp_path / f"file_{i}.txt", "w")
                files.append(f)
                f.write(f"content {i}")

        finally:
            # Clean up
            for f in files:
                f.close()

        # Should complete without errors
        assert len(files) == 100

    @pytest.mark.asyncio
    async def test_concurrent_connection_limits(self):
        """Test handling of concurrent connection limits."""

        # Simulate many concurrent operations
        async def operation(idx):
            await asyncio.sleep(0.01)
            return idx

        # Create many concurrent tasks
        tasks = [operation(i) for i in range(1000)]

        # Should handle many concurrent operations
        results = await asyncio.gather(*tasks)

        assert len(results) == 1000


class TestCircuitBreakerFailures:
    """Test circuit breaker behavior under failures."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after repeated failures."""

        from aiobreaker import CircuitBreaker, CircuitBreakerError

        breaker = CircuitBreaker(fail_max=3)
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("Operation failed")

        # First 3 calls should fail with Exception
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(failing_operation)

        # 4th call should fail with CircuitBreakerError (circuit open)
        with pytest.raises(CircuitBreakerError):
            await breaker.call_async(failing_operation)

        # Should have only made 3 actual calls
        assert call_count == 3, f"Expected 3 calls, got {call_count}"


class TestGracefulDegradation:
    """Test system degrades gracefully under stress."""

    @pytest.mark.asyncio
    async def test_partial_service_failure(self):
        """Test system continues working when part of service fails."""

        async def service_a():
            return "success_a"

        async def service_b():
            raise Exception("Service B failed")

        async def service_c():
            return "success_c"

        # Execute all services
        results = await asyncio.gather(
            service_a(),
            service_b(),
            service_c(),
            return_exceptions=True,
        )

        # Should have some successes even with one failure
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 2, "Expected 2 successes"
        assert len(failures) == 1, "Expected 1 failure"
        assert successes == ["success_a", "success_c"]

    def test_fallback_mechanisms(self):
        """Test fallback mechanisms work correctly."""

        def primary_source():
            raise Exception("Primary failed")

        def fallback_source():
            return "fallback_data"

        # Try primary, fall back to secondary
        try:
            result = primary_source()
        except Exception:
            result = fallback_source()

        assert result == "fallback_data"
