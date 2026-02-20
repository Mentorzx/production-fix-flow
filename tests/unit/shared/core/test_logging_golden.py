"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/core/test_logging_golden.py

"""

import asyncio
import importlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

# Robustly import the module, avoiding potential shadowing by the 'logger' object in __init__.py
# Use a specific variable name to access the package
import pff.shared.core.logging as pff_logger_pkg

# -----------------------------------------------------------------------------
# 1. Import/Config & 2. Format/Invariants
# -----------------------------------------------------------------------------


class TestLoggerConfigAndFormat:
    """Represent TestLoggerConfigAndFormat."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Execute clean env.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        original_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original_env)

        # We need to reload the config module to reset sinks
        import pff.shared.core.logging.config

        importlib.reload(pff.shared.core.logging.config)
        # Reload package to update references
        importlib.reload(pff_logger_pkg)

    def test_smoke_import(self):
        """Execute test smoke import."""

        importlib.reload(pff_logger_pkg)
        assert pff_logger_pkg.logger is not None

    def test_env_vars_log_level(self, tmp_path):
        """Execute test env vars log level.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        env_vars = {
            "LOG_LEVEL": "WARNING",
            "FILE_LOG_LEVEL": "WARNING",
            "LOG_DIR": str(tmp_path),
        }
        with patch.dict(os.environ, env_vars):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            pff_logger_pkg.logger.warning("This is a warning")
            pff_logger_pkg.logger.info("This is info")
            pff_logger_pkg.logger.complete()

            files = list(tmp_path.glob("*.log"))
            assert len(files) > 0
            content = files[0].read_text(encoding="utf-8")

            assert "This is a warning" in content
            assert "This is info" not in content

    def test_env_vars_log_dir(self, tmp_path):
        """Execute test env vars log dir.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        target_dir = tmp_path / "custom_logs"
        with patch.dict(os.environ, {"LOG_DIR": str(target_dir)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            pff_logger_pkg.logger.info("Test log")
            pff_logger_pkg.logger.complete()

            assert target_dir.exists()
            assert len(list(target_dir.glob("*.log"))) > 0

    def test_placeholder_safety(self, tmp_path):
        """Execute test placeholder safety.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            try:
                pff_logger_pkg.logger.info("Simple message")
            except Exception as e:
                pytest.fail(f"Logger raised exception: {e}")

            pff_logger_pkg.logger.complete()
            files = list(tmp_path.glob("*.log"))
            content = files[0].read_text(encoding="utf-8")
            assert "Simple message" in content

            import orjson

            lines = content.strip().split("\n")
            data = orjson.loads(lines[-1])
            # Correct JSON path for Loguru serialize=True
            assert data["record"]["extra"]["task_id"] == "MAIN"

    def test_unicode_and_huge_lines(self, tmp_path):
        """Execute test unicode and huge lines.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            unicode_msg = "Olá Mundo 🌍 🚀 中文"
            pff_logger_pkg.logger.info(unicode_msg)
            huge_msg = "A" * (1024 * 1024)
            pff_logger_pkg.logger.info(huge_msg)
            pff_logger_pkg.logger.complete()

            files = list(tmp_path.glob("*.log"))
            content = files[0].read_text(encoding="utf-8")
            assert unicode_msg in content
            assert len(content) > 1024 * 1024


# -----------------------------------------------------------------------------
# 3. File Operations
# -----------------------------------------------------------------------------


class TestLoggerFileOps:
    """Represent TestLoggerFileOps."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Execute clean env."""

        original_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original_env)
        import pff.shared.core.logging.config

        importlib.reload(pff.shared.core.logging.config)
        importlib.reload(pff_logger_pkg)

    def test_file_write_and_rotation(self, tmp_path):
        """Execute test file write and rotation.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(
            os.environ,
            {
                "LOG_DIR": str(tmp_path),
                "LOG_ROTATION": "500 B",
                "LOG_COMPRESSION": "zip",
            },
        ):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            for i in range(10):
                pff_logger_pkg.logger.info(f"Msg {i} padding " * 5)

            time.sleep(1.0)
            assert len(list(tmp_path.glob("*.log"))) >= 1
            assert len(list(tmp_path.glob("*.zip"))) >= 1

    def test_human_readable_split_logs(self, tmp_path):
        """Execute test human readable split logs.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            pff_logger_pkg.logger.info("Human info")
            pff_logger_pkg.logger.warning("Human warning")
            pff_logger_pkg.logger.error("Human error")
            pff_logger_pkg.logger.complete()

            readable_dir = tmp_path / "readable"
            assert readable_dir.exists()

            info_files = list(readable_dir.glob("*.info.log"))
            warning_files = list(readable_dir.glob("*.warning.log"))
            error_files = list(readable_dir.glob("*.error.log"))

            assert info_files
            assert warning_files
            assert error_files
            assert "Human info" in info_files[0].read_text(encoding="utf-8")
            assert "Human warning" in warning_files[0].read_text(encoding="utf-8")
            assert "Human error" in error_files[0].read_text(encoding="utf-8")

    def test_retention(self, tmp_path):
        """Execute test retention.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        old_file = tmp_path / "2020-01-01.log"
        old_file.write_text("Old logs")
        import time

        old_time = time.time() - (365 * 24 * 3600)
        os.utime(old_file, (old_time, old_time))

        with patch.dict(
            os.environ,
            {
                "LOG_DIR": str(tmp_path),
                "LOG_ROTATION": "100 B",
                "LOG_RETENTION": "1 day",
            },
        ):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            for _ in range(5):
                pff_logger_pkg.logger.info("Trigger cleanup")
            time.sleep(0.5)
            assert not old_file.exists()


# -----------------------------------------------------------------------------
# 4. Concurrency
# -----------------------------------------------------------------------------


class TestLoggerConcurrency:
    """Represent TestLoggerConcurrency."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Execute clean env."""

        original_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original_env)
        import pff.shared.core.logging.config

        importlib.reload(pff.shared.core.logging.config)
        importlib.reload(pff_logger_pkg)

    def test_multi_thread_integrity(self, tmp_path):
        """Execute test multi thread integrity.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            thread_count = 10
            logs_per_thread = 50

            def worker(tid):
                """Execute worker.



                Args:

                    tid: Input value used by this callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                for i in range(logs_per_thread):
                    pff_logger_pkg.logger.info(f"Thread-{tid} msg {i}")

            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(worker, i) for i in range(thread_count)]
                for f in futures:
                    f.result()

            pff_logger_pkg.logger.complete()
            files = list(tmp_path.glob("*.log"))
            lines = files[0].read_text(encoding="utf-8").strip().split("\n")
            data_lines = [line for line in lines if "Thread-" in line]
            assert len(data_lines) == thread_count * logs_per_thread

    @pytest.mark.asyncio
    async def test_asyncio_context_isolation(self, tmp_path):
        """Execute test asyncio context isolation.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            async def worker(name, trace_val, count):
                """Execute worker.



                Args:

                    name: Input value used by this callable.

                    trace_val: Input value used by this callable.

                    count: Input value used by this callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                pff_logger_pkg.bind_trace_id(trace_val)
                for i in range(count):
                    pff_logger_pkg.logger.info(f"Task-{name} msg {i}")
                    await asyncio.sleep(0.001)

            await asyncio.gather(worker("A", "TA", 20), worker("B", "TB", 20))
            pff_logger_pkg.logger.complete()

            files = list(tmp_path.glob("*.log"))
            import orjson

            for line in files[0].read_text(encoding="utf-8").strip().split("\n"):
                data = orjson.loads(line)
                # Correct JSON path
                msg = data["record"]["message"]
                if "Task-A" in msg:
                    assert data["record"]["extra"]["trace_id"] == "TA"
                elif "Task-B" in msg:
                    assert data["record"]["extra"]["trace_id"] == "TB"


# -----------------------------------------------------------------------------
# 5. OTel
# -----------------------------------------------------------------------------


class TestLoggerOTel:
    """Represent TestLoggerOTel."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Execute clean env."""

        original_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original_env)
        import pff.shared.core.logging.config

        importlib.reload(pff.shared.core.logging.config)
        importlib.reload(pff_logger_pkg)

    def test_otel_span_propagation(self, tmp_path):
        """Execute test otel span propagation.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            with pff_logger_pkg.start_span("root") as span:
                _ = span
                pff_logger_pkg.logger.info("Inside")
                ctx = pff_logger_pkg.TraceContext.get()
                assert ctx["trace_id"] is not None

            pff_logger_pkg.logger.info("Outside")
            ctx = pff_logger_pkg.TraceContext.get()
            assert ctx["trace_id"] is None

            pff_logger_pkg.logger.complete()
            files = list(tmp_path.glob("*.log"))
            import orjson

            for line in files[0].read_text(encoding="utf-8").strip().split("\n"):
                data = orjson.loads(line)
                # Correct JSON path
                msg = data["record"]["message"]
                extra = data["record"]["extra"]
                if "Inside" in msg:
                    assert "trace_id" in extra
                elif "Outside" in msg:
                    assert "trace_id" not in extra or extra["trace_id"] is None


# -----------------------------------------------------------------------------
# 6. Utilities
# -----------------------------------------------------------------------------


class TestLoggerUtilities:
    """Represent TestLoggerUtilities."""

    def test_timeit_decorator(self):
        """Execute test timeit decorator.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        logs = []
        sink_id = pff_logger_pkg.logger.add(logs.append, format="{message}")
        try:

            @pff_logger_pkg.timeit
            def fast():
                """Execute fast.



                Returns:

                    Return value produced by the callable.

                """

                return "done"

            assert fast() == "done"
            assert any("fast took" in m for m in logs)
        finally:
            pff_logger_pkg.logger.remove(sink_id)

    def test_catch_decorator(self, tmp_path):
        # Use file verification to avoid sink recursion issues
        """Execute test catch decorator.



        Args:

            tmp_path: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            @pff_logger_pkg.catch(reraise=False, default="safe")
            def fail():
                """Execute fail."""

                raise ValueError("UniqueBoomError")

            assert fail() == "safe"

            pff_logger_pkg.logger.complete()
            files = list(tmp_path.glob("*.log"))
            content = files[0].read_text(encoding="utf-8")

            assert "Error in" in content
            assert "UniqueBoomError" in content

    def test_suppress_output(self, capsys):
        """Execute test suppress output.



        Args:

            capsys: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with pff_logger_pkg.suppress_output():
            print("Hidden")
        assert "Hidden" not in capsys.readouterr().out
        print("Visible")
        assert "Visible" in capsys.readouterr().out

    def test_silence_libs(self):
        """Execute test silence libs."""

        name = "test_lib"
        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(logging.INFO)
        pff_logger_pkg.silence_libs(name, level="ERROR")
        assert logger_instance.level == logging.ERROR


# -----------------------------------------------------------------------------
# 7. Stdlib
# -----------------------------------------------------------------------------


class TestStdlibIntegration:
    """Represent TestStdlibIntegration."""

    def test_stdlib_capture(self, tmp_path):
        """Execute test stdlib capture.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with patch.dict(os.environ, {"LOG_DIR": str(tmp_path)}):
            import pff.shared.core.logging.config as cfg

            importlib.reload(cfg)
            importlib.reload(pff_logger_pkg)

            logging.getLogger("ext.lib").warning("From stdlib")
            pff_logger_pkg.logger.complete()
            files = list(tmp_path.glob("*.log"))
            assert "From stdlib" in files[0].read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# 8. Reorderer
# -----------------------------------------------------------------------------


class TestLogReorderer:
    """Represent TestLogReorderer."""

    def test_reorderer_json(self, tmp_path):
        """Execute test reorderer json.



        Args:

            tmp_path: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        f = tmp_path / "messy.log"
        import orjson

        lines = []
        for i in range(4):
            rec = {
                "record": {
                    "thread": {"name": f"T{i % 2}"},
                    "extra": {"task_id": f"K{i}"},
                    "message": f"M{i}",
                },
                "text": f"M{i}",
            }
            lines.append(orjson.dumps(rec).decode())
        f.write_text("\n".join(lines))

        pff_logger_pkg.LogReorderer.reorder(f)
        content = f.read_text(encoding="utf-8")
        assert "===== THREAD T0 =====" in content
        assert "===== THREAD T1 =====" in content
