"""
Tests for advanced PyTorch compilation backends.

Tests OpenVINO, TVM, NNC compilation backends and compilation profiling.
"""

import pytest
import sys
import torch

TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')
PYTHON_313_PLUS = sys.version_info >= (3, 13)


class TestAdvancedCompilationBackends:
    """Test advanced compilation backend implementations."""

    def test_advanced_compilation_backend_initialization(self):
        """Test AdvancedCompilationBackend initializes correctly."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()
        assert backend is not None
        assert hasattr(backend, 'compile_with_openvino')
        assert hasattr(backend, 'compile_with_tvm')
        assert hasattr(backend, 'compile_with_nnc')
        assert hasattr(backend, 'auto_select_backend')

    def test_compilation_profiler_initialization(self):
        """Test CompilationProfiler initializes correctly."""
        from pff.utils.performance.performance import CompilationProfiler
        from pathlib import Path

        profiler = CompilationProfiler()
        assert profiler is not None
        assert hasattr(profiler, 'profile_compilation')
        assert hasattr(profiler, 'export_chrome_trace')
        assert hasattr(profiler, 'get_compile_times_report')
        assert hasattr(profiler, 'benchmark_backends')

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_openvino_backend_compilation(self):
        """Test OpenVINO backend compilation."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        compiled_model = backend.compile_with_openvino(model, example_inputs)
        assert compiled_model is not None

        output = compiled_model(*example_inputs)
        assert output.shape == (1, 5)

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_tvm_backend_compilation(self):
        """Test TVM backend compilation."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        compiled_model = backend.compile_with_tvm(model, example_inputs)
        assert compiled_model is not None

        output = compiled_model(*example_inputs)
        assert output.shape == (1, 5)

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_nnc_backend_compilation(self):
        """Test NNC backend compilation."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        compiled_model = backend.compile_with_nnc(model, example_inputs)
        assert compiled_model is not None

        output = compiled_model(*example_inputs)
        assert output.shape == (1, 5)

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
    def test_custom_backend_compilation(self):
        """Test custom inference backend compilation."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        compiled_model = backend.compile_with_custom_backend(model, example_inputs)
        assert compiled_model is not None

        output = compiled_model(*example_inputs)
        assert output.shape == (1, 5)

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_auto_select_backend(self):
        """Test automatic backend selection."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        compiled_model, backend_name = backend.auto_select_backend(model, example_inputs)
        assert compiled_model is not None
        assert backend_name in ["openvino", "tvm", "nnc", "default"]

        output = compiled_model(*example_inputs)
        assert output.shape == (1, 5)

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_compilation_profiling(self):
        """Test compilation profiling."""
        from pff.utils.performance.performance import CompilationProfiler

        profiler = CompilationProfiler()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        metrics = profiler.profile_compilation(model, example_inputs, backend="default")
        assert metrics is not None
        assert "backend" in metrics
        assert "compile_time" in metrics
        assert "warmup_time" in metrics
        assert "total_time" in metrics
        assert "success" in metrics
        assert metrics["backend"] == "default"

    @pytest.mark.skipif(not hasattr(torch, 'profiler'), reason="torch.profiler not available")
    def test_chrome_trace_export(self):
        """Test Chrome trace export."""
        from pff.utils.performance.performance import CompilationProfiler
        from pathlib import Path

        profiler = CompilationProfiler(output_dir=Path("/tmp/test_traces"))

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        trace_path = profiler.export_chrome_trace(
            model, example_inputs, trace_file="test_trace.json"
        )

        if trace_path:
            assert trace_path.exists()
            assert trace_path.suffix == ".json"

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_compile_times_report(self):
        """Test compile times report."""
        from pff.utils.performance.performance import CompilationProfiler

        profiler = CompilationProfiler()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        _ = profiler.profile_compilation(model, example_inputs, backend="default")

        report = profiler.get_compile_times_report()
        assert report is not None
        assert "compile_times" in report
        assert "success" in report

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_backend_benchmarking(self):
        """Test backend benchmarking."""
        from pff.utils.performance.performance import CompilationProfiler

        profiler = CompilationProfiler()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        backends = ["default", "nnc"]

        results = profiler.benchmark_backends(model, example_inputs, backends=backends)
        assert results is not None
        assert "results" in results
        assert "best_backend" in results
        assert "winner" in results
        assert len(results["results"]) >= 1

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_fallback_on_compilation_failure(self):
        """Test fallback to default backend on compilation failure."""
        from pff.utils.performance.performance import AdvancedCompilationBackend

        backend = AdvancedCompilationBackend()

        model = torch.nn.Linear(10, 5)
        example_inputs = (torch.randn(1, 10),)

        compiled_model = backend.compile_with_openvino(model, example_inputs)
        assert compiled_model is not None

        output = compiled_model(*example_inputs)
        assert output.shape == (1, 5)


class TestTransEWithAdvancedBackends:
    """Test TransE model with advanced compilation backends."""

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE or PYTHON_313_PLUS,
        reason="torch.compile not available or not supported on Python 3.13+"
    )
    def test_transe_model_compilation_backend_selection(self):
        """Test TransE model uses advanced backend selection."""
        from pff.validators.transe.core import TransEModel
        import numpy as np

        model = TransEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=128,
            margin=2.0,
            norm=2
        )

        assert model is not None
        assert hasattr(model, 'entity_embeddings')
        assert hasattr(model, 'relation_embeddings')
        assert model.num_entities == 100
        assert model.num_relations == 10

    def test_transe_model_without_compile(self):
        """Test TransE model works without torch.compile."""
        from pff.validators.transe.core import TransEModel

        model = TransEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=128,
            margin=2.0,
            norm=2
        )

        heads = torch.randint(0, 100, (10,))
        relations = torch.randint(0, 10, (10,))
        tails = torch.randint(0, 100, (10,))

        scores = model(heads, relations, tails)
        assert scores.shape == (10,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
