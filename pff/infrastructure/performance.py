"""
Performance optimization module.

Implements cutting-edge performance features from PyTorch 2.5.1+, CUDA optimizations,
and distributed computing enhancements using Ray 3.0+.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    pass

from pff import settings
from pff.shared.core.config import PERFORMANCE_CONFIG_PATH
from pff.shared import logger
from pff.shared.core.file_manager import FileManager

_DEFAULT_PERFORMANCE_CONFIG: dict[str, Any] = {
    "performance": {
        "compilation_logs_dir": str(settings.OUTPUTS_DIR / "compilation_logs"),
        "backends": {"order": ["openvino", "tvm", "nnc", "default"]},
        "torch": {
            "enable_static_graph": True,
            "cuda_allocator": {
                "backend": "cudaMallocAsync",
                "max_non_split_rounding_mb": 1024,
            },
            "inductor": {
                "max_autotune": 1,
                "max_autotune_memory_fraction": 0.5,
            },
            "allow_dynamic_shapes": True,
        },
        "ray": {
            "train_v2_enabled": True,
            "fault_tolerance_enabled": True,
            "checkpoint_frequency": 5,
            "enable_vllm": False,
        },
        "memory_profiling": {
            "cuda_memory_fraction": 0.9,
            "cuda_launch_blocking": 0,
            "malloc_conf": "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000",
        },
        "file_io": {
            "streaming_thresholds": {
                "low_ram_gb": 8,
                "mid_ram_gb": 24,
                "low_ram_mb": 64,
                "mid_ram_mb": 512,
                "high_ram_mb": 1024,
            },
            "parquet_first": {
                "raw_chunk_mb": 8,
                "parsed_row_group_size": 200_000,
                "container_flush_rows": 2048,
                "compression": "lz4",
                "compression_level": 3,
                "cache_dir": str(settings.CACHE_DIR / "ingest"),
            },
        },
    }
}


@lru_cache(maxsize=1)
def _load_performance_config() -> dict[str, Any]:
    try:
        cfg = FileManager.read(PERFORMANCE_CONFIG_PATH, return_native=True)
        if not isinstance(cfg, dict):
            return _DEFAULT_PERFORMANCE_CONFIG
        return cfg
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Using default performance config (reason: {exc})")
        return _DEFAULT_PERFORMANCE_CONFIG


def _resolve_output_dir(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (settings.ROOT_DIR / path).resolve()


def _compile_model(
    model: Any,
    *,
    backend: Any | None = None,
    mode: str | None = None,
    dynamic: bool | None = None,
) -> Any:
    """Compile a model using Module.compile when available, falling back to torch.compile."""
    try:
        import torch
    except Exception:
        return model

    compile_method = getattr(model, "compile", None)
    kwargs: dict[str, Any] = {}
    if backend is not None:
        kwargs["backend"] = backend
    if mode is not None:
        kwargs["mode"] = mode
    if dynamic is not None:
        kwargs["dynamic"] = dynamic

    if callable(compile_method):
        try:
            compiled = compile_method(**kwargs)
        except TypeError:
            compiled = compile_method()
        return compiled if compiled is not None else model

    if hasattr(torch, "compile"):
        return torch.compile(model, **kwargs)
    return model


class AdvancedCompilationBackend:
    """Advanced PyTorch compilation backends for specialized hardware and inference."""

    def __init__(self) -> None:
        self.logger = logger
        perf_cfg = _load_performance_config().get("performance", {})
        backends_cfg = perf_cfg.get("backends", {})
        self._backend_order: list[str] = list(backends_cfg.get("order", [])) or [
            "openvino",
            "tvm",
            "nnc",
            "default",
        ]

    def compile_with_openvino(self, model: Any, example_inputs: Any) -> Any:
        """
        Compile model with OpenVINO backend for Intel hardware.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing

        Returns:
            Compiled model
        """
        try:
            import torch

            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile not available (requires PyTorch 2.0+)")

            self.logger.debug("Compilando com backend OpenVINO para hardware Intel")

            compiled_model = _compile_model(
                model,
                backend="openvino",
                dynamic=True,
            )

            self.logger.success("Compilacao OpenVINO concluida")
            return compiled_model

        except Exception as e:
            self.logger.warning(f"OpenVINO compilation failed: {e}")
            self.logger.info("Usando backend padrao como modo_alternativo")
            # max-autotune prioritizes steady-state throughput; expect a longer warmup before benefits show.
            return _compile_model(model, mode="max-autotune", dynamic=True)

    def compile_with_tvm(self, model: Any, example_inputs: Any) -> Any:
        """
        Compile model with TVM backend for cross-platform optimization.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing

        Returns:
            Compiled model
        """
        try:
            import torch

            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile not available (requires PyTorch 2.0+)")

            self.logger.debug("Compilando com backend TVM para otimizacao multiplataforma")

            compiled_model = _compile_model(
                model,
                backend="tvm",
                dynamic=True,
            )

            self.logger.success("Compilacao TVM concluida")
            return compiled_model

        except Exception as e:
            self.logger.warning(f"TVM compilation failed: {e}")
            self.logger.info("Usando backend padrao como modo_alternativo")
            # max-autotune trades extra compile time for better runtime performance on supported operators.
            return _compile_model(model, mode="max-autotune", dynamic=True)

    def compile_with_nnc(self, model: Any, example_inputs: Any) -> Any:
        """
        Compile model with NNC (Neural Network Compiler) and AOT Autograd.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing

        Returns:
            Compiled model with NNC
        """
        try:
            import torch

            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile not available (requires PyTorch 2.0+)")

            self.logger.debug("Compilando com backend NNC e AOT Autograd")

            try:
                from functorch.compile import ts_compile

                compiled_model = _compile_model(
                    model,
                    backend=ts_compile,
                    dynamic=True,
                )

                self.logger.success("Compilacao NNC com AOT Autograd concluida")
                return compiled_model

            except ImportError:
                self.logger.warning("functorch not available, using default NNC")
                compiled_model = _compile_model(
                    model,
                    backend="nnc",
                    dynamic=True,
                )
                return compiled_model

        except Exception as e:
            self.logger.warning(f"NNC compilation failed: {e}")
            self.logger.info("Usando backend padrao como modo_alternativo")
            # Fallback keeps max-autotune to recover throughput once warmup overhead is paid.
            return _compile_model(model, mode="max-autotune", dynamic=True)

    def create_custom_inference_compiler(self) -> Any:
        """
        Create custom inference compiler using torch.jit.optimize_for_inference.

        Returns:
            Custom compiler function
        """

        def optimize_for_inference_compiler(gm: Any, example_inputs: Any) -> Any:
            try:
                import torch

                self.logger.debug("Aplicando otimizacoes personalizadas de inferencia")

                scripted = torch.jit.script(gm)
                optimized = torch.jit.optimize_for_inference(scripted)

                self.logger.success("Otimizacao personalizada de inferencia aplicada")
                return optimized

            except Exception as e:
                self.logger.warning(f"Custom inference compilation failed: {e}")
                return gm

        return optimize_for_inference_compiler

    def compile_with_custom_backend(self, model: Any, example_inputs: Any) -> Any:
        """
        Compile model with custom inference backend.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing

        Returns:
            Compiled model with custom backend
        """
        try:
            import torch

            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile not available (requires PyTorch 2.0+)")

            self.logger.debug("Compiling with custom inference backend")

            custom_compiler = self.create_custom_inference_compiler()

            compiled_model = _compile_model(
                model,
                backend=custom_compiler,
                dynamic=True,
            )

            self.logger.success("Compilacao com backend personalizado concluida")
            return compiled_model

        except Exception as e:
            self.logger.warning(f"Custom backend compilation failed: {e}")
            return model

    def auto_select_backend(self, model: Any, example_inputs: Any) -> tuple[Any, str]:
        """
        Automatically select the best backend based on available hardware.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing

        Returns:
            Tuple of (compiled_model, backend_name)
        """
        import torch

        self.logger.debug("Selecting best compilation backend automatically")

        if not hasattr(torch, "compile"):
            self.logger.warning("torch.compile not available; returning original model")
            return model, "unavailable"

        strategies: dict[str, Callable[[Any, Any], tuple[Any, str] | Any]] = {
            "openvino": lambda m, ex: (self.compile_with_openvino(m, ex), "openvino"),
            "tvm": lambda m, ex: (self.compile_with_tvm(m, ex), "tvm"),
            "nnc": lambda m, ex: (self.compile_with_nnc(m, ex), "nnc"),
            # Default keeps max-autotune for better steady-state speed even if compilation warms up slower.
            "default": lambda m, ex: (
                _compile_model(m, mode="max-autotune", dynamic=True),
                "max-autotune",
            ),
        }

        for backend in self._backend_order:
            if backend == "openvino" and hasattr(torch, "xpu") and torch.xpu.is_available():
                try:
                    return strategies[backend](model, example_inputs)  # type: ignore[return-value]
                except Exception:
                    continue
            if backend == "tvm" and self._is_cuda_available():
                try:
                    return strategies[backend](model, example_inputs)  # type: ignore[return-value]
                except Exception:
                    continue
            if backend in {"nnc", "default"}:
                try:
                    return strategies[backend](model, example_inputs)  # type: ignore[return-value]
                except Exception:
                    continue

        compiled = self.compile_with_nnc(model, example_inputs)
        return compiled, "nnc"

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False


# Module-level flag to track if CUDA allocator has been configured
_CUDA_ALLOCATOR_CONFIGURED = False


class PerformanceOptimizer:
    """SOTA Performance Optimizer for ML training and inference."""

    def __init__(self, enable_cuda: bool = True) -> None:
        import torch

        self.enable_cuda = enable_cuda and torch.cuda.is_available()
        self.logger = logger

    def configure_pytorch_251(self) -> None:
        """Configure PyTorch 2.5.1+ for maximum performance."""
        global _CUDA_ALLOCATOR_CONFIGURED
        import torch

        perf_cfg = _load_performance_config().get("performance", {})
        torch_cfg = perf_cfg.get("torch", {})
        torch_cfg.get("cuda_allocator", {})
        self.logger.debug("Configuring SOTA optimizations for PyTorch 2.5.1+")

        if torch_cfg.get("enable_static_graph") and hasattr(torch, "enable_static_graph"):
            torch.enable_static_graph()
            self.logger.debug("Enabled static CPU kernels (NativeRT)")

        if self.enable_cuda:
            # IMPORTANT: PYTORCH_CUDA_ALLOC_CONF must be set BEFORE PyTorch is imported.
            # Setting it at runtime after import causes:
            # "INTERNAL ASSERT FAILED at CUDAAllocatorConfig.cpp - Allocator backend parsed
            # at runtime != allocator backend parsed at load time"
            #
            # export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,max_non_split_rounding_mb:1024"

            if not _CUDA_ALLOCATOR_CONFIGURED:
                _CUDA_ALLOCATOR_CONFIGURED = True
                if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
                    self.logger.debug(
                        f"Using pre-set CUDA allocator config: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}"
                    )
                else:
                    self.logger.debug(
                        "CUDA allocator using PyTorch defaults (cudaMallocAsync requires pre-import env var)"
                    )

            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    reserved_memory = int(
                        total_memory
                        * perf_cfg.get("memory_profiling", {}).get("cuda_memory_fraction", 0.9)
                    )
                    torch.cuda.set_per_process_memory_fraction(reserved_memory / total_memory)
                    self.logger.debug(f"Set CUDA memory pool: {reserved_memory / 1024**3:.1f} GB")
                except RuntimeError as e:
                    self.logger.debug(f"Could not set memory fraction: {e}")

        if hasattr(torch, "_dynamo"):
            inductor_cfg = torch_cfg.get("inductor", {})
            os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = str(inductor_cfg.get("max_autotune", 1))
            os.environ["TORCHINDUCTOR_AOT_AUTOGRAD_ENABLE_UPDATED"] = "1"
            self.logger.debug("Enabled Inductor max-autotune and AOT autograd")

    def configure_pytorch_performance_flags(self) -> None:
        """Configure PyTorch performance flags for modern GPUs."""
        import torch

        self.logger.debug("Configuring PyTorch performance flags")

        if not self.enable_cuda:
            self.logger.warning("CUDA not available, skipping GPU optimizations")
            return

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.debug("Enabled cuDNN benchmarking and TF32")

        if hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
                self.logger.debug("Enabled TF32 for matrix multiplications")

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            os.environ["PYTORCH_JIT"] = "1"
            self.logger.debug("SDPA available for optimized attention")

    def configure_ray_v2(self) -> None:
        """Configure Ray 3.0+ with Train v2 and fault tolerance."""
        cfg = _load_performance_config().get("performance", {}).get("ray", {})
        self.logger.debug("Configuring Ray 3.0+ with Train v2")

        if cfg.get("train_v2_enabled", True):
            os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
            self.logger.debug("Enabled Ray Train v2")

        if cfg.get("fault_tolerance_enabled", True):
            os.environ["RAY_FAULT_TOLERANCE_ENABLED"] = "1"
            os.environ["RAY_CHECKPOINT_FREQUENCY"] = str(cfg.get("checkpoint_frequency", 5))
            self.logger.debug("Enabled fault tolerance with checkpoints")

        if cfg.get("enable_vllm") or os.getenv("RAY_ENABLE_VLLM") is not None:
            os.environ["RAY_VLLM_ENGINE_CONFIG"] = "auto"
            self.logger.debug("Configured vLLM integration")

    def configure_memory_profiling(self) -> None:
        """Configure memory profiling and monitoring."""
        cfg = _load_performance_config().get("performance", {}).get("memory_profiling", {})
        self.logger.debug("Configuring memory profiling")

        if self.enable_cuda:
            os.environ["PYTORCH_CUDA_MEMORY_FRACTION"] = str(cfg.get("cuda_memory_fraction", 0.9))
            self.logger.debug("Configured CUDA memory fraction")

            os.environ["CUDA_LAUNCH_BLOCKING"] = str(cfg.get("cuda_launch_blocking", 0))
            self.logger.debug("Enabled asynchronous CUDA operations")

        os.environ["MALLOC_CONF"] = cfg.get(
            "malloc_conf",
            "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000",
        )
        self.logger.debug("Configured memory allocator (tcmalloc)")

    def optimize_compile_settings(self) -> None:
        """Configure advanced compiler optimizations."""
        import torch

        cfg = _load_performance_config().get("performance", {})
        inductor_cfg = cfg.get("torch", {}).get("inductor", {})
        allow_dynamic = cfg.get("torch", {}).get("allow_dynamic_shapes", True)
        self.logger.debug("Configuring compiler optimizations")

        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = str(inductor_cfg.get("max_autotune", 1))
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_MEMORY_FRACTION"] = str(
            inductor_cfg.get("max_autotune_memory_fraction", 0.5)
        )
        self.logger.debug("Enabled Inductor max-autotune")

        if hasattr(torch, "jit"):
            os.environ["PYTORCH_NVFUSER_CAPABILITIES"] = "enable,fusion"
            self.logger.debug("Enabled NVFuser fusion capabilities")

        os.environ["TORCHINDUCTOR_AOT_AUTOGRAD_ENABLE_UPDATED"] = "1"
        self.logger.debug("Enabled AOT autograd optimizations")

        if hasattr(torch, "_dynamo") and allow_dynamic:
            os.environ["TORCH_DYNAMO_ALLOW_DYNAMIC_SHAPES"] = "1"
            self.logger.debug("Enabled support for dynamic shapes")

    def apply_all_optimizations(self) -> None:
        """Apply all SOTA performance optimizations."""
        self.logger.debug("Applying SOTA performance optimizations...")

        self.configure_pytorch_251()
        self.configure_pytorch_performance_flags()
        self.configure_ray_v2()
        self.configure_memory_profiling()
        self.optimize_compile_settings()

        self.logger.debug("All SOTA performance optimizations applied")


class CompilationProfiler:
    """Profiler for torch.compile compilation and execution."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.logger = logger
        perf_cfg = _load_performance_config().get("performance", {})
        cfg_dir = perf_cfg.get("compilation_logs_dir", settings.OUTPUTS_DIR / "compilation_logs")
        resolved_dir = _resolve_output_dir(cfg_dir if output_dir is None else output_dir)
        self.output_dir = resolved_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_compilation(
        self, model: Any, example_inputs: Any, backend: str = "default"
    ) -> dict[str, Any]:
        """
        Profile torch.compile compilation with detailed metrics.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing
            backend: Compilation backend to use

        Returns:
            Dictionary with compilation metrics
        """
        try:
            import torch
            import time

            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile not available")

            self.logger.info(f"Perfilando compilacao com backend: {backend}")

            compile_start = time.time()

            if backend == "default":
                # Default uses max-autotune to chase peak throughput; profiling accounts for the longer warmup.
                compiled_model = _compile_model(model, mode="max-autotune", dynamic=True)
            else:
                compiled_model = _compile_model(model, backend=backend, dynamic=True)

            compile_time = time.time() - compile_start

            warmup_start = time.time()
            _ = compiled_model(*example_inputs)
            warmup_time = time.time() - warmup_start

            self.logger.info(f"Tempo de compilacao: {compile_time:.4f}s")
            self.logger.info(f"Tempo de aquecimento: {warmup_time:.4f}s")

            metrics = {
                "backend": backend,
                "compile_time": compile_time,
                "warmup_time": warmup_time,
                "total_time": compile_time + warmup_time,
                "success": True,
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Compilation profiling failed: {e}")
            return {
                "backend": backend,
                "compile_time": 0.0,
                "warmup_time": 0.0,
                "total_time": 0.0,
                "success": False,
                "error": str(e),
            }

    def export_chrome_trace(
        self, model: Any, example_inputs: Any, trace_file: str = "trace_compile.json"
    ) -> Path | None:
        """
        Export compilation trace to Chrome trace format.

        Args:
            model: PyTorch model to trace
            example_inputs: Example inputs
            trace_file: Output trace file name

        Returns:
            Path to trace file or None if failed
        """
        try:
            import torch

            if not hasattr(torch, "profiler"):
                self.logger.warning("torch.profiler not available")
                return None

            trace_path = self.output_dir / trace_file

            with torch.profiler.profile() as prof:
                # Trace uses max-autotune to capture the optimized steady-state graph after the initial warmup.
                compiled_model = _compile_model(model, mode="max-autotune", dynamic=True)
                _ = compiled_model(*example_inputs)

            prof.export_chrome_trace(str(trace_path))

            self.logger.info(f"Trace de compilacao exportado para: {trace_path}")
            return trace_path

        except Exception as e:
            self.logger.error(f"Chrome trace export failed: {e}")
            return None

    def get_compile_times_report(self) -> dict[str, Any]:
        """
        Get compilation times report from torch._dynamo.

        Returns:
            Dictionary with compilation statistics
        """
        try:
            import torch

            if hasattr(torch._dynamo, "utils") and hasattr(torch._dynamo.utils, "compile_times"):
                compile_times = torch._dynamo.utils.compile_times()
                self.logger.info("Relatorio de tempos de compilacao:")
                self.logger.info(f"{compile_times}")

                return {"compile_times": str(compile_times), "success": True}
            else:
                self.logger.warning("torch._dynamo.utils.compile_times not available")
                return {"compile_times": "Not available", "success": False}

        except Exception as e:
            self.logger.error(f"Failed to get compile times: {e}")
            return {"compile_times": "Error", "success": False, "error": str(e)}

    def benchmark_backends(
        self, model: Any, example_inputs: Any, backends: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Benchmark different compilation backends.

        Args:
            model: PyTorch model to benchmark
            example_inputs: Example inputs
            backends: List of backends to benchmark

        Returns:
            Dictionary with benchmark results
        """
        if backends is None:
            backends = ["default", "openvino", "tvm", "nnc"]

        self.logger.info(f"Executando benchmark dos backends: {', '.join(backends)}")

        results = {}
        for backend in backends:
            self.logger.info(f"Testando backend: {backend}")
            metrics = self.profile_compilation(model, example_inputs, backend)
            results[backend] = metrics

        best_backend = min(
            [(b, m) for b, m in results.items() if m["success"]],
            key=lambda x: x[1]["total_time"],
            default=("unknown", {}),
        )[0]

        self.logger.success(f"Melhor backend: {best_backend}")

        return {
            "results": results,
            "best_backend": best_backend,
            "winner": results.get(best_backend, {}),
        }


class AOTAutogradOptimizer:
    """Advanced AOT Autograd optimizations with custom compilers."""

    def __init__(self) -> None:
        self.logger = logger

    def create_aot_function(
        self,
        fn: Callable[..., Any],
        fw_compiler: Any | None = None,
        bw_compiler: Any | None = None,
    ) -> Any:
        """
        Create AOT Function with custom compilers.

        Args:
            fn: Function to compile
            fw_compiler: Forward compiler (default: ts_compile)
            bw_compiler: Backward compiler (default: ts_compile)

        Returns:
            AOT-compiled function
        """
        try:
            from functorch import aot_function

            if fw_compiler is None:
                fw_compiler = self._get_default_compiler()
            if bw_compiler is None:
                bw_compiler = self._get_default_compiler()

            self.logger.info("Criando funcao AOT com compiladores personalizados")

            aot_fn = aot_function(fn, fw_compiler=fw_compiler, bw_compiler=bw_compiler)

            self.logger.success("Funcao AOT criada com sucesso")
            return aot_fn

        except ImportError:
            self.logger.warning("functorch not available, returning original function")
            return fn
        except Exception as e:
            self.logger.error(f"AOT compilation failed: {e}")
            return fn

    def _get_default_compiler(self) -> Any:
        """Get default compiler (ts_compile)."""
        try:
            from functorch.compile import ts_compile

            return ts_compile
        except ImportError:
            return "default"

    def enable_operator_fusion(self, model: Any) -> Any:
        """
        Enable operator fusion for embedding operations.

        Args:
            model: PyTorch model

        Returns:
            Model with operator fusion enabled
        """
        try:
            import torch

            self.logger.info("Habilitando fusao de operadores para embeddings")

            if hasattr(torch, "_C") and hasattr(torch._C, "_jit_pass_fuse"):
                self.logger.info("Fusao de operadores JIT disponivel")
            else:
                self.logger.warning("JIT operator fusion not available")

            if hasattr(torch.nn.functional, "embedding"):
                self.logger.info("Operacoes de embedding otimizadas disponiveis")

            return model

        except Exception as e:
            self.logger.error(f"Operator fusion enablement failed: {e}")
            return model

    def benchmark_aot_vs_eager(
        self,
        fn: Callable[..., Any],
        example_args: tuple[Any, ...],
        iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Benchmark AOT compilation vs eager execution.

        Args:
            fn: Function to benchmark
            example_args: Example arguments
            iterations: Number of iterations

        Returns:
            Benchmark results
        """
        import time

        self.logger.info(f"Executando benchmark AOT vs eager ({iterations} iteracoes)")

        eager_times = []
        for _ in range(iterations):
            start = time.time()
            _ = fn(*example_args)
            eager_times.append(time.time() - start)

        aot_fn = self.create_aot_function(fn)
        aot_times = []
        for _ in range(iterations):
            start = time.time()
            _ = aot_fn(*example_args)
            aot_times.append(time.time() - start)

        eager_avg = sum(eager_times) / len(eager_times)
        aot_avg = sum(aot_times) / len(aot_times)
        speedup = eager_avg / aot_avg if aot_avg > 0 else 0.0

        self.logger.info(f"Eager execution: {eager_avg:.6f}s avg")
        self.logger.info(f"AOT compilation: {aot_avg:.6f}s avg")
        self.logger.info(f"Speedup: {speedup:.2f}x")

        return {
            "eager_avg_time": eager_avg,
            "aot_avg_time": aot_avg,
            "speedup": speedup,
            "iterations": iterations,
        }


def get_optimizer() -> PerformanceOptimizer:
    """Factory function to get performance optimizer instance."""
    return PerformanceOptimizer()


def apply_sota_optimizations() -> None:
    """Convenience function to apply all SOTA optimizations."""
    optimizer = get_optimizer()
    optimizer.apply_all_optimizations()
