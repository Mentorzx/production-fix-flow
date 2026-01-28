import torch
import time
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    def save_checkpoint(self, data, filename):
        pass

    def load_checkpoint(self, filename, map_location=None):
        return None


def run_trial():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        return

    device = torch.device("cuda")
    m_cfg = DSLFMKGCConfig(num_entities=1000, num_relations=100)
    t_cfg = KGCTrainingConfig(epochs=1, batch_size=1024)
    manager = DSLFMKGCManager(m_cfg, t_cfg, persistence_port=MockPersistencePort(), device=device)
    model = manager.model
    optimizer = manager.optimizer

    # Static inputs for graph
    static_h = torch.randint(0, 1000, (1024,), device=device)
    static_r = torch.randint(0, 100, (1024,), device=device)
    static_t = torch.randint(0, 1000, (1024,), device=device)
    static_idx = torch.arange(1024, device=device)

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_loss(static_h, static_r, static_t, triple_indices=static_idx)
            losses["loss"].backward()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        losses = model.compute_loss(static_h, static_r, static_t, triple_indices=static_idx)
        losses["loss"].backward()
        # Note: optimizer.step() usually works in graphs if it doesn't have CPU logic
        optimizer.step()

    # Benchmark Graph
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    print(f"TRAIN_ITER_GRAPH_MS: {(time.perf_counter() - start) * 20:.2f}")


if __name__ == "__main__":
    run_trial()
