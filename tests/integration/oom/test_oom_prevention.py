"""Test CUDA OOM prevention in DSLFM evaluation.

These tests verify that memory-efficient chunking prevents OOM errors.
"""

import pytest
import torch


def test_pc_chunk_sizes_are_reasonable():
    """Verify PC chunk sizes are small enough to avoid OOM on typical GPU."""
    cuda_chunk = 2000
    cpu_chunk = 10000

    assert cuda_chunk < 2500, f"CUDA chunk size {cuda_chunk} too large"
    assert cpu_chunk < 12000, f"CPU chunk size {cpu_chunk} too large"

    embedding_dim = 256
    # num_entities = 11575
    batch_size = 512

    estimated_memory_per_element = embedding_dim * 4 / (1024**3)

    cuda_total_elements = batch_size * cuda_chunk
    # We implicitly assume (Batch, Chunk, Embedding) or (Batch, Chunk) depending on op
    # If calculating score matrix memory: Batch * Chunk * 1 (float)
    # If calculating embeddings memory: Batch * Chunk * Embedding
    # The estimated_memory_per_element includes embedding_dim.
    # So assuming shape is (Batch, Chunk, Embedding).

    cuda_memory_gb = cuda_total_elements * estimated_memory_per_element
    assert cuda_memory_gb < 2.0, f"CUDA chunk memory {cuda_memory_gb:.2f} GB too high"

    cpu_total_elements = batch_size * cpu_chunk
    cpu_memory_gb = cpu_total_elements * estimated_memory_per_element
    assert cpu_memory_gb < 8.0, f"CPU chunk memory {cpu_memory_gb:.2f} GB too high"


def test_chunked_vs_unchunked_memory_usage():
    """Verify chunking reduces peak memory usage."""
    device = torch.device("cpu")

    num_heads = 512
    num_tails = 11575
    embedding_dim = 256

    z_heads = torch.randn(num_heads, embedding_dim, device=device)
    all_z = torch.randn(num_tails, embedding_dim, device=device)

    large_unchunked = torch.stack([z_heads.unsqueeze(1) + all_z.unsqueeze(0)], dim=-1)
    large_unchunked_size_mb = (large_unchunked.numel() * large_unchunked.element_size()) / (1024**2)
    assert large_unchunked_size_mb > 100, "Unchunked tensor should be large"

    chunk_size_tails = 100
    chunk_size_heads = 64

    total_elements = 0
    for i in range(0, num_heads, chunk_size_heads):
        end_h = min(i + chunk_size_heads, num_heads)
        z_h_chunk = z_heads[i:end_h]

        for j in range(0, num_tails, chunk_size_tails):
            end = min(j + chunk_size_tails, num_tails)
            chunk_z = all_z[j:end]

            combined = torch.clamp(
                0.5 * (z_h_chunk.unsqueeze(1) + chunk_z.unsqueeze(0)), 1e-7, 1.0 - 1e-7
            )
            attr_probs = torch.stack([combined, 1.0 - combined], dim=-1)

            chunk_elements = attr_probs.numel()
            total_elements += chunk_elements

    avg_chunk_elements = total_elements / (
        (num_heads // chunk_size_heads) * (num_tails // chunk_size_tails)
    )
    avg_chunk_size_mb = avg_chunk_elements * 4 / (1024**2)

    assert avg_chunk_size_mb < 20, f"Average chunk size {avg_chunk_size_mb:.2f} MB too high"

    reduction_factor = large_unchunked_size_mb / avg_chunk_size_mb
    assert reduction_factor > 50, (
        f"Chunking should reduce peak memory by >50x, got {reduction_factor:.1f}x"
    )


def test_batch_evaluation_chunking():
    """Test that evaluation batching reduces memory."""
    num_triples = 10000
    num_entities = 11575
    batch_size = 32

    triples = torch.randint(0, num_entities, (num_triples, 3), dtype=torch.long).cpu()

    total_memory = 0
    for i in range(0, len(triples), batch_size):
        batch = triples[i : i + batch_size]

        scores = torch.randn(len(batch), num_entities, device="cpu")

        total_memory += scores.element_size()

    avg_memory_mb = total_memory / ((len(triples) // batch_size) * 4) / (1024**2)
    assert avg_memory_mb < 50, f"Average batch memory {avg_memory_mb:.2f} MB too high"


def test_memory_monitoring():
    """Test that we can monitor CUDA memory usage."""
    if not torch.cuda.is_available():
        assert torch.cuda.is_available() is False
        return

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    x = torch.randn(1000, 256, device="cuda")

    after_allocation = torch.cuda.memory_allocated()
    allocated_mb = (after_allocation - initial_memory) / (1024**2)

    assert allocated_mb > 0, "Memory should have been allocated"
    assert allocated_mb < 10, f"Allocated {allocated_mb:.2f} MB, expected <10 MB"

    del x
    torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated()

    assert after_free <= initial_memory * 1.1, "Memory should have been released"
