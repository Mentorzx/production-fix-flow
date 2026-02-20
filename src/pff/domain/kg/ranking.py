"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/domain/kg/ranking.py

"""

import numpy as np

from pff.shared import logger

"""Parallel ranking worker utilities for the KGC pipeline."""


def create_test_data_chunks(
    test_triples: np.ndarray,
    chunk_size: int,
    num_workers: int | None = None,
    max_chunk_size: int = 1000,
) -> list[np.ndarray]:
    """Splits test triples into chunks for parallel processing."""
    effective_chunk_size = min(chunk_size, max_chunk_size)
    number_of_chunks = (len(test_triples) + effective_chunk_size - 1) // effective_chunk_size
    chunks = [
        test_triples[i * effective_chunk_size : (i + 1) * effective_chunk_size]
        for i in range(number_of_chunks)
    ]

    logger.info(
        f"Dados de teste divididos em {len(chunks)} chunks (tamanho: {effective_chunk_size})"
    )

    return chunks
