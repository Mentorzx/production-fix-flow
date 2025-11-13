"""Add pgvector indexes and JSONB optimizations."""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20241106_01_add_pgvector_indexes"
down_revision: Union[str, None] = "473699ca0a14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kg_embeddings_vector
            ON kg_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_logs_metadata_gin
            ON execution_logs USING gin ((metadata -> 'tags'));
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_training_metrics_model_epoch
            ON training_metrics (model_name, epoch)
            WHERE metadata IS NOT NULL;
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS kg_splits_summary AS
        SELECT
            split_name,
            split_type,
            COUNT(*) AS triples,
            COUNT(DISTINCT sample_id) AS unique_samples,
            MIN(created_at) AS first_seen,
            MAX(created_at) AS last_seen
        FROM kg_splits
        GROUP BY split_name, split_type;
        """
    )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS kg_splits_summary")
    op.execute("DROP INDEX IF EXISTS idx_training_metrics_model_epoch")
    op.execute("DROP INDEX IF EXISTS idx_execution_logs_metadata_gin")
    op.execute("DROP INDEX IF EXISTS idx_kg_embeddings_vector")
