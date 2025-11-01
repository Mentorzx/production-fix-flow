"""create_training_metrics_table

Revision ID: 0ca8ec697d0c
Revises: 25185587c991
Create Date: 2025-10-30 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0ca8ec697d0c'
down_revision: Union[str, Sequence[str], None] = '25185587c991'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Create training_metrics table."""
    op.execute("""
        CREATE TABLE training_metrics (
            id BIGSERIAL PRIMARY KEY,
            execution_log_id BIGINT,
            model_name VARCHAR(100) NOT NULL,
            epoch INTEGER,
            metric_name VARCHAR(50) NOT NULL,
            metric_value DOUBLE PRECISION NOT NULL,
            split VARCHAR(20),
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_execution_log
                FOREIGN KEY (execution_log_id)
                REFERENCES execution_logs (id)
                ON DELETE CASCADE
        )
    """)

    op.execute("CREATE INDEX idx_training_metrics_execution ON training_metrics (execution_log_id)")
    op.execute("CREATE INDEX idx_training_metrics_model ON training_metrics (model_name, metric_name)")
    op.execute("CREATE INDEX idx_training_metrics_epoch ON training_metrics (model_name, epoch)")
    op.execute("CREATE INDEX idx_training_metrics_metadata ON training_metrics USING gin (metadata)")

    op.execute("COMMENT ON TABLE training_metrics IS 'Métricas de treinamento por época'")
    op.execute("COMMENT ON COLUMN training_metrics.execution_log_id IS 'FK para execution_logs (rastreamento)'")
    op.execute("COMMENT ON COLUMN training_metrics.model_name IS 'Nome do modelo: transe, lightgbm, ensemble'")
    op.execute("COMMENT ON COLUMN training_metrics.epoch IS 'Número da época (NULL para métricas finais)'")
    op.execute("COMMENT ON COLUMN training_metrics.metric_name IS 'Nome da métrica: loss, mrr, hits@1, hits@10, auc'")
    op.execute("COMMENT ON COLUMN training_metrics.metric_value IS 'Valor da métrica (float)'")
    op.execute("COMMENT ON COLUMN training_metrics.split IS 'Split: train, valid, test'")
    op.execute("COMMENT ON COLUMN training_metrics.metadata IS 'Metadados adicionais (JSONB)'")


def downgrade() -> None:
    """Downgrade schema: Drop training_metrics table."""
    op.execute("""
        DROP TABLE IF EXISTS training_metrics CASCADE;
    """)
