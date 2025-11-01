"""create_ml_models_table

Revision ID: 25185587c991
Revises: 0ba16afe7bdb
Create Date: 2025-10-30 15:36:06.600938

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '25185587c991'
down_revision: Union[str, Sequence[str], None] = '0ba16afe7bdb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Create ml_models table."""
    op.execute("""
        CREATE TABLE ml_models (
            id BIGSERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            model_data BYTEA NOT NULL,
            model_version VARCHAR(50),
            metrics JSONB,
            hyperparameters JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_model_version UNIQUE (model_name, model_type, model_version)
        )
    """)

    op.execute("CREATE INDEX idx_ml_models_lookup ON ml_models (model_name, model_type)")
    op.execute("CREATE INDEX idx_ml_models_metrics ON ml_models USING gin (metrics)")

    op.execute("COMMENT ON TABLE ml_models IS 'Armazenamento de modelos ML com versioning'")
    op.execute("COMMENT ON COLUMN ml_models.model_name IS 'Nome do modelo: transe, lightgbm, ensemble'")
    op.execute("COMMENT ON COLUMN ml_models.model_type IS 'Tipo: checkpoint, binary, metadata'")
    op.execute("COMMENT ON COLUMN ml_models.model_data IS 'Dados binários comprimidos (gzip)'")
    op.execute("COMMENT ON COLUMN ml_models.model_version IS 'Versão do modelo (timestamp ou tag)'")
    op.execute("COMMENT ON COLUMN ml_models.metrics IS 'Métricas: MRR, Hits@1, Hits@10, AUC, etc.'")
    op.execute("COMMENT ON COLUMN ml_models.hyperparameters IS 'Hiperparâmetros usados no treinamento'")


def downgrade() -> None:
    """Downgrade schema: Drop ml_models table."""
    op.execute("""
        DROP TABLE IF EXISTS ml_models CASCADE;
    """)
