"""
Repository Pattern for PostgreSQL Data Access.

Design Patterns:
- Repository Pattern: Abstracts database access
- Unit of Work: Transaction management via connection pool
- Data Mapper: Maps between domain objects and database records
"""

from pff.db.repositories.kg_splits import KGSplitsRepository
from pff.db.repositories.kg_mappings import KGMappingsRepository
from pff.db.repositories.kg_rules import KGRulesRepository
from pff.db.repositories.embeddings import EmbeddingsRepository
from pff.db.repositories.ml_models import MLModelsRepository
from pff.db.repositories.execution_logs import ExecutionLogsRepository
from pff.db.repositories.training_metrics import TrainingMetricsRepository
from pff.db.repositories.pipeline_checkpoints import PipelineCheckpointsRepository

__all__ = [
    "KGSplitsRepository",
    "KGMappingsRepository",
    "KGRulesRepository",
    "EmbeddingsRepository",
    "MLModelsRepository",
    "ExecutionLogsRepository",
    "TrainingMetricsRepository",
    "PipelineCheckpointsRepository",
]
