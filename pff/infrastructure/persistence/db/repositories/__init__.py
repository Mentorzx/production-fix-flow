"""
Repository Pattern for PostgreSQL Data Access.

Design Patterns:
- Repository Pattern: Abstracts database access
- Unit of Work: Transaction management via connection pool
- Data Mapper: Maps between domain objects and database records
"""

from pff.infrastructure.persistence.db.repositories.audit_analysis import (
    AuditAnalysisRepository,
)
from pff.infrastructure.persistence.db.repositories.audit_artifacts import (
    AuditArtifactsRepository,
)
from pff.infrastructure.persistence.db.repositories.audit_reports import (
    AuditReportsRepository,
)
from pff.infrastructure.persistence.db.repositories.audit_semantics import (
    AuditSemanticsRepository,
)
from pff.infrastructure.persistence.db.repositories.embeddings import (
    EmbeddingsRepository,
)
from pff.infrastructure.persistence.db.repositories.execution_logs import (
    ExecutionLogsRepository,
)
from pff.infrastructure.persistence.db.repositories.kg_mappings import (
    KGMappingsRepository,
)
from pff.infrastructure.persistence.db.repositories.kg_rules import KGRulesRepository
from pff.infrastructure.persistence.db.repositories.kg_splits import KGSplitsRepository
from pff.infrastructure.persistence.db.repositories.ml_models import MLModelsRepository
from pff.infrastructure.persistence.db.repositories.pipeline_checkpoints import (
    PipelineCheckpointsRepository,
)
from pff.infrastructure.persistence.db.repositories.training_metrics import (
    TrainingMetricsRepository,
)

__all__ = [
    "KGSplitsRepository",
    "KGMappingsRepository",
    "KGRulesRepository",
    "EmbeddingsRepository",
    "MLModelsRepository",
    "ExecutionLogsRepository",
    "TrainingMetricsRepository",
    "PipelineCheckpointsRepository",
    "AuditArtifactsRepository",
    "AuditAnalysisRepository",
    "AuditSemanticsRepository",
    "AuditReportsRepository",
]
