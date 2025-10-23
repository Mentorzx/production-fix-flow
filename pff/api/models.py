from enum import Enum
from pydantic import BaseModel, Field

class ExecutionStatus(str, Enum):
    queued   = "queued"
    running  = "running"
    done     = "done"
    error    = "error"

class ExecutionRequest(BaseModel):
    msisdns: list[str] = Field(..., examples=["5511999999999", "5511999998888"])
    labels:  list[str] = Field(..., examples=["Cancelar", "Swap"])

class ExecutionResponse(BaseModel):
    execution_id: str
    status: ExecutionStatus

class SequenceInfo(BaseModel):
    name: str
    steps: int
    description: str | None = None
