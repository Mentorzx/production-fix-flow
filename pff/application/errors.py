"""Application-layer errors with contextual information."""


class LearnUseCaseError(RuntimeError):
    """Base error for learn use case failures."""


class StrategyResolutionError(LearnUseCaseError):
    """Raised when a training strategy cannot be resolved."""


class PreprocessedDataMissingError(LearnUseCaseError):
    """Raised when required preprocessed artifacts are missing."""
