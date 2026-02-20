"""Search Space Advisor package.

Public API remains stable at:
`pff.infrastructure.hpo.search_space_advisor`
"""

from .dataset_profile import compute_dataset_profile_fingerprint
from .patching import generate_search_space_patch
from .service import (
    ADVISOR_VERSION,
    SearchSpaceAdvisor,
)

__all__ = [
    "ADVISOR_VERSION",
    "SearchSpaceAdvisor",
    "compute_dataset_profile_fingerprint",
    "generate_search_space_patch",
]
