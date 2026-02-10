import re

SECRET_PATTERNS = [
    re.compile(
        r'(?i)(password|secret|token|key|auth|api_key|access_token)["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?'
    ),
    re.compile(r'(?i)(msisdn|phone|mobile)["\']?\s*[:=]\s*["\']?(\d+)["\']?'),
]
MASK = "***"
_MAX_MASK_LENGTH = 10_000


def mask_secrets(text: str) -> str:
    """Scrub sensitive data from text."""
    if not text:
        return text

    if len(text) > _MAX_MASK_LENGTH:
        return text

    masked = text
    for pattern in SECRET_PATTERNS:
        masked = pattern.sub(
            lambda m: f"{m.group(1)}{m.group(0)[len(m.group(1)) :].replace(m.group(2), MASK)}",
            masked,
        )
    return masked
