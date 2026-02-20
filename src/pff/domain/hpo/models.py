"""Domain constants and helpers for HPO model naming."""

KGE_MODEL_DSLFM = "dslfm"

KGE_MODEL_ALIASES: dict[str, str] = {
    "dslfm": KGE_MODEL_DSLFM,
    "dslfm-kgc": KGE_MODEL_DSLFM,
    "dslfm_kgc": KGE_MODEL_DSLFM,
}


def resolve_kge_model(model_name: str) -> str:
    """Resolve KGE model alias to canonical name."""
    normalized = model_name.lower().replace("_", "-")
    resolved = KGE_MODEL_ALIASES.get(normalized)
    if resolved is None:
        valid_options = list(KGE_MODEL_ALIASES.keys())
        raise ValueError(f"Unknown KGE model '{model_name}'. Valid options: {valid_options}")
    return resolved
