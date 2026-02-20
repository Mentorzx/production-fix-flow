from pff.domain.hpo.search_space import SearchSpaceFactory


def test_adaptive_training_space_respects_range_overrides() -> None:
    bounds = SearchSpaceFactory.create_adaptive_training_space(
        num_train_triples=1000,
        num_valid_triples=200,
        num_entities=500,
        num_relations=10,
        range_factors={
            "early_stopping_patience_low": 7,
            "early_stopping_patience_high": 9,
            "validate_every_low": 4,
            "validate_every_high": 6,
            "min_delta_low": 0.0002,
            "min_delta_high": 0.0004,
        },
    )

    assert bounds["early_stopping_patience"] == (7.0, 9.0)
    assert bounds["validate_every"] == (4.0, 6.0)
    assert bounds["min_delta"] == (0.0002, 0.0004)
