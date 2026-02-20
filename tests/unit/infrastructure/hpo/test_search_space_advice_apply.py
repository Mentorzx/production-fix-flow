from pff.infrastructure.hpo.dashboard.server import _apply_search_space_patch_to_config


def test_apply_search_space_patch_updates_config() -> None:
    config = {
        "dslfm_kgc": {
            "training": {},
            "architecture": {},
            "logic": {},
            "contrastive": {},
            "sampling": {},
            "pc": {},
        },
        "adaptive_range_factors": {},
    }
    patch = {
        "learning_rate": {"low": 1.0e-4, "high": 2.0e-4},
        "embedding_dim": {"type": "categorical", "choices": [128, 256]},
        "min_delta": {"low": 0.0002, "high": 0.0004},
    }

    updated, applied, skipped = _apply_search_space_patch_to_config(config, patch)

    assert updated["dslfm_kgc"]["training"]["lr_low"] == 1.0e-4
    assert updated["dslfm_kgc"]["training"]["lr_high"] == 2.0e-4
    assert updated["dslfm_kgc"]["architecture"]["feature_dim_choices"] == [128, 256]
    assert updated["adaptive_range_factors"]["min_delta_low"] == 0.0002
    assert updated["adaptive_range_factors"]["min_delta_high"] == 0.0004
    assert set(applied) == {"learning_rate", "embedding_dim", "min_delta"}
    assert skipped == []


def test_apply_search_space_patch_skips_noop_updates() -> None:
    config = {
        "dslfm_kgc": {"training": {"lr_low": 1.0e-4, "lr_high": 2.0e-4}},
        "adaptive_range_factors": {"min_delta_low": 0.0002, "min_delta_high": 0.0004},
    }
    patch = {
        "learning_rate": {"low": 1.0e-4, "high": 2.0e-4, "type": "float"},
        "min_delta": {"low": 0.0002, "high": 0.0004, "type": "float"},
    }

    _, applied, skipped = _apply_search_space_patch_to_config(config, patch)

    assert applied == []
    assert set(skipped) == {"learning_rate", "min_delta"}


def test_apply_search_space_patch_casts_int_bounds() -> None:
    config = {"dslfm_kgc": {"pc": {"rebuild_every_low": 0, "rebuild_every_high": 50}}}
    patch = {"rebuild_every": {"low": 0.0, "high": 25.0, "type": "int"}}

    updated, applied, skipped = _apply_search_space_patch_to_config(config, patch)

    assert updated["dslfm_kgc"]["pc"]["rebuild_every_low"] == 0
    assert updated["dslfm_kgc"]["pc"]["rebuild_every_high"] == 25
    assert applied == ["rebuild_every"]
    assert skipped == []


def test_apply_search_space_patch_skips_extreme_expansion() -> None:
    config = {"dslfm_kgc": {"training": {"lr_low": 1.0e-4, "lr_high": 2.0e-4}}}
    patch = {"learning_rate": {"low": 1.0e-4, "high": 20.0, "type": "float"}}

    updated, applied, skipped = _apply_search_space_patch_to_config(config, patch)

    assert updated["dslfm_kgc"]["training"]["lr_low"] == 1.0e-4
    assert updated["dslfm_kgc"]["training"]["lr_high"] == 2.0e-4
    assert applied == []
    assert skipped == ["learning_rate"]


def test_apply_search_space_patch_skips_negative_low_when_current_is_non_negative() -> None:
    config = {"dslfm_kgc": {"logic": {"lambda_logic_low": 0.0, "lambda_logic_high": 0.05}}}
    patch = {"lambda_logic": {"low": -0.1, "high": 0.08, "type": "float"}}

    updated, applied, skipped = _apply_search_space_patch_to_config(config, patch)

    assert updated["dslfm_kgc"]["logic"]["lambda_logic_low"] == 0.0
    assert updated["dslfm_kgc"]["logic"]["lambda_logic_high"] == 0.05
    assert applied == []
    assert skipped == ["lambda_logic"]
