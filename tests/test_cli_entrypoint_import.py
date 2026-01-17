def test_cli_entrypoint_module_import() -> None:
    import pff.cli

    assert hasattr(pff.cli, "cli_entrypoint")
