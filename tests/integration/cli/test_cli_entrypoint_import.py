def test_cli_entrypoint_module_import() -> None:
    import pff.drivers.cli.main

    assert hasattr(pff.drivers.cli.main, "cli_entrypoint")
