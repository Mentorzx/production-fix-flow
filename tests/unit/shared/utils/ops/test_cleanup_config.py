from pff import settings
from pff.infrastructure import cleanup
from pff.infrastructure.cleanup import config as cleanup_config
from pff.infrastructure.cleanup.commands.postgres import PostgreSQLBackupCommand
from pff.shared.core.file_manager import FileManager


def test_load_cleanup_config_defaults(monkeypatch, tmp_path):
    """Config loader returns defaults when file is missing."""
    monkeypatch.setattr(cleanup_config, "CONFIG_PATH", tmp_path / "missing.yaml")

    cfg = cleanup.load_cleanup_config()

    assert cfg["retention"]["execution_logs_days"] == 30
    assert cfg["backup"]["keep_last"] == 5
    assert cfg["backup"]["dir"] == "outputs/backups/postgres"
    assert cfg["performance"]["max_concurrent_io"] == 10


def test_load_cleanup_config_from_yaml(monkeypatch, tmp_path):
    """Config loader reads values from cleanup.yaml with nested section."""
    config_path = tmp_path / "cleanup.yaml"
    custom_cfg = {
        "cleanup": {
            "retention": {"execution_logs_days": 12},
            "backup": {"dir": "outputs/custom_backups", "keep_last": 2},
            "performance": {"max_concurrent_io": 3, "large_dir_threshold_bytes": 2048},
        }
    }
    FileManager.save(custom_cfg, config_path)
    monkeypatch.setattr(cleanup_config, "CONFIG_PATH", config_path)

    cfg = cleanup.load_cleanup_config()

    assert cfg["retention"]["execution_logs_days"] == 12
    assert cfg["backup"]["keep_last"] == 2
    assert cfg["backup"]["dir"] == "outputs/custom_backups"
    assert cfg["performance"]["large_dir_threshold_bytes"] == 2048


def test_retention_days_configurable(monkeypatch):
    """Database cleanup command uses configured retention days."""
    monkeypatch.setattr(
        cleanup_config,
        "CLEANUP_CONFIG",
        {"retention": {"execution_logs_days": 7}, "backup": {}, "performance": {}},
    )

    cmd = cleanup.DatabaseCleanCommand()

    assert cmd._retention_days == 7


def test_keep_backups_configurable(monkeypatch):
    """PostgreSQL backup command respects cleanup backup config."""
    import pff.infrastructure.cleanup.commands.postgres as pg_cleanup

    monkeypatch.setattr(
        pg_cleanup,
        "CLEANUP_CONFIG",
        {"backup": {"dir": "outputs/db/backups", "keep_last": 2}},
    )

    command = PostgreSQLBackupCommand(tables=["kg_rules"])

    assert command.keep_backups == 2
    assert command.backup_dir == settings.ROOT_DIR / "outputs/db/backups"
