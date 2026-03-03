"""Architecture guardrails for application layer port usage."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_SERVICE_PATH = (
    REPO_ROOT / "src" / "pff" / "application" / "services" / "sequence_service.py"
)
LINE_SERVICE_BASE_PATH = (
    REPO_ROOT / "src" / "pff" / "application" / "services" / "line_service" / "base.py"
)
LEARN_USE_CASE_PATH = REPO_ROOT / "src" / "pff" / "application" / "learn_use_case.py"
INTELLIGENT_PREPROCESSOR_PATH = (
    REPO_ROOT / "src" / "pff" / "application" / "services" / "intelligent_preprocessor.py"
)
BUSINESS_SERVICE_CORE_PATH = (
    REPO_ROOT / "src" / "pff" / "application" / "services" / "business_service" / "core.py"
)
MODEL_INTEGRATION_PATH = (
    REPO_ROOT
    / "src"
    / "pff"
    / "application"
    / "services"
    / "business_service"
    / "model_integration.py"
)
RULE_ENGINE_PATH = (
    REPO_ROOT / "src" / "pff" / "application" / "services" / "business_service" / "rule_engine.py"
)
RULE_VALIDATOR_PATH = (
    REPO_ROOT
    / "src"
    / "pff"
    / "application"
    / "services"
    / "business_service"
    / "rule_validator.py"
)
ORCHESTRATOR_PATH = REPO_ROOT / "src" / "pff" / "drivers" / "orchestrator.py"
KG_PREPROCESS_PERSISTENCE_SERVICE_WRAPPER_PATH = (
    REPO_ROOT / "src" / "pff" / "application" / "services" / "kg_preprocess_persistence.py"
)


def test_sequence_service_must_use_file_manager_port_injection() -> None:
    """SequenceService must not instantiate concrete FileManager directly."""
    content = SEQUENCE_SERVICE_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.file_manager import FileManagerPort" in content
    assert "self._file_manager = file_manager" in content
    assert "FileManager()" not in content


def test_line_service_base_must_support_port_injection() -> None:
    """LineServiceBase must accept injected HTTP/File ports and use them first."""
    content = LINE_SERVICE_BASE_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.file_manager import FileManagerPort" in content
    assert "from pff.application.ports.http_client import HttpClientPort" in content
    assert "from pff.application.ports.line_api import LineApiPort" in content
    assert "from pff.application.ports.settings import SettingsPort" in content
    assert "http_client: HttpClientPort | None = None" in content
    assert "file_manager: FileManagerPort | None = None" in content
    assert "api_client: LineApiPort | None = None" in content
    assert "settings_obj: SettingsPort | None = None" in content
    assert "self._http_client = http_client or HttpClient(" in content
    assert "self._file_manager = file_manager or FileManager()" in content
    assert "from pff.shared.clients.http_client import API as LineApiDefault" not in content


def test_learn_use_case_must_forward_file_manager_port_to_strategies() -> None:
    """LearnUseCase must pass injected file manager into strategy construction."""
    content = LEARN_USE_CASE_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.file_manager import FileManagerPort" in content
    assert "from pff.application.ports.settings import SettingsPort" in content
    assert "file_manager: FileManagerPort | None = None" in content
    assert "settings_obj: SettingsPort | None = None" in content
    assert "self._file_manager = file_manager or FileManager()" in content
    assert "file_manager=self._file_manager" in content
    assert "settings_obj=self._settings" in content
    assert "from pff.application.kg_preprocess_persistence import" in content
    assert "from pff.application.services.kg_preprocess_persistence import" not in content


def test_application_must_not_keep_kg_preprocess_services_wrapper() -> None:
    """Legacy wrapper module must stay removed to avoid duplicate entrypoints."""
    assert not KG_PREPROCESS_PERSISTENCE_SERVICE_WRAPPER_PATH.exists()


def test_intelligent_preprocessor_must_accept_file_manager_port() -> None:
    """IntelligentPreprocessor must support injected file manager port."""
    content = INTELLIGENT_PREPROCESSOR_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.file_manager import FileManagerPort" in content
    assert "from pff.application.ports.settings import SettingsPort" in content
    assert "file_manager: FileManagerPort | None = None" in content
    assert "settings_obj: SettingsPort | None = None" in content
    assert "self.file_manager = file_manager or FileManager()" in content


def test_business_service_must_forward_file_manager_to_model_integration() -> None:
    """BusinessService should reuse its file_manager in default ModelIntegration."""
    content = BUSINESS_SERVICE_CORE_PATH.read_text(encoding="utf-8")

    assert "self.model_integration = model_integration or ModelIntegration(" in content
    assert "file_manager=self.file_manager" in content


def test_business_service_must_use_config_loader_port() -> None:
    """BusinessService must type config loader via application port and forward it."""
    content = BUSINESS_SERVICE_CORE_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.config_loader import ConfigLoaderPort" in content
    assert "from pff.application.ports.settings import SettingsPort" in content
    assert "config_loader: ConfigLoaderPort | None = None" in content
    assert "settings_obj: SettingsPort | None = None" in content
    assert "self._config_loader = config_loader or load_config" in content
    assert "self._settings = settings_obj or default_settings" in content
    assert "config_loader=self._config_loader" in content


def test_model_integration_must_accept_file_manager_port() -> None:
    """ModelIntegration must type file manager as application port."""
    content = MODEL_INTEGRATION_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.file_manager import FileManagerPort" in content
    assert "file_manager: FileManagerPort | None = None" in content


def test_rule_engine_must_accept_file_manager_port() -> None:
    """RuleEngine must type file manager dependency via application port."""
    content = RULE_ENGINE_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.file_manager import FileManagerPort" in content
    assert "from pff.application.ports.config_loader import ConfigLoaderPort" in content
    assert "from pff.application.ports.settings import SettingsPort" in content
    assert "file_manager: FileManagerPort | None = None" in content
    assert "config_loader: ConfigLoaderPort | None = None" in content
    assert "settings_obj: SettingsPort | None = None" in content
    assert "self.file_manager = file_manager or FileManager()" in content


def test_rule_validator_must_accept_config_loader_port() -> None:
    """RuleValidator must type config loader via application port."""
    content = RULE_VALIDATOR_PATH.read_text(encoding="utf-8")

    assert "from pff.application.ports.config_loader import ConfigLoaderPort" in content
    assert "config_loader: ConfigLoaderPort | None = None" in content


def test_orchestrator_must_not_use_application_services_wrapper_imports() -> None:
    """Drivers should import concrete service modules instead of package aggregators."""
    content = ORCHESTRATOR_PATH.read_text(encoding="utf-8")

    assert "from pff.application.services import" not in content
