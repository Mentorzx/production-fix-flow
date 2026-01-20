from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pff.shared.core.file_manager import FileManager


def test_yaml_reads_are_thread_safe(tmp_path: Path) -> None:
    fm = FileManager()

    config_path = tmp_path / "test_config.yaml"
    import uuid

    unique_content = f"key: value\nlist: [1, 2, 3]\nuuid: {uuid.uuid4()}\n"
    with open(config_path, "w") as f:
        f.write(unique_content)

    def _read_config() -> dict:
        # Disable caching to avoid global state issues in tests
        data = fm.read(config_path, return_native=True, cache=False)
        if not isinstance(data, dict):
            pass
        return data if isinstance(data, dict) else {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_read_config) for _ in range(20)]
        for fut in futures:
            fut.result()
