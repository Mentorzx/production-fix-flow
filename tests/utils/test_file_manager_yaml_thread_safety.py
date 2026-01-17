from concurrent.futures import ThreadPoolExecutor

from pff.shared.core.file_manager import FileManager
from pff.config import OPTIMIZATION_CONFIG_PATH


def test_yaml_reads_are_thread_safe() -> None:
    fm = FileManager()

    def _read_config() -> dict:
        data = fm.read(OPTIMIZATION_CONFIG_PATH, return_native=True)
        assert isinstance(data, dict)
        return data

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_read_config) for _ in range(20)]
        for fut in futures:
            fut.result()
