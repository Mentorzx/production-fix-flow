import signal
import threading
from typing import Callable, Optional

from pff.utils import logger


class GlobalInterruptManager:
    _instance: Optional["GlobalInterruptManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "GlobalInterruptManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._should_stop = False
        self._signal_received = False
        self._callbacks: list[Callable[[], None]] = []
        self._original_handlers = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum: int, frame) -> None:
            signal_name = signal.Signals(signum).name
            logger.warning(
                f"🛑 {signal_name} recebido - iniciando shutdown coordenado..."
            )
            self._should_stop = True
            self._signal_received = True
            for callback in self._callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Erro em callback de interrupção: {e}")

            logger.info("📢 Sinal de parada propagado para todos os componentes")

        for sig in [signal.SIGINT, signal.SIGTERM]:
            self._original_handlers[sig] = signal.signal(sig, signal_handler)

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def signal_received(self) -> bool:
        return self._signal_received

    def register_callback(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def force_stop(self, reason: str = "Manual") -> None:
        logger.warning(f"🛑 Parada forçada solicitada: {reason}")
        self._should_stop = True
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Erro em callback de parada forçada: {e}")

    def reset(self) -> None:
        logger.debug("Resetando GlobalInterruptManager")
        self._should_stop = False
        self._signal_received = False
        self._callbacks.clear()

    def restore_original_handlers(self) -> None:
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        logger.debug("Signal handlers originais restaurados")

    def __del__(self):
        try:
            self.restore_original_handlers()
        except Exception:
            pass


def get_interrupt_manager() -> GlobalInterruptManager:
    return GlobalInterruptManager()


def should_stop() -> bool:
    return get_interrupt_manager().should_stop


def register_interrupt_callback(callback: Callable[[], None]) -> None:
    get_interrupt_manager().register_callback(callback)


def check_interruption() -> None:
    """
    Verifica se houve interrupção e levanta exceção se necessário.
    
    Raises:
        KeyboardInterrupt: Se interrupção foi detectada
    """
    manager = get_interrupt_manager()
    if manager.should_stop:
        logger.warning("🛑 Operação interrompida pelo GlobalInterruptManager")
        raise KeyboardInterrupt("Operação foi interrompida")


def interruptible(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if should_stop():
            logger.warning(f"🛑 Função {func.__name__} interrompida pelo GlobalInterruptManager")
            raise KeyboardInterrupt(f"Função {func.__name__} foi interrompida")

        try:
            result = func(*args, **kwargs)
            return result
        except KeyboardInterrupt:
            logger.info(f"🛑 {func.__name__} interrompida graciosamente")
            raise

    return wrapper


if __name__ == "__main__":
    import time

    manager = get_interrupt_manager()
    print(f"Should stop: {manager.should_stop}")

    def test_callback():
        print("Callback executado!")

    manager.register_callback(test_callback)

    print("Teste signal handler (CTRL+C para testar)...")
    try:
        for i in range(10):
            if should_stop():
                print("Parada detectada!")
                break
            print(f"Iteração {i + 1}/10")
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt capturado")

    print("Teste concluído")
