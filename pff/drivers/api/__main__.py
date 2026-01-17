import os

import uvicorn

from pff.drivers.api.main import app
from pff.shared.core.logger import logger

# trunk-ignore(bandit/B104)
HOST = os.getenv("HOST", "0.0.0.0")
try:
    PORT = int(os.getenv("PORT", "8000"))
except ValueError:
    PORT = 8000

logger.info(f"Iniciando Uvicorn em {HOST}:{PORT} ...")

if __name__ == "__main__":
    from pff.shared.determinism import (
        configure_numba_threads,
        configure_torch_determinism,
    )
    from pff.shared.system.runtime import initialize_runtime
    from pff import __version__

    configure_torch_determinism(enforce=True)
    configure_numba_threads()
    initialize_runtime(__version__)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        # Se quiser live-reload em dev, basta descomentar a linha abaixo:
        # reload=True,
    )
