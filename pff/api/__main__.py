import os

import uvicorn

from pff.api.main import app
from pff.utils.logger import logger

# trunk-ignore(bandit/B104)
HOST = os.getenv("HOST", "0.0.0.0")
try:
    PORT = int(os.getenv("PORT", "8000"))
except ValueError:
    PORT = 8000

logger.info(f"Iniciando Uvicorn em {HOST}:{PORT} ...")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        # Se quiser live-reload em dev, basta descomentar a linha abaixo:
        # reload=True,
    )
