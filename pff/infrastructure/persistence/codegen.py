import shutil
import subprocess
from pathlib import Path

from pff.shared import logger
from pff.shared.core.config import settings


def generate_model(
    schema_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    """
    Generates a Pydantic model from a JSON schema using datamodel-codegen.

    Args:
        schema_path (str | Path | None): Path to the input JSON schema file. Defaults to outputs/ensemble/rules/schema.json.
        output_path (str | Path | None): Path where the generated model will be saved. Defaults to pff/validators/model.py.
    """
    default_schema = settings.OUTPUTS_DIR / "ensemble" / "rules" / "schema.json"
    default_output = settings.ROOT_DIR / "pff" / "validators" / "model.py"

    resolved_schema = Path(schema_path) if schema_path else default_schema
    resolved_output = Path(output_path) if output_path else default_output

    binary = shutil.which("datamodel-codegen")
    if not binary:
        logger.error("datamodel-codegen not found in PATH")
        raise RuntimeError("datamodel-codegen not found in PATH")

    if not resolved_schema.exists():
        logger.error(f"Schema not found at {resolved_schema}")
        raise FileNotFoundError(f"Schema not found at {resolved_schema}")

    try:
        subprocess.run(
            [
                binary,
                "--input",
                str(resolved_schema),
                "--input-file-type",
                "jsonschema",
                "--output",
                str(resolved_output),
            ],
            check=True,
        )
        logger.success(f"Modelo Pydantic gerado em {resolved_output}")
    except subprocess.CalledProcessError as e:
        logger.error(f"datamodel-codegen failed with exit code {e.returncode}")
        raise RuntimeError(
            f"datamodel-codegen failed with exit code {e.returncode}"
        ) from e


if __name__ == "__main__":
    generate_model()
