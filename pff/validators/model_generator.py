import shutil
import subprocess
from pathlib import Path

from pff.utils import FileManager, logger


def generate_model(
    schema_path: str = "pff/validators/patterns/schema.json",
    output_path: str = "pff/validators/model.py",
) -> None:
    """
    Generates a Pydantic model from a JSON schema using datamodel-codegen.

    Args:
        schema_path (str): Path to the input JSON schema file.
        output_path (str): Path where the generated model will be saved.

    Raises:
        RuntimeError: If datamodel-codegen is not found or generation fails.
        FileNotFoundError: If the specified schema file does not exist.
    """
    binary = shutil.which("datamodel-codegen")
    if not binary:
        logger.error("datamodel-codegen not found in PATH")
        raise RuntimeError("datamodel-codegen not found in PATH")
    FileManager.read(schema_path)

    try:
        subprocess.run(
            [
                binary,
                "--input",
                str(Path(schema_path)),
                "--input-file-type",
                "jsonschema",
                "--output",
                str(Path(output_path)),
            ],
            check=True,
        )
        logger.success(f"Pydantic model generated at {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"datamodel-codegen failed with exit code {e.returncode}")
        raise RuntimeError(
            f"datamodel-codegen failed with exit code {e.returncode}"
        ) from e


if __name__ == "__main__":
    generate_model()
