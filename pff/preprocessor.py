import re

import pyperclip

from pff import settings
from pff.shared import FileManager, logger


class IntelligentPreprocessor:
    """Preprocesses raw text into structured task data with manifest generation.

    Design Patterns Applied:
        - **Strategy Pattern:** Uses configurable regex PATTERNS list to select
          parsing strategy for different input formats (MSISDN-only vs MSISDN+sequence).
        - **Template Method:** The `process_from_clipboard()` method defines the
          skeleton algorithm: read -> parse -> generate manifest.
        - **Factory Method (implicit):** Pattern matching produces task dictionaries
          based on input format detection.

    Performance Optimizations:
        - Pre-compiled regex patterns (REGEX_MSISDN_ONLY, REGEX_MSISDN_AND_SEQUENCE).
        - FileManager used for all I/O operations (AGENTS.md compliance).

    Attributes:
        file_manager: Handles file operations for manifest saving.
        PATTERNS: List of pattern configurations for parsing input lines.

    Methods:
        parse_text: Parses raw text into task dictionaries.
        generate_manifest_file: Creates manifest YAML from tasks.
        process_from_clipboard: End-to-end clipboard processing pipeline.
    """

    def __init__(self):
        self.file_manager = FileManager()

    REGEX_MSISDN_ONLY = re.compile(r"^\s*(\d{11,13})\s*$")
    REGEX_MSISDN_AND_SEQUENCE = re.compile(
        r"^\s*(\d{11,13})\s*[-–—\s]+\s*([\w\.]+)\s*$"
    )
    PATTERNS = [
        {"regex": REGEX_MSISDN_AND_SEQUENCE, "fields": ["msisdn", "sequence"]},
        {"regex": REGEX_MSISDN_ONLY, "fields": ["msisdn"]},
    ]

    def parse_text(
        self, raw_text: str, default_sequence: str | None = None
    ) -> list[dict]:
        """
        Parses raw text input to extract tasks containing MSISDN and sequence information.
        Each line in the input text is processed to identify either:
          - A complete task with both MSISDN and sequence.
          - A task with only MSISDN, using a default sequence if provided.
          - Lines not matching expected patterns are ignored.
        Args:
            raw_text (str): The raw input text, with each line potentially containing an MSISDN and an optional sequence.
            default_sequence (str | None, optional): The default sequence to use for lines containing only an MSISDN.
                If not provided, such lines are ignored.
        Returns:
            list[dict]: A list of dictionaries, each containing 'msisdn' and 'sequence' keys for valid tasks extracted from the input.
        """

        tasks = []
        logger.info("--- Iniciando pré-processamento de texto ---")
        if not default_sequence:
            logger.warning(
                "Nenhuma sequência padrão fornecida. Linhas contendo apenas MSISDNs serão ignoradas."
            )

        for line_num, line in enumerate(raw_text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            match_full = self.PATTERNS[0]["regex"].match(line)
            if match_full:
                task = {
                    "msisdn": match_full.groups()[0],
                    "sequence": match_full.groups()[1],
                }
                tasks.append(task)
                logger.debug(
                    f"[Linha {line_num:02d} ] Tarefa completa encontrada: {task}"
                )
                continue
            match_msisdn_only = self.PATTERNS[1]["regex"].match(line)
            if match_msisdn_only:
                if default_sequence:
                    task = {
                        "msisdn": match_msisdn_only.groups()[0],
                        "sequence": default_sequence,
                    }
                    tasks.append(task)
                    logger.debug(
                        f"[Linha {line_num:02d} ] MSISDN encontrado, usando sequência padrão: {task}"
                    )
                else:
                    logger.warning(
                        f"[Linha {line_num:02d} ] Ignorada: MSISDN '{match_msisdn_only.groups()[0]}' encontrado sem sequência associada (e nenhuma padrão foi fornecida)."
                    )
                continue

            logger.info(
                f"[Linha {line_num:02d} ] Ignorada (formato não reconhecido): '{line[:70]}...'"
            )
        logger.success(
            f"--- Pré-processamento finalizado: {len(tasks)} tarefas válidas encontradas. ---"
        )

        return tasks

    def generate_manifest_file(
        self, tasks: list[dict], output_path: str, exec_id: str = "execucao-gerada"
    ):
        """
        Generates a manifest file containing execution metadata and a list of tasks.
        Args:
            tasks (list[dict]): A list of task dictionaries to include in the manifest.
            output_path (str): The relative path where the manifest file will be saved.
            exec_id (str, optional): The execution identifier to include in the manifest. Defaults to "execucao-gerada".
        Returns:
            None
        Logs:
            - Warning if no tasks are provided.
            - Success message if the manifest is generated and saved successfully.
            - Error message if saving the manifest fails.
        Raises:
            Exception: If an error occurs while saving the manifest file.
        """
        if not tasks:
            logger.warning("No tasks found for manifest generation.")
            return

        manifest_data = {"execution_id": exec_id, "max_workers": 16, "tasks": tasks}

        full_path = settings.DATA_DIR / output_path
        try:
            self.file_manager.save(manifest_data, full_path)
            logger.success(f" Manifesto gerado com sucesso em: {full_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest using FileManager: {e}")

    def process_from_clipboard(self, default_sequence: str, output_path: str):
        """
        Reads text from the clipboard, parses it into tasks, and generates a manifest file.
        Args:
            default_sequence (str): The default sequence to use when parsing the clipboard text.
            output_path (str): The file path where the generated manifest file will be saved.
        Logs:
            - Info message when reading from the clipboard.
            - Warning if no text is found in the clipboard.
            - Error if the clipboard cannot be accessed.
        Returns:
            None
        """
        logger.info("Lendo texto da área de transferência...")
        try:
            raw_text = pyperclip.paste()
            if not raw_text.strip():
                logger.warning("No text found in clipboard.")
                return
        except Exception as e:
            logger.error(
                f"Could not access clipboard. Check if graphical environment is available. Error: {e}"
            )
            return

        tasks = self.parse_text(raw_text, default_sequence=default_sequence)
        self.generate_manifest_file(tasks, output_path)
