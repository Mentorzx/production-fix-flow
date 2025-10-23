import re

import pyperclip

from pff import settings
from pff.utils import FileManager, logger


class IntelligentPreprocessor:
    """
    IntelligentPreprocessor is a utility class for preprocessing raw text input into structured task data,
    generating manifest files, and integrating clipboard operations for streamlined workflow automation.
    This class provides methods to:
    - Parse raw text into a list of task dictionaries using configurable regular expression patterns.
    - Generate and save a manifest file containing execution metadata and parsed tasks.
    - Read text from the clipboard, process it into tasks, and generate a manifest file.
    Attributes:
        file_manager (FileManager): Handles file operations such as saving manifest files.
        REGEX_MSISDN_ONLY (re.Pattern): Regex pattern to match lines containing only an MSISDN.
        REGEX_MSISDN_AND_SEQUENCE (re.Pattern): Regex pattern to match lines containing an MSISDN and a sequence.
        PATTERNS (list[dict]): List of pattern configurations for parsing input lines.
    Methods:
        parse_text(raw_text: str, default_sequence: str | None = None) -> list[dict]: Parses raw text into a list of task dictionaries based on predefined regex patterns.
        generate_manifest_file(tasks: list[dict], output_path: str, exec_id: str = "execucao-gerada"): Generates and saves a manifest file containing execution metadata and tasks.
        process_from_clipboard(default_sequence: str, output_path: str): Processes text from the clipboard and generates a manifest file.
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
                    f"[Linha {line_num:02d} ✔️] Tarefa completa encontrada: {task}"
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
                        f"[Linha {line_num:02d} ✔️] MSISDN encontrado, usando sequência padrão: {task}"
                    )
                else:
                    logger.warning(
                        f"[Linha {line_num:02d} ⚠️] Ignorada: MSISDN '{match_msisdn_only.groups()[0]}' encontrado sem sequência associada (e nenhuma padrão foi fornecida)."
                    )
                continue

            logger.info(
                f"[Linha {line_num:02d} ❌] Ignorada (formato não reconhecido): '{line[:70]}...'"
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
            logger.warning("Nenhuma tarefa encontrada para gerar o manifesto.")
            return

        manifest_data = {"execution_id": exec_id, "max_workers": 16, "tasks": tasks}

        full_path = settings.DATA_DIR / output_path
        try:
            self.file_manager.save(manifest_data, full_path)
            logger.success(f"🎉 Manifesto gerado com sucesso em: {full_path}")
        except Exception as e:
            logger.error(f"Falha ao salvar o manifesto usando o FileManager: {e}")

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
                logger.warning("Nenhum texto encontrado na área de transferência.")
                return
        except Exception as e:
            logger.error(
                f"Não foi possível acessar a área de transferência. Verifique se o ambiente gráfico está disponível. Erro: {e}"
            )
            return

        tasks = self.parse_text(raw_text, default_sequence=default_sequence)
        self.generate_manifest_file(tasks, output_path)
