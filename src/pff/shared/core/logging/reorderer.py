"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/shared/core/logging/reorderer.py

"""

from __future__ import annotations

import re
from pathlib import Path

import orjson


class LogReorderer:
    """Reorders log entries in specified file by thread and MSISDN."""

    HEADER_PREFIX: str = "===== THREAD"
    TASK_PATTERN = re.compile(r"\[([^\]]+)\]")
    THREAD_PATTERN = re.compile(r"Thread-\d+")

    @staticmethod
    def _extract(line: str) -> tuple[str, str | None, str]:
        """Execute extract.



        Args:

            line: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not line or line.startswith(LogReorderer.HEADER_PREFIX):
            return "_meta", None, line
        try:
            rec = orjson.loads(line)

            record_dict = rec.get("record", {})
            tname = record_dict.get("thread", {}).get("name", "_meta")
            extra = record_dict.get("extra", {})
            msisdn = extra.get("msisdn") or extra.get("task_id")

            text = rec.get("text", "")
            if not text and "message" in record_dict:
                text = record_dict["message"]

            if not text:
                text = f"{rec['text']}" if "text" in rec else line

            return tname, msisdn, text.rstrip()
        except orjson.JSONDecodeError:
            parts = line.split("|")
            if len(parts) >= 4:
                task_match = LogReorderer.TASK_PATTERN.search(line)
                msisdn = task_match.group(1) if task_match else None
                thread_match = LogReorderer.THREAD_PATTERN.search(line)
                tname = thread_match.group(0) if thread_match else "MainThread"
                return tname, msisdn, line
            return "_meta", None, line

    @staticmethod
    def reorder(file_path: Path) -> Path:
        """Execute reorder.



        Args:

            file_path: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        from pff.shared.core.file_manager import FileManager

        thread_handles: dict[str, list[tuple[str | None, str]]] = {}

        output_path = file_path.with_suffix(".tmp")
        content = FileManager.read_text(file_path)

        for line in content.splitlines():
            line = line.rstrip("\n")
            if not line:
                continue
            thr, msisdn, txt = LogReorderer._extract(line)

            if thr not in thread_handles:
                thread_handles[thr] = []
            thread_handles[thr].append((msisdn, txt))

        output_lines = []
        for thr in sorted(thread_handles.keys()):
            if thr == "_meta":
                for _, txt in thread_handles[thr]:
                    output_lines.append(f"{txt}\n")
                continue

            output_lines.append(f"\n{LogReorderer.HEADER_PREFIX} {thr} =====\n")
            last_msisdn = None
            for msisdn, txt in thread_handles[thr]:
                if msisdn and msisdn != last_msisdn:
                    output_lines.append("\n")
                    last_msisdn = msisdn
                output_lines.append(f"{txt}\n")

        FileManager.write_text("".join(output_lines), output_path)
        output_path.replace(file_path)
        return file_path
