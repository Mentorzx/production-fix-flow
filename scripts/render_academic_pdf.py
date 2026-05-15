#!/usr/bin/env python3

"""Render a constrained Markdown article into styled HTML and PDF."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import unicodedata
from pathlib import Path

from pff.shared.core.file_manager import FileManager


CSS = """
@page {
  size: A4;
  margin: 2.2cm 2cm 2.4cm 2cm;
}

body {
  font-family: "Liberation Serif", "DejaVu Serif", serif;
  color: #1f1b16;
  line-height: 1.62;
  font-size: 11.5pt;
  margin: 0;
  background: #f5f1e8;
}

.sheet {
  background: #fffdfa;
  padding: 0;
}

.cover {
  min-height: 25.5cm;
  padding: 2.2cm 1.7cm 1.8cm 1.7cm;
  background:
    linear-gradient(180deg, #f1e7d2 0%, #fffdfa 32%, #fffdfa 100%);
  border: 1.2pt solid #cab58b;
  position: relative;
}

.cover::before {
  content: "";
  position: absolute;
  left: 1.2cm;
  right: 1.2cm;
  top: 1.2cm;
  height: 0.18cm;
  background: linear-gradient(90deg, #8a6a2f 0%, #c99b3d 100%);
}

.cover-kicker {
  margin-top: 1.6cm;
  font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
  font-size: 10pt;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #7b6030;
}

.cover h1 {
  margin: 4.2cm 0 1.1cm 0;
  font-size: 25pt;
  line-height: 1.18;
  color: #2f2416;
}

.cover .subtitle {
  font-size: 13pt;
  color: #5b4f41;
  margin-bottom: 4cm;
}

.cover-meta {
  margin-top: 5.2cm;
  padding: 0.7cm 0.8cm;
  border-left: 4pt solid #b98a2f;
  background: #f8f1e4;
  font-size: 11pt;
}

.page-break {
  page-break-before: always;
}

.content {
  padding: 0.5cm 0 0 0;
}

.lead {
  font-size: 12.3pt;
  color: #473d31;
}

.toc {
  border: 1pt solid #d9ccba;
  background: #fcf8f1;
  padding: 0.8cm 0.9cm;
  margin: 0 0 1cm 0;
}

.toc h2 {
  margin-top: 0;
}

.toc ul {
  margin: 0.2cm 0 0 0.45cm;
  padding: 0;
}

.toc li {
  margin: 0.12cm 0;
}

h1, h2, h3 {
  color: #2f2416;
  page-break-after: avoid;
}

h1 {
  font-size: 22pt;
  margin: 0 0 0.7cm 0;
}

h2 {
  font-size: 15.5pt;
  margin: 1cm 0 0.35cm 0;
  padding-bottom: 0.12cm;
  border-bottom: 1pt solid #d6c09a;
}

h3 {
  font-size: 12.5pt;
  margin: 0.7cm 0 0.2cm 0;
}

p {
  margin: 0.22cm 0 0.28cm 0;
  text-align: justify;
}

ul, ol {
  margin: 0.18cm 0 0.35cm 0.65cm;
  padding: 0;
}

li {
  margin: 0.12cm 0;
}

code {
  font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
  font-size: 9.5pt;
  background: #f5efe3;
  padding: 0.03cm 0.08cm;
  border-radius: 2px;
}

pre {
  background: #fbf6ec;
  border: 1pt solid #dccdb4;
  padding: 0.38cm 0.42cm;
  white-space: pre-wrap;
  line-height: 1.38;
  font-size: 9.8pt;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.35cm 0 0.45cm 0;
  font-size: 10.2pt;
}

th, td {
  border: 1pt solid #d8c9b2;
  padding: 0.14cm 0.18cm;
  vertical-align: top;
}

th {
  background: #efe5d2;
  font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
  font-weight: bold;
}

tr:nth-child(even) td {
  background: #fdfaf4;
}

a {
  color: #72521a;
  text-decoration: none;
}

.abstract-box {
  border: 1pt solid #d8c8ae;
  background: #fcf8f1;
  padding: 0.55cm 0.65cm;
  margin: 0.55cm 0 0.7cm 0;
}
"""


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower())
    return slug.strip("-") or "secao"


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head, *body = rows
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{inline_markup(cell)}</th>" for cell in head)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{inline_markup(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_markdown(markdown_text: str) -> tuple[str, str, str, list[tuple[int, str, str]]]:
    lines = markdown_text.splitlines()
    html_parts: list[str] = []
    headings: list[tuple[int, str, str]] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_kind: str | None = None
    table_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    title = ""
    subtitle = ""
    lead = ""

    def flush_paragraph() -> None:
      nonlocal lead, subtitle
      if not paragraph:
        return
      text = " ".join(part.strip() for part in paragraph)
      if title and not headings and not subtitle:
        subtitle = text
        paragraph.clear()
        return
      css_class = " class=\"lead\"" if not lead and title else ""
      html_parts.append(f"<p{css_class}>{inline_markup(text)}</p>")
      if not lead and title:
        lead = text
      paragraph.clear()

    def flush_list() -> None:
        nonlocal list_kind
        if not list_items:
            return
        tag = "ol" if list_kind == "ol" else "ul"
        html_parts.append(f"<{tag}>")
        html_parts.extend(f"<li>{inline_markup(item)}</li>" for item in list_items)
        html_parts.append(f"</{tag}>")
        list_items.clear()
        list_kind = None

    def flush_table() -> None:
        if not table_lines:
            return
        html_parts.append(render_table(table_lines))
        table_lines.clear()

    def flush_code() -> None:
        if not code_lines:
            return
        html_parts.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
        code_lines.clear()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_list()
                flush_table()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            flush_list()
            table_lines.append(line)
            continue

        flush_table()

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            if level == 1 and not title:
                title = text
                continue
            if level == 1 and title and not subtitle:
                subtitle = text
                continue
            anchor = slugify(text)
            headings.append((level, text, anchor))
            box_class = " class=\"abstract-box\"" if text in {"Resumo", "Abstract"} else ""
            html_parts.append(f"<section id=\"{anchor}\"{box_class}><h{level}>{inline_markup(text)}</h{level}>")
            html_parts.append("</section>")
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        unordered_match = re.match(r"^-\s+(.*)$", stripped)
        if ordered_match or unordered_match:
            flush_paragraph()
            item = ordered_match.group(1) if ordered_match else unordered_match.group(1)
            kind = "ol" if ordered_match else "ul"
            if list_kind and list_kind != kind:
                flush_list()
            list_kind = kind
            list_items.append(item)
            continue

        paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()
    if in_code:
        flush_code()

    content = "\n".join(html_parts)
    return title, subtitle, content, headings


def inject_sections(content: str) -> str:
    pattern = re.compile(r"<section id=\"([^\"]+)\"( class=\"abstract-box\")?><h([1-3])>(.*?)</h\3>\s*</section>")
    result = []
    cursor = 0
    matches = list(pattern.finditer(content))
    for index, match in enumerate(matches):
        result.append(content[cursor:match.start()])
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        result.append(
            f'<section id="{match.group(1)}"{match.group(2) or ""}><h{match.group(3)}>{match.group(4)}</h{match.group(3)}>{content[start:end]}</section>'
        )
        cursor = end
    result.append(content[cursor:])
    return "".join(result)


def build_toc(headings: list[tuple[int, str, str]]) -> str:
    items = []
    for level, text, anchor in headings:
        if level > 2:
            continue
        items.append(f'<li><a href="#{anchor}">{inline_markup(text)}</a></li>')
    if not items:
        return ""
    return '<div class="toc"><h2>Sumário</h2><ul>' + "".join(items) + "</ul></div>"


def build_html(title: str, subtitle: str, content: str, headings: list[tuple[int, str, str]], author: str, institution: str) -> str:
    rendered_content = inject_sections(content)
    toc = build_toc(headings)
    subtitle_html = f'<div class="subtitle">{inline_markup(subtitle)}</div>' if subtitle else ""
    return f"""<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class=\"sheet\">
    <section class=\"cover\">
      <div class=\"cover-kicker\">Documento acadêmico</div>
      <h1>{inline_markup(title)}</h1>
      {subtitle_html}
      <div class=\"cover-meta\">
        <p><strong>Instituição:</strong> {inline_markup(institution)}</p>
        <p><strong>Autor institucional:</strong> {inline_markup(author)}</p>
        <p><strong>Data de geração:</strong> 06 de maio de 2026</p>
        <p><strong>Escopo:</strong> Search Space Advisor, confiabilidade estatística e governança de HPO</p>
      </div>
    </section>
    <div class=\"page-break\"></div>
    <main class=\"content\">
      {toc}
      {rendered_content}
    </main>
  </div>
</body>
</html>
"""


def convert_to_pdf(html_path: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(html_path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Markdown source article")
    parser.add_argument("--html-output", type=Path, help="HTML output path")
    parser.add_argument("--pdf-output", type=Path, help="PDF output path")
    parser.add_argument("--author", default="Projeto PFF")
    parser.add_argument("--institution", default="PFF Platform")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    html_output = args.html_output.resolve() if args.html_output else source_path.with_suffix(".html")
    pdf_output = args.pdf_output.resolve() if args.pdf_output else source_path.with_suffix(".pdf")

    markdown_text = FileManager.read_text(source_path)
    title, subtitle, content, headings = render_markdown(markdown_text)
    html_document = build_html(title, subtitle, content, headings, args.author, args.institution)
    FileManager.write_text(html_document, html_output)
    convert_to_pdf(html_output, pdf_output.parent)

    generated_pdf = html_output.with_suffix(".pdf")
    if generated_pdf != pdf_output:
        FileManager.write_bytes(FileManager.read_bytes(generated_pdf), pdf_output)


if __name__ == "__main__":
    main()
