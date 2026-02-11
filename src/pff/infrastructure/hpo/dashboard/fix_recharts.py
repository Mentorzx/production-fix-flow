import re
from pathlib import Path

from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


def fix_recharts(file_path: Path) -> bool:
    content = FileManager.read_text(file_path)

    recharts_components = [
        "LineChart",
        "BarChart",
        "AreaChart",
        "ComposedChart",
        "ScatterChart",
        "PieChart",
        "RadarChart",
        "RadialBarChart",
        "Treemap",
        "FunnelChart",
        "Line",
        "Bar",
        "Area",
        "Scatter",
        "Pie",
        "Radar",
        "RadialBar",
        "Treemap",
        "Funnel",
        "XAxis",
        "YAxis",
        "ZAxis",
        "CartesianGrid",
        "PolarGrid",
        "PolarAngleAxis",
        "PolarRadiusAxis",
        "Tooltip",
        "Legend",
        "ResponsiveContainer",
        "Cell",
        "ReferenceLine",
        "ReferenceDot",
        "ReferenceArea",
        "ErrorBar",
    ]

    used_components = []
    for comp in recharts_components:
        if comp in content and f"import {{ {comp}" not in content:
            used_components.append(comp)

    if not used_components:
        return False

    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if "window.Recharts" in line or "window.recharts" in line or "RechartsLib" in line:
            continue
        new_lines.append(line)

    content = "\n".join(new_lines)

    import_stmt = f"import {{ {', '.join(used_components)} }} from 'recharts';"

    if "import React" in content:
        content = re.sub(r"(import React.*?;)", f"\\1\n{import_stmt}", content)
    else:
        content = import_stmt + "\n" + content

    FileManager.write_text(content, file_path)
    return True


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    target_dirs = [
        base_dir / "static" / "js" / "features" / "hpo" / "charts",
        base_dir / "static" / "js" / "features" / "hpo",
        base_dir / "static" / "js" / "ui",
        base_dir / "static" / "js" / "layout",
    ]
    for directory in target_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.jsx"):
            if fix_recharts(path):
                logger.info(f"Recharts corrigido em {path}")


if __name__ == "__main__":
    main()
