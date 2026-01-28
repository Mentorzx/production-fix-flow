import os
import re


def fix_recharts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

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
        return

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

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed Recharts in {file_path}")


target_dirs = [
    "static/js/features/hpo/charts",
    "static/js/features/hpo",
    "static/js/ui",
    "static/js/layout",
]

for directory in target_dirs:
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jsx"):
                fix_recharts(os.path.join(directory, filename))
