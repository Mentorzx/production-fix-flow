import xml.etree.ElementTree as ET
import os
from collections import defaultdict

report_files = [
    "report_unit_light.xml",
    "report_unit_repro.xml",
    "report_int.xml",
    "report_misc.xml",
    "report_perf.xml",
]

stats = defaultdict(
    lambda: {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "total": 0, "failures": []}
)

for report_file in report_files:
    if not os.path.exists(report_file):
        print(f"Warning: {report_file} not found.")
        continue

    try:
        tree = ET.parse(report_file)
        root = tree.getroot()

        for testcase in root.findall(".//testcase"):
            classname = testcase.get("classname")
            name = testcase.get("name")

                                  
            parts = classname.split(".")
            if len(parts) >= 2 and parts[0] == "tests":
                category = f"{parts[0]}/{parts[1]}"
            else:
                category = "tests/other"

            stats[category]["total"] += 1

            skipped = testcase.find("skipped")
            failure = testcase.find("failure")
            error = testcase.find("error")

            if skipped is not None:
                stats[category]["skipped"] += 1
            elif failure is not None:
                stats[category]["failed"] += 1
                msg = failure.get("message", "No failure message")
                stats[category]["failures"].append(f"{classname}::{name} -> {msg}")
            elif error is not None:
                stats[category]["error"] += 1
                msg = error.get("message", "No error message")
                stats[category]["failures"].append(f"{classname}::{name} (ERROR) -> {msg}")
            else:
                stats[category]["passed"] += 1

    except Exception as e:
        print(f"Error parsing {report_file}: {e}")

print("\n" + "=" * 100)
print(f"{'FOLDER':<30} | {'TOTAL':<6} | {'PASS':<6} | {'FAIL':<6} | {'SKIP':<6} | {'ERR':<6}")
print("-" * 100)

total_passed = 0
total_failed = 0
total_skipped = 0
total_error = 0

for category in sorted(stats.keys()):
    data = stats[category]
    print(
        f"{category:<30} | {data['total']:<6} | {data['passed']:<6} | {data['failed']:<6} | {data['skipped']:<6} | {data['error']:<6}"
    )
    total_passed += data["passed"]
    total_failed += data["failed"]
    total_skipped += data["skipped"]
    total_error += data["error"]

print("-" * 100)
print(
    f"{'TOTAL':<30} | {total_passed + total_failed + total_skipped + total_error:<6} | {total_passed:<6} | {total_failed:<6} | {total_skipped:<6} | {total_error:<6}"
)
print("=" * 100)

print("\n\n=== FAILURE DETAILS ===")
for category in sorted(stats.keys()):
    if stats[category]["failures"]:
        print(f"\n📁 {category}:")
        for fail in stats[category]["failures"]:
            print(f"  ❌ {fail}")
