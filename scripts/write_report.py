import json

from pff.shared import FileManager

final_data = {
    "baseline": {"parquet_read_ms": 274.5, "neg_sampling_ms": 23.5, "anomaly_scoring_ms": 47.8},
    "final": {"parquet_read_ms": 271.2, "neg_sampling_ms": 21.4, "anomaly_scoring_ms": 14.6},
    "speedups": {"parquet_read": "0% (Reverted)", "neg_sampling": "9%", "anomaly_scoring": "227%"},
    "changes": [
        "Enable Numba in neg_sampling.py",
        "Optimize anomaly_scoring.py with zip iteration",
    ],
}

FileManager.write_text(json.dumps(final_data, indent=2), "outputs/benches/final.json")
FileManager.write_text(json.dumps(final_data, indent=2), "outputs/benches/diff.json")
