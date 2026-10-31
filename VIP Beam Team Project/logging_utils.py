"""CSV (one row per attempted tool execution / invalid proposal) + JSONL (raw LLM traffic) logging."""
from __future__ import annotations

import csv
import json
from pathlib import Path

COLUMNS = [
    "workflow", "run", "iteration", "request", "row_type", "requested_tool", "model_source",
    "width_mm", "height_mm", "reason", "mass_kg", "deflection_mm", "nominal_stress_mpa",
    "deflection_margin_mm", "stress_margin_mpa", "feasible", "status", "error", "elapsed_s",
    "cache_hit", "prior_evidence", "evaluation_id", "mesh", "project_version", "llm_elapsed_s",
]


class ResultsLogger:
    def __init__(self, csv_path, jsonl_path):
        self.csv_path, self.jsonl_path = Path(csv_path), Path(jsonl_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            self._w.writeheader()
        self._j = open(self.jsonl_path, "a", encoding="utf-8")

    def log_tool(self, workflow, run, iteration, request, row_type, requested_tool, reason, res,
                 llm_elapsed_s=0.0):
        row = {k: res.get(k) for k in COLUMNS if k in res}
        row.update(workflow=workflow, run=run, iteration=iteration, request=request, row_type=row_type,
                   requested_tool=requested_tool, reason=reason, llm_elapsed_s=llm_elapsed_s,
                   prior_evidence=bool(res.get("prior_evidence", False)))
        self._w.writerow(row)
        self._f.flush()

    def log_invalid(self, workflow, run, iteration, requested_tool, reason, error, llm_elapsed_s,
                    row_type="invalid_proposal"):
        self._w.writerow(dict(workflow=workflow, run=run, iteration=iteration, request=0,
                              row_type=row_type, requested_tool=requested_tool, reason=reason,
                              status="invalid_response", error=error, llm_elapsed_s=llm_elapsed_s))
        self._f.flush()

    def log_llm(self, record: dict):
        self._j.write(json.dumps(record, default=str) + "\n")
        self._j.flush()

    def close(self):
        self._f.close()
        self._j.close()
