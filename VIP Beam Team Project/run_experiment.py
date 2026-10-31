"""Task 4 driver: workflows A/B/C with the same LLM/settings/prompt/start/iterations.
Usage: python run_experiment.py [--runs 1] [--workflows ABC] [--mock] [--stub-llm]
Run run_verification.py first (creates the shared paired cases)."""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")

import analysis
import ansys_backend
import beam_tools
import config as C
import final_checks
from agent import StubLLM, make_llm
from logging_utils import ResultsLogger
from workflows import WorkflowRun


def run_experiment(mock=False, stub_llm=False, runs=1, workflows="ABC", backend=None):
    """Pass an already-started `backend` to reuse one persistent ANSYS session (it is then not closed here)."""
    paired_path = C.path("beam_paired.json")
    if not paired_path.exists():
        raise SystemExit("Run run_verification.py first (shared paired cases missing).")
    paired = json.load(open(paired_path))
    if any(p["ansys"]["status"] != "ok" or p["python"]["status"] != "ok" for p in paired):
        raise SystemExit("Paired cases contain failed evaluations; fix and rerun run_verification.py.")
    if not mock and any(p["ansys"]["model_source"] == "ansys_mock" for p in paired):
        raise SystemExit("Paired data came from the mock backend; rerun run_verification.py with the real ANSYS.")

    own_backend = backend is None
    startup_s = 0.0
    if own_backend:
        backend = ansys_backend.make_backend(mock)
        t0 = time.perf_counter()
        backend.start()
        startup_s = time.perf_counter() - t0
    beam_tools.set_backend(backend)
    beam_tools.self_test_python()

    llm = StubLLM() if stub_llm else make_llm()   # ONE model + settings for every workflow
    logger = ResultsLogger(C.path("beam_results.csv"), C.path("beam_llm_log.jsonl"))
    summaries = []
    for run in range(1, runs + 1):
        for wf in workflows:
            print(f"--- workflow {wf}, run {run}")
            s = WorkflowRun(wf, run, llm, logger, paired).run()
            print({k: s[k] for k in ("accepted_width_mm", "accepted_height_mm", "mass_kg", "confirmation_status",
                                     "ansys_requests", "new_ansys_solves", "invalid_replies", "total_time_s")})
            summaries.append(s)

    checks_df, check_s = final_checks.check_final_designs(summaries, logger)
    checks_df.to_csv(C.path("beam_final_checks.csv"), index=False)
    table = analysis.comparison_table(summaries)
    table.to_csv(C.path("beam_comparison.csv"), index=False)
    analysis.plot_mass_vs_time(summaries, paired, C.path("plot_mass_vs_time.png"))
    json.dump(summaries, open(C.path("beam_summaries.json"), "w"), indent=1, default=str)

    vc = {}
    if C.path("beam_verification_costs.json").exists():
        vc = json.load(open(C.path("beam_verification_costs.json")))
    search_total = sum(s["total_time_s"] for s in summaries)
    costs = {"startup_s_this_session": startup_s, "startup_s_verification_session": vc.get("startup_s"),
             "shared_verification_s": (vc.get("mesh_study_s") or 0) + (vc.get("paired_s") or 0),
             "final_mesh_checks_s": check_s, "search_and_confirmation_s": search_total,
             "project_total_s": startup_s + (vc.get("startup_s") or 0) + (vc.get("mesh_study_s") or 0)
                                + (vc.get("paired_s") or 0) + check_s + search_total,
             "llm_provider": C.LLM_PROVIDER, "llm_model": getattr(llm, "model", "stub"),
             "temperature": C.LLM_TEMPERATURE, "effort": C.LLM_EFFORT, "max_tokens": C.LLM_MAX_TOKENS, "mock_backend": mock, "stub_llm": stub_llm}
    json.dump(costs, open(C.path("beam_costs.json"), "w"), indent=1)
    logger.close()
    if own_backend:
        backend.close()
    print(table.to_string())
    print(checks_df.to_string())
    print(costs)
    return summaries, checks_df, table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workflows", default="ABC")
    ap.add_argument("--mock", action="store_true", help="MockBackend (testing only)")
    ap.add_argument("--stub-llm", action="store_true", help="StubLLM (testing only)")
    a = ap.parse_args()
    run_experiment(a.mock, a.stub_llm, a.runs, a.workflows)
