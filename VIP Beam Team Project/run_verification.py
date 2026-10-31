"""Milestone 1/2: mesh convergence at the starting design + five paired Python-vs-ANSYS cases.
Usage: python run_verification.py [--mock]"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import analysis
import ansys_backend
import beam_tools
import config as C
from logging_utils import ResultsLogger


GEOMETRY_DEMO = [(50.0, 80.0), (40.0, 100.0), (60.0, 70.0)]  # start + two successive geometry changes


def demo_successive_geometry(logger=None) -> pd.DataFrame:
    """Task 2: change geometry twice in the same ANSYS session (no cache) and show the outputs change."""
    rows = []
    for i, (w, h) in enumerate(GEOMETRY_DEMO, 1):
        for src, fn in (("python", beam_tools.evaluate_python), ("ansys", beam_tools.evaluate_ansys)):
            r = fn(w, h)
            if logger:
                logger.log_tool("verification", 0, 0, len(rows) + 1, "geometry_demo", src, f"geometry change {i - 1}", r)
            rows.append({k: r.get(k) for k in ("model_source", "width_mm", "height_mm", "mass_kg", "deflection_mm",
                                               "nominal_stress_mpa", "status", "error", "elapsed_s", "evaluation_id")})
    df = pd.DataFrame(rows)
    an = df[df.model_source != "python"]
    ok = (an.status == "ok").all() and an[["deflection_mm", "nominal_stress_mpa"]].round(9).drop_duplicates().shape[0] == len(an)
    df.attrs["ansys_results_changed"] = bool(ok)
    print(f"Successive geometry demo: ANSYS results {'changed with every geometry' if ok else 'DID NOT all change / failed'}")
    return df


def run_verification(mock=False, logger=None, backend=None):
    """Pass an already-started `backend` to reuse one persistent ANSYS session (it is then not closed here)."""
    own_backend = backend is None
    startup_s = 0.0
    if own_backend:
        backend = ansys_backend.make_backend(mock)
        t0 = time.perf_counter()
        backend.start()
        startup_s = time.perf_counter() - t0
    beam_tools.set_backend(backend)
    beam_tools.self_test_python()

    # ---- mesh convergence (>= 3 successively refined meshes) at the starting design
    t0 = time.perf_counter()
    rows = []
    for mesh in C.MESH_LEVELS:
        r = beam_tools.evaluate_ansys(*C.START_DESIGN, mesh=mesh)
        if logger:
            logger.log_tool("verification", 0, 0, len(rows) + 1, "mesh_study", "ansys", "mesh convergence", r)
        info = r.get("mesh_info", {}) or {}
        rows.append({"mesh": mesh, "element_type": info.get("element_type"), "n_elements": info.get("n_elements"),
                     "deflection_mm": r["deflection_mm"], "nominal_stress_mpa": r["nominal_stress_mpa"],
                     "global_peak_stress_mpa": r.get("global_peak_stress_mpa"),
                     "reaction_left_n": r.get("reaction_left_n"), "reaction_right_n": r.get("reaction_right_n"),
                     "solve_time_s": r["solve_elapsed_s"], "status": r["status"]})
    mesh_df = pd.DataFrame(rows)
    if len(mesh_df) >= 2 and (mesh_df.status == "ok").all():
        a, b = mesh_df.iloc[-2], mesh_df.iloc[-1]   # two finest, change relative to finest
        mesh_df.attrs["defl_change_pct"] = abs(b.deflection_mm - a.deflection_mm) / abs(b.deflection_mm) * 100
        mesh_df.attrs["stress_change_pct"] = abs(b.nominal_stress_mpa - a.nominal_stress_mpa) / abs(b.nominal_stress_mpa) * 100
        print(f"Two finest meshes: deflection change {mesh_df.attrs['defl_change_pct']:.2f}% "
              f"(target <{C.DEFL_CONV_TARGET_PCT}%), stress change {mesh_df.attrs['stress_change_pct']:.2f}% "
              f"(target <{C.STRESS_CONV_TARGET_PCT}%)")
    mesh_df.to_csv(C.path("beam_mesh_study.csv"), index=False)
    mesh_s = time.perf_counter() - t0

    # ---- Task 2: two successive geometry changes return different results
    demo_df = demo_successive_geometry(logger)
    demo_df.to_csv(C.path("beam_geometry_demo.csv"), index=False)

    # ---- five paired cases on the search mesh
    t0 = time.perf_counter()
    paired = []
    for i, (w, h) in enumerate(C.PAIRED_DESIGNS, 1):
        py = beam_tools.evaluate_python(w, h)
        an = beam_tools.evaluate_ansys(w, h, mesh=C.SEARCH_MESH)
        if logger:
            logger.log_tool("verification", 0, 0, 2 * i - 1, "paired", "python", "paired case", py)
            logger.log_tool("verification", 0, 0, 2 * i, "paired", "ansys", "paired case", an)
        paired.append({"python": py, "ansys": an})
    paired_s = time.perf_counter() - t0
    json.dump(paired, open(C.path("beam_paired.json"), "w"), indent=1, default=str)
    df = analysis.paired_table(paired)
    df.to_csv(C.path("beam_paired.csv"), index=False)
    analysis.plot_paired(df, C.path("plot_paired.png"))
    json.dump({"startup_s": startup_s, "mesh_study_s": mesh_s, "paired_s": paired_s,
               "mock": mock}, open(C.path("beam_verification_costs.json"), "w"), indent=1)
    print(df.to_string())
    if own_backend:
        backend.close()
    return mesh_df, paired, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use MockBackend (testing only)")
    args = ap.parse_args()
    lg = ResultsLogger(C.path("beam_results.csv"), C.path("beam_llm_log.jsonl"))
    run_verification(args.mock, lg)
    lg.close()
