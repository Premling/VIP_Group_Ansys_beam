"""Task 4 'Check the final designs': search mesh vs finer mesh for each distinct accepted geometry."""
from __future__ import annotations

import time

import pandas as pd

import config as C
from beam_tools import evaluate_ansys


def check_final_designs(summaries: list[dict], logger=None) -> tuple[pd.DataFrame, float]:
    owners: dict[tuple, list] = {}
    for s in summaries:
        if s["accepted_width_mm"] is not None:
            owners.setdefault((s["accepted_width_mm"], s["accepted_height_mm"]), []).append(f"{s['workflow']}{s['run']}")
    rows, t0 = [], time.perf_counter()
    for i, ((w, h), who) in enumerate(owners.items(), 1):
        a = evaluate_ansys(w, h, mesh=C.SEARCH_MESH)   # no cache: these are real, reported solves
        b = evaluate_ansys(w, h, mesh=C.CHECK_MESH)
        for res in (a, b):
            if logger:
                logger.log_tool("final_check", 0, 0, i, "mesh_check", "ansys", f"accepted by {','.join(who)}", res)
        row = {"width_mm": w, "height_mm": h, "accepted_by": ",".join(who),
               "search_mesh": C.SEARCH_MESH, "check_mesh": C.CHECK_MESH}
        if a["status"] != "ok" or b["status"] != "ok":
            row.update(status="error", acceptance="UNVERIFIED (mesh check failed)")
        else:
            dd = abs(b["deflection_mm"] - a["deflection_mm"])
            ds = abs(b["nominal_stress_mpa"] - a["nominal_stress_mpa"])
            row.update(
                defl_search=a["deflection_mm"], defl_fine=b["deflection_mm"],
                defl_change_pct=dd / abs(b["deflection_mm"]) * 100,
                stress_search=a["nominal_stress_mpa"], stress_fine=b["nominal_stress_mpa"],
                stress_change_pct=ds / abs(b["nominal_stress_mpa"]) * 100,
                feasible_search=a["feasible"], feasible_fine=b["feasible"],
                defl_margin_fine=b["deflection_margin_mm"], stress_margin_fine=b["stress_margin_mpa"],
                solve_time_s=a["elapsed_s"] + b["elapsed_s"], status="ok")
            row["defl_target_met"] = row["defl_change_pct"] < C.DEFL_CONV_TARGET_PCT
            row["stress_target_met"] = row["stress_change_pct"] < C.STRESS_CONV_TARGET_PCT
            row["constraint_flipped"] = a["feasible"] != b["feasible"]
            # borderline: remaining margin is smaller than the change caused by mesh refinement
            row["borderline"] = bool(row["constraint_flipped"]
                                     or abs(b["deflection_margin_mm"]) <= dd
                                     or abs(b["stress_margin_mpa"]) <= ds)
            row["acceptance"] = "accepted" if b["feasible"] else "REVOKED: infeasible on finer mesh"
        rows.append(row)
    df = pd.DataFrame(rows)
    for s in summaries:  # revise acceptance status
        key = (s["accepted_width_mm"], s["accepted_height_mm"])
        if key[0] is None or df.empty:
            continue
        r = df[(df.width_mm == key[0]) & (df.height_mm == key[1])].iloc[0]
        s["mesh_check_acceptance"] = r["acceptance"]
        # failed-check rows have borderline=NaN, and bool(NaN) is True
        s["borderline"] = bool(r["borderline"]) if r["status"] == "ok" else False
        if r["acceptance"] != "accepted":  # REVOKED, or UNVERIFIED because the mesh check failed
            s["verified_feasible"] = False
    return df, time.perf_counter() - t0
