"""Tables and plots for the report."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C


def paired_table(paired: list[dict]) -> pd.DataFrame:
    rows = []
    for p in paired:
        py, an = p["python"], p["ansys"]
        r = {"width_mm": py["width_mm"], "height_mm": py["height_mm"]}
        if py["status"] == "ok" and an["status"] == "ok":
            r.update(
                mass_py=py["mass_kg"], mass_ansys=an["mass_kg"],
                defl_py=py["deflection_mm"], defl_ansys=an["deflection_mm"],
                defl_diff=py["deflection_mm"] - an["deflection_mm"],
                defl_abs_pct=abs(py["deflection_mm"] - an["deflection_mm"]) / abs(an["deflection_mm"]) * 100,
                stress_py=py["nominal_stress_mpa"], stress_ansys=an["nominal_stress_mpa"],
                stress_diff=py["nominal_stress_mpa"] - an["nominal_stress_mpa"],
                stress_abs_pct=abs(py["nominal_stress_mpa"] - an["nominal_stress_mpa"]) / abs(an["nominal_stress_mpa"]) * 100,
                feasible_py=py["feasible"], feasible_ansys=an["feasible"])
        rows.append(r)
    return pd.DataFrame(rows)


def plot_paired(df: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (x, y, title, lim, unit) in zip(axes, [
            ("defl_ansys", "defl_py", "Midspan deflection", C.DEFL_LIMIT_MM, "mm"),
            ("stress_ansys", "stress_py", "Nominal stress", C.STRESS_LIMIT_MPA, "MPa")]):
        ax.scatter(df[x], df[y])
        for _, r in df.iterrows():
            ax.annotate(f"({r.width_mm:.0f},{r.height_mm:.0f})", (r[x], r[y]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        top = max(df[x].max(), df[y].max()) * 1.05
        ax.plot([0, top], [0, top], "k--", lw=0.8, label="y = x")
        ax.axvline(lim, color="r", lw=0.6, ls=":")
        ax.axhline(lim, color="r", lw=0.6, ls=":", label="limit")
        ax.set(xlabel=f"ANSYS ({unit})", ylabel=f"Python ({unit})", title=title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def comparison_table(summaries: list[dict]) -> pd.DataFrame:
    cols = ["workflow", "run", "nominated_width_mm", "nominated_height_mm", "nomination_source",
            "accepted_width_mm", "accepted_height_mm", "accepted_source", "mass_kg", "ansys_deflection_mm",
            "ansys_nominal_stress_mpa", "deflection_margin_mm", "stress_margin_mpa", "confirmation_status",
            "verified_feasible", "mesh_check_acceptance", "borderline", "python_requests", "ansys_requests",
            "ansys_search_requests", "new_python_solves", "new_ansys_solves", "cache_hits", "failures",
            "invalid_replies", "llm_calls", "search_time_s", "confirm_time_s", "total_time_s",
            "llm_input_tokens", "llm_output_tokens", "api_cost_usd", "trigger_fraction"]
    df = pd.DataFrame([{k: s.get(k) for k in cols} for s in summaries])
    return df


def plot_mass_vs_time(summaries: list[dict], paired: list[dict], path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in summaries:
        pts = [(t["cum_time_s"], t["best_verified_mass"]) for t in s["trace"]]
        x = [p[0] for p in pts]
        y = [np.nan if p[1] is None else p[1] for p in pts]
        ax.step(x, y, where="post", marker="o", ms=3, label=f"{s['workflow']} run {s['run']}")
    prior = [p["ansys"]["mass_kg"] for p in paired if p["ansys"]["status"] == "ok" and p["ansys"]["feasible"]]
    if prior:
        ax.axhline(min(prior), color="gray", ls="--", lw=0.8,
                   label="best feasible among shared prior ANSYS cases (not new)")
    ax.set(xlabel="Cumulative search time (s) incl. LLM + final confirmation", ylabel="Best ANSYS-confirmed feasible mass (kg)",
           title="Verified best mass vs search time")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
