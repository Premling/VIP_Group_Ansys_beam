"""Tool wrappers: evaluate_python / evaluate_ansys with common result format, caching, error handling."""
from __future__ import annotations

import json
import math
import time
import uuid

import config as C

_BACKEND = None


def set_backend(backend) -> None:
    global _BACKEND
    _BACKEND = backend


def get_backend():
    if _BACKEND is None:
        raise RuntimeError("No ANSYS backend set. Call beam_tools.set_backend(...) first.")
    return _BACKEND


# ---------------- cache ----------------
class EvalCache:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def get(self, key):
        return self._store.get(key)

    def put(self, key, result):
        self._store[key] = dict(result)

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)


def cache_key(source: str, w: float, h: float, mesh: str | None = None) -> str:
    return json.dumps(
        {
            "source": source,
            "w": round(float(w), 6),
            "h": round(float(h), 6),
            "version": C.MODEL_VERSION if source == "ansys" else "beam_model.py",
            "mesh": mesh,
            "mesh_settings": C.MESH_SETTINGS.get(mesh) if mesh else None,
            "load": {"P_N": C.P_N, "L_MM": C.L_MM, "strip_mm": C.LOAD_STRIP_MM},
            "material": {"E_GPa": C.E_GPA, "rho": C.RHO, "nu": C.NU},
        },
        sort_keys=True,
    )


# ---------------- result helpers ----------------
def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ok_result(source, w, h, mass, defl, stress, elapsed, mesh, version, extra=None):
    defl_margin = C.DEFL_LIMIT_MM - abs(defl)
    stress_margin = C.STRESS_LIMIT_MPA - stress
    res = {
        "width_mm": float(w), "height_mm": float(h), "model_source": source,
        "mass_kg": float(mass), "deflection_mm": abs(float(defl)), "nominal_stress_mpa": float(stress),
        "status": "ok", "feasible": bool(defl_margin >= 0 and stress_margin >= 0),
        "deflection_margin_mm": defl_margin, "stress_margin_mpa": stress_margin,
        "elapsed_s": elapsed, "solve_elapsed_s": elapsed, "cache_hit": False,
        "evaluation_id": _new_id(source), "mesh": mesh, "project_version": version, "error": None,
    }
    if extra:
        res.update(extra)
    return res


def _error_result(source, w, h, exc, elapsed, mesh, version):
    """Failed evaluation: no numbers, no feasibility decision."""
    return {
        "width_mm": float(w), "height_mm": float(h), "model_source": source,
        "mass_kg": None, "deflection_mm": None, "nominal_stress_mpa": None,
        "status": "error", "feasible": None, "deflection_margin_mm": None, "stress_margin_mpa": None,
        "elapsed_s": elapsed, "solve_elapsed_s": elapsed, "cache_hit": False,
        "evaluation_id": _new_id(source), "mesh": mesh, "project_version": version, "error": str(exc),
    }


def _as_cache_hit(stored: dict, t0: float) -> dict:
    res = dict(stored)
    res["cache_hit"] = True
    res["elapsed_s"] = time.perf_counter() - t0
    res["evaluation_id"] = _new_id(res["model_source"])
    return res


# ---------------- Python tool ----------------
def _run_beam_model(w_mm: float, h_mm: float):
    """Adapter around the supplied, unchanged beam_model.evaluate_beam. Edit ONLY here if its
    signature/return format differs (units are handled via config.BEAM_*_SCALE)."""
    from beam_model import evaluate_beam  # lazy import so import errors become error results

    raw = evaluate_beam(w_mm * C.BEAM_INPUT_SCALE, h_mm * C.BEAM_INPUT_SCALE)
    if isinstance(raw, dict):
        def pick(*names):
            for n in names:
                if n in raw:
                    return raw[n]
            raise KeyError(f"none of {names} in evaluate_beam result keys {list(raw)}")
        mass = pick("mass_kg", "mass")
        defl = pick("deflection_mm", "midspan_deflection_mm", "max_deflection_mm", "deflection", "delta_mm")
        stress = pick("nominal_stress_mpa", "max_von_mises_stress_MPa", "stress_mpa", "max_stress_mpa", "stress", "sigma_mpa")
    elif isinstance(raw, (tuple, list)) and len(raw) >= 3:
        mass, defl, stress = raw[0], raw[1], raw[2]
    else:
        raise ValueError(f"Unrecognised evaluate_beam return type {type(raw)}; adapt _run_beam_model")
    return mass * C.BEAM_MASS_SCALE, defl * C.BEAM_DEFL_SCALE, stress * C.BEAM_STRESS_SCALE


def evaluate_python(width_mm: float, height_mm: float, cache: EvalCache | None = None) -> dict:
    t0 = time.perf_counter()
    key = cache_key("python", width_mm, height_mm)
    if cache is not None and (hit := cache.get(key)) is not None:
        return _as_cache_hit(hit, t0)
    try:
        mass, defl, stress = _run_beam_model(width_mm, height_mm)
        vals = (mass, defl, stress)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite Python model output {vals}")
        res = _ok_result("python", width_mm, height_mm, mass, defl, stress,
                         time.perf_counter() - t0, "n/a", "beam_model.py")
    except Exception as exc:  # noqa: BLE001
        res = _error_result("python", width_mm, height_mm, exc, time.perf_counter() - t0, "n/a", "beam_model.py")
    if cache is not None and res["status"] == "ok":
        cache.put(key, res)
    return res


def self_test_python(tol=0.01) -> dict:
    """Check the wrapper reproduces the stated starting-design values (~31.4 kg, 0.488 mm, 46.9 MPa)."""
    res = evaluate_python(*C.START_DESIGN)
    if res["status"] != "ok":
        raise RuntimeError(f"Python tool failed: {res['error']}")
    for k, v in C.EXPECTED_START.items():
        if abs(res[k] - v) / v > tol:
            raise AssertionError(f"{k}={res[k]:.4f} differs from expected {v} by >{tol:.0%}; check units/adapter")
    return res


# ---------------- ANSYS tool ----------------
_REQUIRED_ANSYS_KEYS = ("deflection_mm", "nominal_stress_mpa")


def evaluate_ansys(width_mm: float, height_mm: float, mesh: str | None = None,
                   cache: EvalCache | None = None) -> dict:
    mesh = mesh or C.SEARCH_MESH
    t0 = time.perf_counter()
    key = cache_key("ansys", width_mm, height_mm, mesh)
    if cache is not None and (hit := cache.get(key)) is not None:
        return _as_cache_hit(hit, t0)
    backend = get_backend()
    source = "ansys_mock" if getattr(backend, "is_mock", False) else "ansys"
    try:
        out = backend.solve(width_mm, height_mm, mesh)  # integration lives in ansys_backend.py
        for k in _REQUIRED_ANSYS_KEYS:
            if k not in out or not isinstance(out[k], (int, float)) or not math.isfinite(out[k]):
                raise ValueError(f"backend returned invalid '{k}': {out.get(k)!r}")
        mass = out.get("mass_kg")
        if mass is None:  # fall back to geometry mass only if backend did not report volume-based mass
            mass = width_mm * height_mm * C.L_MM * 1e-9 * C.RHO
        extra = {k: out[k] for k in ("global_peak_stress_mpa", "reaction_left_n", "reaction_right_n",
                                     "mesh_info") if k in out}
        res = _ok_result(source, width_mm, height_mm, mass, out["deflection_mm"], out["nominal_stress_mpa"],
                         time.perf_counter() - t0, mesh, out.get("project_version", C.MODEL_VERSION), extra)
        res["solve_elapsed_s"] = out.get("solve_time_s", res["elapsed_s"])
    except Exception as exc:  # noqa: BLE001
        res = _error_result(source, width_mm, height_mm, exc, time.perf_counter() - t0, mesh, C.MODEL_VERSION)
    if cache is not None and res["status"] == "ok":
        cache.put(key, res)
    return res
