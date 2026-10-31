"""ANSYS backend interface.

`AnsysBackend` is the ONLY place the Python <-> ANSYS (PyWorkbench/PyAnsys) integration goes.
Everything else in this package calls backend.start() / backend.solve() / backend.close().
`MockBackend` lets you test the agent, budgets, caching and plots without ANSYS. Mock results are
labelled model_source='ansys_mock' and must never be used for the real submission.
"""
from __future__ import annotations

import atexit
import math
import re
import shutil
import time

import config as C

# Workbench journal (IronPython) run inside the session. $$name%%default%% is filled in by PyWorkbench.
# Any failure is caught and reported in the JSON, so the Python side can refuse the result.
_SOLVE_SCRIPT = r'''
import json
res = {"ok": False}
try:
    dp = Parameters.GetDesignPoint(Name="0")
    for pid, expr in json.loads(r"""$$inputs%%{}%%""").items():
        dp.SetParameterExpression(Parameter=Parameters.GetParameter(Name=pid), Expression=expr)
    Update()
    res["ok"] = True
except Exception as e:
    res["error"] = str(e)
try:
    res["uptodate"] = bool(Parameters.GetDesignPoint(Name="0").IsUpToDate)
except Exception as e:
    res["uptodate"] = False
res["values"] = dict((p.Name, str(p.Value)) for p in Parameters.GetAllParameters())
wb_script_result = json.dumps(res)
'''

_UNIT_SCALE = {  # to mm, MPa, N, kg
    "mm": 1.0, "m": 1e3, "cm": 10.0, "in": 25.4, "um": 1e-3,
    "MPa": 1.0, "Pa": 1e-6, "kPa": 1e-3, "GPa": 1e3, "N mm^-2": 1.0, "N m^-2": 1e-6,
    "N": 1.0, "kN": 1e3, "mN": 1e-3,
    "kg": 1.0, "t": 1e3, "tonne": 1e3, "g": 1e-3,
}


def _quantity(text: str) -> float:
    """'-0.0502 [mm]' -> -0.0502 (converted to mm / MPa / N / kg)."""
    m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*(?:\[(.*)\])?\s*", str(text))
    if not m:
        raise ValueError(f"cannot parse ANSYS quantity {text!r}")
    unit = (m.group(2) or "").strip()
    if unit and unit not in _UNIT_SCALE:
        raise ValueError(f"unknown ANSYS unit {unit!r} in {text!r}; add it to _UNIT_SCALE")
    return float(m.group(1)) * _UNIT_SCALE.get(unit, 1.0)


class AnsysBackend:
    """Persistent PyWorkbench session on the parameterized project in C.ANSYS_ARCHIVE."""
    is_mock = False

    def __init__(self):
        self.wb = None
        self.project = None
        self.params = C.ANSYS_PARAMS

    def _run(self, script: str, **args) -> dict:
        out = self.wb.run_script_string(script, args=args or None, log_level="error")
        if out is None:  # PyWorkbench returns None (and only logs) when the journal itself errors
            raise RuntimeError("Workbench script failed (see the PyWorkbench log above)")
        return out

    def start(self) -> None:
        """Launch Workbench, unpack a fresh copy of the archive and check the parameters exist."""
        from ansys.workbench.core import launch_workbench

        if not C.ANSYS_ARCHIVE.exists():
            raise FileNotFoundError(f"ANSYS archive not found: {C.ANSYS_ARCHIVE}")
        # fresh folder per session; old ones are removed unless a lingering Workbench still locks them
        for old in C.ANSYS_WORK_DIR.glob("session_*"):
            shutil.rmtree(old, ignore_errors=True)
        self.work = work = C.ANSYS_WORK_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        work.mkdir(parents=True)
        self.project = work / "beam.wbpj"
        self.wb = launch_workbench(show_gui=C.ANSYS_SHOW_GUI, version=C.ANSYS_VERSION,
                                   client_workdir=str(work), server_workdir=str(work))
        atexit.register(self.close)  # don't leave a headless Workbench running if Python stops early
        found = self._run(r'''
import json
Unarchive(ArchivePath=r"$$arch%%%%", ProjectPath=r"$$proj%%%%", Overwrite=True)
wb_script_result = json.dumps(dict((p.Name, p.DisplayText) for p in Parameters.GetAllParameters()))
''', arch=C.ANSYS_ARCHIVE.as_posix(), proj=self.project.as_posix())
        missing = {k: v for k, v in self.params.items() if v and v not in found}
        if missing:
            raise RuntimeError(f"Parameters {missing} not in project (project has {found}); fix config.ANSYS_PARAMS")
        self.available = found

    def solve(self, width_mm: float, height_mm: float, mesh: str) -> dict:
        """Set b/h (+ mesh size), update geometry/mesh/solution, confirm success, read outputs.
        Raises on any failed or stale update, so evaluate_ansys returns an error and never old numbers."""
        p = self.params
        if p["nominal_stress"] is None:
            raise RuntimeError("No nominal-stress output parameter: add a bottom-centre equivalent-stress probe "
                               "in Mechanical, tick its P box and set ANSYS_PARAMS['nominal_stress'] in config.py")
        inputs = {p["width_mm"]: f"{width_mm!r} [mm]", p["height_mm"]: f"{height_mm!r} [mm]"}
        if p["element_size"]:
            inputs[p["element_size"]] = f"{C.MESH_SETTINGS[mesh]['element_size_mm']!r} [mm]"
        elif mesh != C.SEARCH_MESH:
            raise RuntimeError(f"Mesh level '{mesh}' needs an element-size parameter (ANSYS_PARAMS['element_size'])")

        import json
        t0 = time.perf_counter()
        res = self._run(_SOLVE_SCRIPT, inputs=json.dumps(inputs))
        solve_s = time.perf_counter() - t0
        if not res["ok"]:
            raise RuntimeError(f"ANSYS update failed: {res.get('error')}")
        if not res["uptodate"]:
            raise RuntimeError("ANSYS design point is not up to date after Update(); refusing stale results")
        vals = res["values"]
        for key, want in (("width_mm", width_mm), ("height_mm", height_mm)):
            got = _quantity(vals[p[key]])
            if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"ANSYS {key} is {got}, requested {want}; refusing result")

        def out(key):
            return _quantity(vals[p[key]]) if p[key] else None

        result = {
            "deflection_mm": abs(out("deflection")),
            "nominal_stress_mpa": out("nominal_stress"),
            "global_peak_stress_mpa": out("peak_stress"),
            "project_version": C.MODEL_VERSION,
            "solve_time_s": solve_s,
            "mesh_info": self._mesh_info(),
        }
        if p["mass"]:
            result["mass_kg"] = out("mass")
        if p["reaction_left"] and p["reaction_right"]:
            result["reaction_left_n"], result["reaction_right_n"] = abs(out("reaction_left")), abs(out("reaction_right"))
        return result

    def _mesh_info(self) -> dict:
        """Element type/count from the solver output of the solve that just ran."""
        info = {"element_type": None, "n_elements": None, "n_nodes": None}
        try:
            txt = next(self.work.glob("*_files/dp0/SYS*/MECH/solve.out")).read_text(errors="ignore")
        except (StopIteration, OSError):
            return info
        if m := re.findall(r"ELEMENT TYPE\s+1 IS (\w+)", txt):
            info["element_type"] = m[-1]
        if m := re.findall(r"Number of solid elements = (\d+)", txt):
            info["n_elements"] = int(m[-1])
        if m := re.findall(r"Number of total nodes = (\d+)", txt):
            info["n_nodes"] = int(m[-1])
        return info

    def close(self) -> None:
        if self.wb is not None:
            wb, self.wb = self.wb, None
            wb.exit()


class MockBackend:
    """Timoshenko-corrected analytic stand-in for ANSYS (testing only)."""
    is_mock = True

    def start(self) -> None:
        time.sleep(0.05)

    def solve(self, width_mm, height_mm, mesh):
        t0 = time.perf_counter()
        b, h, L, P = width_mm, height_mm, C.L_MM, C.P_N
        E = C.E_GPA * 1e3
        G = E / (2 * (1 + C.NU))
        I = b * h**3 / 12
        defl = P * L**3 / (48 * E * I) + P * L / (4 * (5 / 6) * G * b * h)  # bending + shear
        stress = (P * L / 4) * (h / 2) / I * 1.02
        k = C.MESH_LEVELS.index(mesh) + 1
        defl *= 1 - 0.04 / k**2          # fake mesh dependence that converges
        stress *= 1 - 0.08 / k**2
        time.sleep(0.02 * k)
        return {
            "deflection_mm": defl, "nominal_stress_mpa": stress,
            "mass_kg": b * h * L * 1e-9 * C.RHO, "global_peak_stress_mpa": stress * 1.5,
            "reaction_left_n": P / 2, "reaction_right_n": P / 2,
            "mesh_info": {"element_type": "MOCK", "n_elements": 1000 * k**3},
            "project_version": "mock", "solve_time_s": time.perf_counter() - t0,
        }

    def close(self) -> None:
        pass


def make_backend(mock: bool = False):
    return MockBackend() if mock else AnsysBackend()
