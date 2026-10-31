"""Controller: runs workflows A/B/C, enforces tools/bounds/budgets in Python, logs everything."""
from __future__ import annotations

import math
from collections import Counter

import agent
import config as C
from beam_tools import EvalCache, cache_key, evaluate_ansys, evaluate_python

WORKFLOW_LABELS = {"A": "ANSYS throughout", "B": "Python search + 1 ANSYS check", "C": "Adaptive selection"}
INITIAL_TOOL = {"A": "ansys", "B": "python", "C": "python"}  # iteration 1 = common start (cache hit from prior data)
PRIMARY_SOURCE = {"A": "ansys", "B": "python", "C": "ansys"}


class WorkflowRun:
    def __init__(self, name: str, run_id: int, llm, logger, paired: list[dict]):
        self.name, self.run_id, self.llm, self.logger, self.paired = name, run_id, llm, logger, paired
        self.cache = EvalCache()                       # fresh cache per workflow
        self.history: list[dict] = []                  # fresh history per workflow
        self.n = Counter()
        self.request_no = 0
        self.ansys_search_used = 0
        self.t_search = 0.0                            # tool + LLM seconds during search
        self.t_confirm = 0.0                           # nomination LLM + final confirmation seconds
        self.in_tok = self.out_tok = 0
        self.trace: list[dict] = []
        self.stats = agent.paired_stats(paired)
        self.trigger = agent.effective_trigger(self.stats)
        self.system = agent.system_prompt(name, self.trigger, self.stats)

    # ------------------------------------------------------------ setup
    def _preload(self):
        for i, pr in enumerate(self.paired, 1):
            entry = {"label": f"PRIOR-{i}", "iteration": 0, "prior": True, "invalid": None, "results": {}}
            for src in ("python", "ansys"):
                res = dict(pr[src])
                res["prior_evidence"] = True
                mesh = C.SEARCH_MESH if src == "ansys" else None
                self.cache.put(cache_key("ansys" if src == "ansys" else "python", res["width_mm"], res["height_mm"], mesh), res)
                entry["results"][src] = res
                self.logger.log_tool(self.name, self.run_id, 0, 0, "prior", src, "shared prior evidence", res)
            self.history.append(entry)

    # ------------------------------------------------------------ helpers
    def _search_entries(self):
        return [e for e in self.history if not e["prior"] and not e["invalid"]]

    def _count_result(self, src, res, phase):
        if res["status"] != "ok":
            self.n["failures"] += 1
        elif res["cache_hit"]:
            self.n["cache_hits"] += 1
        else:
            self.n[f"new_{src}_solves"] += 1
        if phase == "search":
            self.t_search += res["elapsed_s"]
        else:
            self.t_confirm += res["elapsed_s"]

    def _execute(self, it, w, h, tool, reason, phase="search", llm_dt=0.0):
        results = {}
        for src in (["python", "ansys"] if tool == "both" else [tool]):
            if src == "python":
                res = evaluate_python(w, h, cache=self.cache)
                self.n["python_requests"] += 1
            else:
                res = evaluate_ansys(w, h, cache=self.cache)
                self.n["ansys_requests"] += 1
                if phase == "search":
                    self.ansys_search_used += 1
            self.request_no += 1
            self._count_result(src, res, phase)
            self.logger.log_tool(self.name, self.run_id, it, self.request_no,
                                 "search" if phase == "search" else "final_confirm",
                                 tool, reason, res, llm_dt if not results else 0.0)
            results[src] = res
        return results

    def _llm_call(self, user_prompt, kind, it):
        out = self.llm.complete(self.system, user_prompt)
        self.n["llm_calls"] += 1
        self.in_tok += out["input_tokens"]
        self.out_tok += out["output_tokens"]
        self.logger.log_llm({"workflow": self.name, "run": self.run_id, "iteration": it, "kind": kind,
                             "system": self.system, "user": user_prompt, "response": out["text"],
                             "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
                             "stop_reason": out.get("stop_reason"), "elapsed_s": out["elapsed_s"]})
        return out

    def _best_verified_mass(self):
        masses = [e["results"]["ansys"]["mass_kg"] for e in self._search_entries()
                  if "ansys" in e["results"] and e["results"]["ansys"]["status"] == "ok"
                  and e["results"]["ansys"]["feasible"]]
        return min(masses) if masses else None

    def _record_invalid(self, it, out, err, requested_tool=None):
        self.n["invalid_replies"] += 1
        self.history.append({"label": f"#{it}", "iteration": it, "prior": False, "invalid": err, "results": {}})
        self.logger.log_invalid(self.name, self.run_id, it, requested_tool, out["text"][:300] if out else "",
                                err, out["elapsed_s"] if out else 0.0)

    def _log_bad_nomination(self, out, err):
        self.n["invalid_replies"] += 1
        self.logger.log_invalid(self.name, self.run_id, C.N_ITERATIONS, None, out["text"][:300] if out else "",
                                err, out["elapsed_s"] if out else 0.0, row_type="invalid_nomination")

    # ------------------------------------------------------------ main loop
    def run(self) -> dict:
        self._preload()
        for it in range(1, C.N_ITERATIONS + 1):
            if it == 1:
                prop = {"width_mm": C.START_DESIGN[0], "height_mm": C.START_DESIGN[1],
                        "tool": INITIAL_TOOL[self.name], "reason": "common starting design"}
                llm_dt = 0.0
            else:
                remaining = (C.ADAPTIVE_SEARCH_ANSYS_MAX - self.ansys_search_used) if self.name == "C" else None
                prompt = agent.build_iteration_prompt(self.name, it, C.N_ITERATIONS, self.history, remaining, self.trigger)
                try:
                    out = self._llm_call(prompt, "proposal", it)
                except Exception as exc:  # noqa: BLE001  (API failure consumes the iteration)
                    self._record_invalid(it, None, f"llm_error: {exc}")
                    self.trace.append(self._trace_point(it))
                    continue
                llm_dt = out["elapsed_s"]
                self.t_search += llm_dt
                prop, err = agent.parse_proposal(out["text"], agent.ALLOWED_TOOLS[self.name], remaining)
                if err:
                    self._record_invalid(it, out, err)
                    self.trace.append(self._trace_point(it))
                    continue
            results = self._execute(it, prop["width_mm"], prop["height_mm"], prop["tool"], prop["reason"], "search", llm_dt)
            self.history.append({"label": f"#{it}", "iteration": it, "prior": False, "invalid": None,
                                 "tool": prop["tool"], "reason": prop["reason"], "results": results,
                                 "width_mm": prop["width_mm"], "height_mm": prop["height_mm"]})
            self.trace.append(self._trace_point(it))
        return self._finalize()

    def _trace_point(self, it):
        return {"iteration": it, "cum_time_s": self.t_search + self.t_confirm,
                "best_verified_mass": self._best_verified_mass()}

    # ------------------------------------------------------------ nomination + confirmation
    def _controller_nominee(self):
        src = PRIMARY_SOURCE[self.name]
        best = None
        for e in self._search_entries():
            r = e["results"].get(src)
            if r and r["status"] == "ok" and r["feasible"] and (best is None or r["mass_kg"] < best[1]):
                best = (e, r["mass_kg"])
        if best is None and src == "ansys":  # C: fall back to python-feasible candidate
            for e in self._search_entries():
                r = e["results"].get("python")
                if r and r["status"] == "ok" and r["feasible"] and (best is None or r["mass_kg"] < best[1]):
                    best = (e, r["mass_kg"])
        return best[0] if best else None

    def _nominate(self):
        entries = self._search_entries()
        if not entries:
            return None, "none"
        valid = {e["iteration"] for e in entries}
        nom_it = None
        try:
            out = self._llm_call(agent.build_nomination_prompt(self.name, self.history, self.trigger), "nomination", C.N_ITERATIONS)
            self.t_confirm += out["elapsed_s"]
            nom_it, _, err = agent.parse_nomination(out["text"], valid)
            if err:
                self._log_bad_nomination(out, err)
        except Exception as exc:  # noqa: BLE001
            self._log_bad_nomination(None, f"llm_error(nomination): {exc}")
        if nom_it is not None:
            return next(e for e in entries if e["iteration"] == nom_it), "llm"
        return self._controller_nominee(), "controller_fallback"

    def _finalize(self) -> dict:
        nominee, nom_source = self._nominate()
        s = {"workflow": self.name, "run": self.run_id, "label": WORKFLOW_LABELS[self.name],
             "nominated_width_mm": None, "nominated_height_mm": None, "nomination_source": nom_source,
             "accepted_width_mm": None, "accepted_height_mm": None, "accepted_source": None,
             "mass_kg": None, "ansys_deflection_mm": None, "ansys_nominal_stress_mpa": None,
             "deflection_margin_mm": None, "stress_margin_mpa": None,
             "confirmation_status": "no_candidate", "verified_feasible": False}
        accepted_res = None
        if nominee is not None:
            w, h = nominee["width_mm"], nominee["height_mm"]
            s.update(nominated_width_mm=w, nominated_height_mm=h)
            conf = self._execute(C.N_ITERATIONS, w, h, "ansys", "final confirmation", "confirm")["ansys"]
            if self.name == "C":
                assert self.n["ansys_requests"] <= C.ADAPTIVE_TOTAL_ANSYS_MAX, "ANSYS budget exceeded"
            if conf["status"] != "ok":
                s["confirmation_status"] = "confirmation_error"
            elif conf["feasible"]:
                s["confirmation_status"], s["accepted_source"], accepted_res = "confirmed_feasible", "nominated", conf
            else:
                s["confirmation_status"] = "failed_constraint"
                fb = [(e["results"]["ansys"]["mass_kg"], e) for e in self._search_entries()
                      if e["iteration"] != nominee["iteration"] and "ansys" in e["results"]
                      and e["results"]["ansys"]["status"] == "ok" and e["results"]["ansys"]["feasible"]]
                if fb:
                    _, e = min(fb, key=lambda t: t[0])
                    s["accepted_source"], accepted_res = "fallback_prior_ansys_confirmed", e["results"]["ansys"]
            s["nominated_ansys_feasible"] = conf.get("feasible")
        if accepted_res is not None:
            s.update(accepted_width_mm=accepted_res["width_mm"], accepted_height_mm=accepted_res["height_mm"],
                     mass_kg=accepted_res["mass_kg"], ansys_deflection_mm=accepted_res["deflection_mm"],
                     ansys_nominal_stress_mpa=accepted_res["nominal_stress_mpa"],
                     deflection_margin_mm=accepted_res["deflection_margin_mm"],
                     stress_margin_mpa=accepted_res["stress_margin_mpa"], verified_feasible=True)
            # verified best mass is plotted only once a feasible ANSYS result exists; it is a running
            # best, so a heavier nominee must not raise the curve above an earlier verified point
            prev = self._best_verified_mass()
            best = accepted_res["mass_kg"] if prev is None else min(prev, accepted_res["mass_kg"])
            self.trace.append({"iteration": "final", "cum_time_s": self.t_search + self.t_confirm,
                               "best_verified_mass": best})
        else:
            s["verdict"] = "no verified feasible design"
        price_ok = math.isfinite(C.PRICE_IN_PER_MTOK) and math.isfinite(C.PRICE_OUT_PER_MTOK)
        s.update({k: self.n[k] for k in ("python_requests", "ansys_requests", "new_python_solves",
                                         "new_ansys_solves", "cache_hits", "failures", "invalid_replies",
                                         "llm_calls")})
        s.update(ansys_search_requests=self.ansys_search_used, search_time_s=self.t_search,
                 confirm_time_s=self.t_confirm, total_time_s=self.t_search + self.t_confirm,
                 llm_input_tokens=self.in_tok, llm_output_tokens=self.out_tok,
                 api_cost_usd=(self.in_tok * C.PRICE_IN_PER_MTOK + self.out_tok * C.PRICE_OUT_PER_MTOK) / 1e6
                 if price_ok else None, trigger_fraction=self.trigger, trace=self.trace)
        return s
