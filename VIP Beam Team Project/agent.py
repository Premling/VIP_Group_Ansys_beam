"""Prompts, selection policy, LLM clients, and strict response validation (no silent repair)."""
from __future__ import annotations

import json
import math
import re
import time

import config as C

# ---------------------------------------------------------------- prompts
BASE_PROMPT = f"""You are the design agent in an optimization loop. Minimize the mass of a simply supported
rectangular steel beam.

Fixed problem:
- Span L = {C.L_MM:.0f} mm, central downward static load P = {C.P_N:.0f} N, self-weight excluded.
- Material: E = {C.E_GPA:.0f} GPa, density = {C.RHO:.0f} kg/m^3.
- Design variables: width b = {C.B_BOUNDS[0]:.0f}-{C.B_BOUNDS[1]:.0f} mm, vertical height h = {C.H_BOUNDS[0]:.0f}-{C.H_BOUNDS[1]:.0f} mm.
- Objective: minimize beam mass.
- Constraints: midspan vertical deflection magnitude <= {C.DEFL_LIMIT_MM} mm; nominal midspan bending-surface
  von Mises stress <= {C.STRESS_LIMIT_MPA:.0f} MPa.
- Linear elastic, small-deflection static analysis.

Each turn you receive the history of evaluated designs, each labelled with its model source (python or
ansys). Rows labelled PRIOR are shared prior evidence available before the search. Use ONLY the numbers
returned in the history; never invent or estimate results as if they were evaluations. Your reasoning about
trends is welcome, but your stated confidence is not a measured uncertainty.

Reply with ONLY a JSON object and no other text:
{{"width_mm": <number>, "height_mm": <number>, "tool": "<one allowed tool>", "reason": "<one short sentence>"}}
The tool must be one of the allowed tools stated in the policy below. Out-of-bounds dimensions, disallowed
tools, or malformed replies are rejected, logged, and still consume an iteration."""

POLICY_A = """POLICY (workflow A): Allowed tools: ansys only. Every candidate is evaluated by ANSYS; set
"tool" to "ansys". Your final nominated design is taken from ANSYS results."""

POLICY_B = """POLICY (workflow B): Allowed tools: python only. Every candidate is evaluated by the fast Python
model; set "tool" to "python". After the search ends, one ANSYS check of your nominated design is run by the
controller; you cannot request ANSYS."""


def policy_c(trigger: float, stats: dict, ansys_search_max: int) -> str:
    return f"""POLICY (workflow C, adaptive selection): Allowed tools: python, ansys, both.
- "both" runs python and ansys on the same geometry and counts as ONE ANSYS request.
- ANSYS budget during search: at most {ansys_search_max} requests (failed and cached requests count).
  One additional ANSYS request is reserved by the controller to confirm your nominated design.
- Use python for broad exploration when it is informative.
- Use ansys (or both) when (a) a predicted constraint value is within {trigger:.0%} of its limit
  (deflection >= {(1-trigger)*C.DEFL_LIMIT_MM:.3f} mm or stress >= {(1-trigger)*C.STRESS_LIMIT_MPA:.1f} MPa),
  (b) you propose a promising improvement you want verified, or (c) paired results indicate a discrepancy.
- Trigger rationale: base trigger {C.TRIGGER_FRACTION:.0%}, widened to {C.TRIGGER_DISAGREEMENT_MULT:.0f}x the
  worst observed Python-vs-ANSYS disagreement on the five shared paired designs (worst deflection
  difference {stats['max_defl_pct']:.1f}%, worst stress difference {stats['max_stress_pct']:.1f}%, relative to
  ANSYS). Rows flagged near_limit=True in the history meet this trigger.
- Save at least some ANSYS budget for designs you intend to nominate."""


ALLOWED_TOOLS = {"A": ("ansys",), "B": ("python",), "C": ("python", "ansys", "both")}


def paired_stats(paired: list[dict]) -> dict:
    d, s = [], []
    for p in paired:
        py, an = p["python"], p["ansys"]
        if py["status"] == "ok" and an["status"] == "ok":
            d.append(abs(py["deflection_mm"] - an["deflection_mm"]) / abs(an["deflection_mm"]) * 100)
            s.append(abs(py["nominal_stress_mpa"] - an["nominal_stress_mpa"]) / abs(an["nominal_stress_mpa"]) * 100)
    return {"max_defl_pct": max(d, default=0.0), "max_stress_pct": max(s, default=0.0)}


def effective_trigger(stats: dict) -> float:
    worst = max(stats["max_defl_pct"], stats["max_stress_pct"]) / 100.0
    return min(C.TRIGGER_MAX, max(C.TRIGGER_FRACTION, C.TRIGGER_DISAGREEMENT_MULT * worst))


def system_prompt(workflow: str, trigger: float, stats: dict) -> str:
    policy = {"A": POLICY_A, "B": POLICY_B}.get(workflow) or policy_c(trigger, stats, C.ADAPTIVE_SEARCH_ANSYS_MAX)
    return BASE_PROMPT + "\n\n" + policy


# ---------------------------------------------------------------- history formatting
def near_limit(res: dict, trigger: float) -> bool:
    return (res["deflection_mm"] >= (1 - trigger) * C.DEFL_LIMIT_MM
            or res["nominal_stress_mpa"] >= (1 - trigger) * C.STRESS_LIMIT_MPA)


def format_history(history: list[dict], show_near_limit: bool = False, trigger: float = 0.1) -> str:
    lines = []
    for e in history:
        label = e["label"]
        if e.get("invalid"):
            lines.append(f"{label} | INVALID RESPONSE ({e['invalid']}) - iteration consumed, nothing evaluated")
            continue
        for src in ("python", "ansys"):
            r = e["results"].get(src)
            if r is None:
                continue
            tag = " [PRIOR shared evidence]" if e.get("prior") else ""
            head = f"{label} | src={src} | b={r['width_mm']:.2f} h={r['height_mm']:.2f}"
            if r["status"] != "ok":
                lines.append(f"{head} | STATUS=ERROR ({r['error']}) - no feasibility decision{tag}")
                continue
            line = (f"{head} | mass={r['mass_kg']:.3f} kg | defl={r['deflection_mm']:.4f} mm "
                    f"(margin {r['deflection_margin_mm']:+.4f}) | stress={r['nominal_stress_mpa']:.2f} MPa "
                    f"(margin {r['stress_margin_mpa']:+.2f}) | feasible={r['feasible']}")
            if show_near_limit:
                line += f" | near_limit={near_limit(r, trigger)}"
            lines.append(line + tag)
    return "\n".join(lines) if lines else "(empty)"


def build_iteration_prompt(workflow, iteration, n_iter, history, ansys_remaining, trigger) -> str:
    allowed = ", ".join(ALLOWED_TOOLS[workflow])
    parts = [
        f"Iteration {iteration} of {n_iter}. Iteration 1 was the common starting design; you propose iterations 2-{n_iter}.",
        f"Allowed tools: {allowed}",
    ]
    if workflow == "C":
        parts.append(f"ANSYS requests remaining for search: {ansys_remaining}")
    parts += ["History:", format_history(history, show_near_limit=(workflow == "C"), trigger=trigger),
              "Propose the next design as the JSON object."]
    return "\n".join(parts)


def build_nomination_prompt(workflow, history, trigger) -> str:
    return "\n".join([
        "The search is finished. Nominate ONE previously evaluated candidate (a '#N' row; PRIOR rows cannot be nominated) "
        "as your final recommendation. It will then be confirmed with ANSYS.",
        "History:", format_history(history, show_near_limit=(workflow == "C"), trigger=trigger),
        'Reply with ONLY JSON: {"nominated_iteration": <integer N>, "reason": "<one short sentence>"}',
    ])


# ---------------------------------------------------------------- strict validation
def _extract_json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_proposal(text: str, allowed: tuple, ansys_remaining: int | None):
    """Return (proposal, None) or (None, error). Never repairs or clips values."""
    try:
        obj = _extract_json(text)
    except Exception as exc:  # noqa: BLE001
        return None, f"unparseable JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "reply is not a JSON object"
    for k in ("width_mm", "height_mm", "tool", "reason"):
        if k not in obj:
            return None, f"missing field '{k}'"
    b, h, tool, reason = obj["width_mm"], obj["height_mm"], obj["tool"], obj["reason"]
    if not (_num(b) and _num(h)):
        return None, "width_mm/height_mm must be finite numbers"
    if not (C.B_BOUNDS[0] <= b <= C.B_BOUNDS[1]):
        return None, f"width_mm {b} outside bounds {C.B_BOUNDS}"
    if not (C.H_BOUNDS[0] <= h <= C.H_BOUNDS[1]):
        return None, f"height_mm {h} outside bounds {C.H_BOUNDS}"
    if tool not in allowed:
        return None, f"tool '{tool}' not allowed (allowed: {allowed})"
    if tool in ("ansys", "both") and ansys_remaining is not None and ansys_remaining <= 0:
        return None, "ANSYS search budget exhausted"
    if not isinstance(reason, str) or not reason.strip():
        return None, "reason must be a non-empty string"
    return {"width_mm": float(b), "height_mm": float(h), "tool": tool, "reason": reason.strip()[:300]}, None


def parse_nomination(text: str, valid_iterations: set):
    try:
        obj = _extract_json(text)
        it, reason = obj["nominated_iteration"], obj.get("reason", "")
    except Exception as exc:  # noqa: BLE001
        return None, "", f"unparseable nomination: {exc}"
    if isinstance(it, bool) or not isinstance(it, int) or it not in valid_iterations:
        return None, "", f"nominated_iteration {it!r} is not a previously evaluated candidate"
    return it, str(reason)[:300], None


# ---------------------------------------------------------------- LLM clients
class AnthropicLLM:
    def __init__(self, model=C.LLM_MODEL, temperature=C.LLM_TEMPERATURE, max_tokens=C.LLM_MAX_TOKENS,
                 effort=C.LLM_EFFORT):
        import anthropic  # ANTHROPIC_API_KEY is loaded from .env by config.py (never commit .env)
        self.client = anthropic.Anthropic()
        self.model, self.temperature, self.max_tokens, self.effort = model, temperature, max_tokens, effort

    def complete(self, system: str, user: str) -> dict:
        t0 = time.perf_counter()
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        # No refusal `fallbacks`: the experiment must use ONE model throughout, so a refusal is
        # logged as an invalid reply instead of being silently answered by a different model.
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}], **kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        if r.stop_reason in ("max_tokens", "refusal"):
            text = text or f"<no usable reply: stop_reason={r.stop_reason}>"
        return {"text": text, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                "stop_reason": r.stop_reason, "elapsed_s": time.perf_counter() - t0}


class OpenAILLM:
    def __init__(self, model=C.LLM_MODEL, temperature=C.LLM_TEMPERATURE, max_tokens=C.LLM_MAX_TOKENS,
                 effort=C.LLM_EFFORT):
        import os
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not found. Put OPENAI_API_KEY=... in .env (and save the file).")
        from openai import OpenAI  # OPENAI_API_KEY is loaded from .env by config.py (never commit .env)
        self.client = OpenAI()
        self.model, self.temperature, self.max_tokens, self.effort = model, temperature, max_tokens, effort

    def complete(self, system: str, user: str) -> dict:
        t0 = time.perf_counter()
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        if self.effort:
            kw["reasoning_effort"] = self.effort
        r = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=self.max_tokens, **kw,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return {"text": r.choices[0].message.content or "", "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens, "stop_reason": r.choices[0].finish_reason,
                "elapsed_s": time.perf_counter() - t0}


def make_llm():
    return OpenAILLM() if C.LLM_PROVIDER == "openai" else AnthropicLLM()


class StubLLM:
    """Deterministic fake LLM for dry runs of the pipeline ONLY (not a valid experiment)."""
    model = "stub"
    PROPOSALS = [(40, 80), (30, 90), (25, 95), (20, 100), (20, 90), (20, 86), (20, 88), (20, 87), (20, 120)]

    def __init__(self):
        self.i = 0

    def complete(self, system, user):
        t0 = time.perf_counter()
        if "nominated_iteration" in user:
            best = None
            for m in re.finditer(r"^#(\d+) \| src=\w+ .*?mass=([\d.]+) kg.*?feasible=True", user, re.M):
                n, mass = int(m.group(1)), float(m.group(2))
                if best is None or mass < best[1]:
                    best = (n, mass)
            text = json.dumps({"nominated_iteration": best[0] if best else 1, "reason": "lowest feasible mass"})
        else:
            b, h = self.PROPOSALS[self.i % len(self.PROPOSALS)]
            self.i += 1
            allowed = re.search(r"Allowed tools: (.*)", user).group(1).split(", ")
            tool = allowed[0] if len(allowed) == 1 else ("ansys" if self.i % 2 == 0 else "python")
            text = json.dumps({"width_mm": b, "height_mm": h, "tool": tool, "reason": "stub proposal"})
        return {"text": text, "input_tokens": 0, "output_tokens": 0, "elapsed_s": time.perf_counter() - t0}
