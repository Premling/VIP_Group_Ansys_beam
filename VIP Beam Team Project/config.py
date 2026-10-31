"""Central configuration. Load, supports, mesh policy and limits stay fixed during the search."""
import os
from pathlib import Path

try:  # load .env BEFORE reading any settings below
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

TEAM_ID = "XX"  # <-- replace with your team identifier
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

# ---------------- Problem definition (fixed) ----------------
L_MM = 1000.0
P_N = 10_000.0
E_GPA = 200.0
RHO = 7850.0
NU = 0.30
LOAD_STRIP_MM = 10.0
B_BOUNDS = (20.0, 100.0)
H_BOUNDS = (40.0, 120.0)
DEFL_LIMIT_MM = 1.0
STRESS_LIMIT_MPA = 120.0
START_DESIGN = (50.0, 80.0)
EXPECTED_START = {"mass_kg": 31.4, "deflection_mm": 0.488, "nominal_stress_mpa": 46.9}
PAIRED_DESIGNS = [(50.0, 80.0), (20.0, 40.0), (20.0, 120.0), (100.0, 40.0), (100.0, 120.0)]

# ---------------- beam_model.py adapter (beam_model.py itself stays unchanged) ----------------
BEAM_INPUT_SCALE = 1.0   # 1.0 if evaluate_beam takes mm; 1e-3 if it takes metres
BEAM_MASS_SCALE = 1.0    # multiply returned mass to get kg
BEAM_DEFL_SCALE = 1.0    # multiply returned deflection to get mm
BEAM_STRESS_SCALE = 1.0  # multiply returned stress to get MPa

# ---------------- ANSYS meshes / versioning (used for cache key + backend) ----------------
MESH_LEVELS = ["coarse", "medium", "fine", "finest"]   # >= 3 successively refined meshes
SEARCH_MESH = "fine"
CHECK_MESH = "finest"                                  # finer mesh for final checks
MESH_SETTINGS = {                                      # <-- fill in your real settings
    "coarse": {"element_size_mm": 10.0},
    "medium": {"element_size_mm": 6.0},
    "fine": {"element_size_mm": 4.0},
    "finest": {"element_size_mm": 2.5},
}

# ---------------- ANSYS Workbench project (used by ansys_backend.AnsysBackend) ----------------
ANSYS_ARCHIVE = ROOT / "ansys" / "beam demo.wbpz"      # never modified; unpacked into ANSYS_WORK_DIR
ANSYS_WORK_DIR = ROOT / "ansys" / "_work"
ANSYS_VERSION = "261"                                  # 2026 R1
ANSYS_SHOW_GUI = False
# Workbench parameter IDs (Parameter Set in Workbench). None = not defined in the project yet.
ANSYS_PARAMS = {
    "width_mm": "P3",            # XYPlane.H1 (x direction)
    "height_mm": "P4",           # XYPlane.V2 (y direction = vertical)
    "deflection": "P5",          # Directional Deformation (Y) Minimum
    "peak_stress": "P6",         # Equivalent Stress Maximum (global peak; NOT the constraint stress)
    "nominal_stress": None,      # REQUIRED: equivalent stress probe at bottom-centre under the load
    "element_size": None,        # mesh element size parameter (needed for the mesh study)
    "reaction_left": None,       # optional: vertical reaction at the pin
    "reaction_right": None,      # optional: vertical reaction at the roller
    "mass": None,                # optional: geometry mass (else computed from b*h*L*rho)
}


def _archive_tag() -> str:
    import hashlib
    try:
        return hashlib.sha256(ANSYS_ARCHIVE.read_bytes()).hexdigest()[:8]
    except OSError:
        return "missing"


MODEL_VERSION = f"ansys-beam-{_archive_tag()}"          # changes automatically when the .wbpz changes
DEFL_CONV_TARGET_PCT = 2.0
STRESS_CONV_TARGET_PCT = 5.0

# ---------------- Search budgets ----------------
N_ITERATIONS = 10                 # includes iteration 1 = common starting design
ADAPTIVE_SEARCH_ANSYS_MAX = 5     # workflow C, during search
ADAPTIVE_TOTAL_ANSYS_MAX = 6      # workflow C, including final confirmation
TRIGGER_FRACTION = 0.10           # base: use ANSYS when predicted value is within 10% of a limit
TRIGGER_DISAGREEMENT_MULT = 2.0   # widen trigger to 2x worst observed Python-vs-ANSYS disagreement
TRIGGER_MAX = 0.50

# ---------------- LLM (identical for all workflows) ----------------
LLM_PROVIDER = os.getenv("BEAM_LLM_PROVIDER", "openai")   # "openai" (ChatGPT, default) or "anthropic"
LLM_MODEL = os.getenv("BEAM_LLM_MODEL") or ("gpt-5" if LLM_PROVIDER == "openai" else "claude-sonnet-5-5")
LLM_TEMPERATURE = None   # None = provider default. GPT-5 / current Claude models reject temperature - keep None
LLM_EFFORT = os.getenv("BEAM_LLM_EFFORT") or None   # reasoning effort (OpenAI: minimal|low|medium|high); None = model default
# Reasoning models think before answering and those tokens count against the output cap; a small cap
# (e.g. 500) truncates replies -> empty/invalid JSON that still consumes an iteration.
LLM_MAX_TOKENS = 16000
_KNOWN_PRICES = {  # USD per 1M (input, output) tokens - check current pricing; override via .env
    "gpt-5": (1.25, 10.0), "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50)}
_DEFAULT_PRICE = _KNOWN_PRICES.get(LLM_MODEL, (float("nan"), float("nan")))
PRICE_IN_PER_MTOK = float(os.getenv("LLM_PRICE_IN_PER_MTOK") or _DEFAULT_PRICE[0])    # USD per 1M input tokens
PRICE_OUT_PER_MTOK = float(os.getenv("LLM_PRICE_OUT_PER_MTOK") or _DEFAULT_PRICE[1])  # USD per 1M output tokens


def path(name: str) -> Path:
    """Output file path, e.g. path('beam_results.csv') -> outputs/teamXX_beam_results.csv"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"team{TEAM_ID}_{name}"
