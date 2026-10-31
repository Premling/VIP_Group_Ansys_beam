# Team XX – Beam design with ANSYS and AI agents

## Setup (VS Code terminal)
```
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Create `.env` (never commit it):
```
OPENAI_API_KEY=your_key_here
# optional: model (default gpt-5) and reasoning effort (minimal|low|medium|high; default = model default)
# BEAM_LLM_MODEL=gpt-5
# BEAM_LLM_EFFORT=low
# optional, for API cost reporting (USD per 1M tokens; gpt-5 defaults to 1.25 / 10.00):
LLM_PRICE_IN_PER_MTOK=
LLM_PRICE_OUT_PER_MTOK=
```
Place the unchanged `beam_model.py` in this folder. Set `TEAM_ID` and the real mesh settings in `config.py`.
Put the Workbench archive at `ansys/beam demo.wbpz` and map its parameter IDs in `config.ANSYS_PARAMS`.

## Run order
1. `python run_verification.py` – mesh study + five paired cases (+ paired plot)
2. `python run_experiment.py --runs 1` – workflows A/B/C, final mesh checks, tables, plots
Testing without ANSYS/API: add `--mock` (and `--stub-llm` to run_experiment). Mock output is not valid for submission;
delete `outputs/` before the real run.

## Files
- `config.py` – problem constants, bounds, meshes, budgets, LLM settings
- `beam_tools.py` – `evaluate_python`, `evaluate_ansys`, cache, common result format
- `ansys_backend.py` – ANSYS integration point + mock
- `agent.py` – prompts, selection policy, strict validation, LLM client
- `workflows.py` – workflows A/B/C, budgets, nomination, confirmation, fallback
- `final_checks.py` – finer-mesh checks of accepted designs
- `analysis.py`, `run_verification.py`, `run_experiment.py`, `teamXX_beam_agent.ipynb`

## Record here before submitting
AI model: ___ (OpenAI model from .env, default gpt-5; provider-default temperature, max_completion_tokens 16000) | ANSYS version: 2026 R1 Student (v261) | PyAnsys: ansys-workbench-core 0.14.0
Mesh settings: see `config.MESH_SETTINGS` | Prompts: `agent.py` and `outputs/teamXX_beam_llm_log.jsonl`
ANSYS archive location: ___ | Individual contributions: ___
