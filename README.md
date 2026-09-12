# DECIVERSE Decision Agent

## AI-Powered Social Decision Simulator
### 🚀 Tech Zephyr 4.0 — IIT Bhubaneswar Agentic AI Hackathon

> **DECIVERSE Decision Agent is an autonomous decision-making agent that uses a simulated society as its environment.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast%20python-purple.svg)](https://docs.astral.sh/uv/)
[![Test Suite](https://img.shields.io/badge/Tests-52%20Passed-brightgreen.svg)](tests/)
[![Architecture: Closed Loop](https://img.shields.io/badge/Agent-Autonomous%20Closed%20Loop-orange.svg)](#architecture)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-lightgrey.svg)](LICENSE)

---

### ⚡ 60-Second Judge Overview

| Question | Answer |
| :--- | :--- |
| **What is the problem?** | Decision-makers cannot predict emergent polarization and public opposition before deploying complex policies (transit fares, zoning, clean energy). |
| **What did we build?** | An autonomous closed-loop agent that formulates policy hypotheses, simulates them across hundreds of digital twins in an OASIS social network, evaluates multi-dimensional social impact metrics, and dynamically adapts policies when constraints fail. |
| **Why is it agentic?** | Unlike static prompt workflows, it maintains internal state, ranks candidate actions via a 7-factor mathematical utility function, responds to intermediate simulation feedback, replans upon failure, survives disruptions via provider failover, and halts safely without fabricating success. |
| **How does it work?** | **Goal & Constraints** $\rightarrow$ **Observe** $\rightarrow$ **Plan & Rank** $\rightarrow$ **Act (OASIS Simulation)** $\rightarrow$ **Evaluate (`SocialImpactModel`)** $\rightarrow$ **Verify** $\rightarrow$ **Replan**. |
| **How can I run it?** | `uv run deciverse autonomous-demo` (instant deterministic verification + artifact export). |
| **Where is the Web UI?** | `uv run streamlit run app/streamlit_app.py` $\rightarrow$ Navigate to **🤖 Autonomous Decision Agent**. |

---

## 🏛️ System Architecture

![DECIVERSE Decision Agent Architecture](architecture.png)

```text
User Goal & Multi-Dimensional Constraints
                    ↓
         Autonomous Controller
                    ↓
         Action Utility Planner / Replanner
                    ↓
               Tool Registry
                    ↓
        DECIVERSE / OASIS Simulation
                    ↓
        Emergent Social Outcomes
                    ↓
          SocialImpactModel
                    ↓
               Verifier
          ┌──────────┴──────────┐
          ↓                     ↓
       VERIFIED          FAILED / UNCERTAIN
          ↓                     ↓
    Final Decision           Replan
```

---

## 🤖 Instant Demo & Quick Start (Choose One)

### 1. Deterministic Hackathon Demo (CLI — 30 Seconds)
```bash
uv run deciverse autonomous-demo
```
*Executes candidate utility rankings, simulates social impact, diagnoses constraint failures, replans, and exports 7 structured audit artifacts.*

### 2. Interactive Web Application (Streamlit)
```bash
uv run python -m streamlit run app/streamlit_app.py
```
*Navigate to **🤖 Autonomous Decision Agent** in the sidebar to test policy presets, adjust constraint sliders, inject live disruptions, and interact with human checkpoints.*

### 3. Custom Autonomous Goal Optimization (CLI)
```bash
uv run deciverse autonomous-run \
  --goal "Find the most socially viable public transit fare structure" \
  --constraints "polarization<0.40,conflict<0.30,adoption>0.50" \
  --max-iterations 5 \
  --json
```

### 4. Run Automated Test Suite
```bash
uv run pytest -m "not integration"
```
*All 52 unit & end-to-end agentic tests passing (0 failures).*

---

## 📚 Key Hackathon Documentation

- 🎤 [**DEMO_SCRIPT.md**](DEMO_SCRIPT.md) — 3–5 minute presentation & live demo script with timed talking points.
- 🧠 [**JUDGE_QA.md**](JUDGE_QA.md) — 20 concise, technical answers for judge interrogation.
- 📋 [**DEMO_CHECKLIST.md**](DEMO_CHECKLIST.md) — Step-by-step pre-demo runbook and emergency recovery steps.
- 📋 [**SUBMISSION_CHECKLIST.md**](SUBMISSION_CHECKLIST.md) — Final hackathon submission readiness checklist.
- 📐 [**CHANGES_AGENTIC.md**](CHANGES_AGENTIC.md) — Mathematical utility formula, component changes, and verification proofs.
- 🗺️ [**CODEMAP.md**](CODEMAP.md) — Complete codebase map covering all 63 files and agent modules.
- 🔄 [**flow.md**](flow.md) — Runtime execution sequence and tool lifecycle.

---

## 🌟 Core Agentic Capabilities

### 1. Transparent Action Utility Planning
At each decision cycle, the agent evaluates a pool of candidate actions using a mathematical 7-factor utility formula:
$$\text{Utility} = w_g U_{\text{goal}} + w_c U_{\text{recovery}} + w_i U_{\text{info}} + w_e U_{\text{improve}} - w_{\text{cost}} C_{\text{action}} - w_r R_{\text{risk}} - w_p P_{\text{rep}}$$
- **Goal Progress ($w_g=0.25$)**: Direct contribution toward problem definition.
- **Constraint Recovery ($w_c=0.30$)**: Potential to resolve active violated metrics.
- **Information Gain ($w_i=0.15$)**: Uncertainty reduction value.
- **Expected Improvement ($w_e=0.20$)**: Anticipated viability uplift.
- **Action Cost ($w_{\text{cost}}=0.10$)**: Computational & token overhead.
- **Failure Risk ($w_r=0.10$)**: Probability of execution fault.
- **Repetition Penalty ($w_p=0.15$)**: Penalizes unadapted re-executions ($P_{\text{rep}}=0.90$).

### 2. Multi-Agent Society as External Environment
The agent interacts with an OASIS multi-agent social network containing heterogeneous stakeholder digital twins (citizens, business owners, advocacy groups, public officials) interacting across multiple rounds on Twitter/Reddit platforms.

### 3. Integrated Verifier & Social Impact Consequence Scoring
Emergent actions from the social simulation are converted into quantitative metrics by `SocialImpactModel`:
$$\text{Overall Score} = 0.20 \cdot \text{Acceptance} + 0.15 \cdot \text{Consensus} + 0.15 \cdot (1 - \text{Polarization}) + 0.15 \cdot (1 - \text{Conflict}) + 0.10 \cdot \text{Equity} + 0.15 \cdot \text{Adoption} + 0.10 \cdot \text{Stability}$$
The `Verifier` evaluates user-defined constraint expressions (`<`, `<=`, `>`, `>=`, `==`) against these scores.

### 4. Dynamic Replanning & Policy Mutation
When verification fails, the `Replanner` diagnoses root deficit dimensions (e.g. demographic divide vs. economic cost vs. low adoption) and generates targeted scenario mutations to reconcile conflicts.

### 5. Human-in-the-Loop Directive Ingestion
When high-stakes arbitration is required, the controller pauses at a human checkpoint. If a stakeholder rejects a proposal and enters a directive (e.g., *"Add a bus discount and cap peak toll at $3"*), the agent applies a repetition penalty to the rejected candidate, ingests the guidance into the next mutation, and resumes optimization.

### 6. Multi-Provider Failover & Fault Tolerance
Survives technical disruptions (LLM timeouts, malformed JSON, connection drops) via automated regex repair and a multi-provider fallback chain:
$$\text{Ollama (Local)} \longrightarrow \text{Claude Code CLI} \longrightarrow \text{Codex CLI}$$

### 7. Verifiable Audit Artifacts
Every run generates 7 immutable audit files in `uploads/runs/<run_id>/autonomous_run/`:
1. `state.json` — Complete serializable snapshot of the entire `DecisionState`.
2. `decisions.json` — Chronological candidate rankings and utility logs.
3. `tool_events.json` — Detailed tool call records with durations and payloads.
4. `failures.json` — Classified domain and technical failure logs.
5. `verification.json` — Formal metric checks against constraint thresholds.
6. `final_result.json` — Executive policy recommendation with final scores.
7. `demo_trace.md` — Human-readable markdown trace for judge review.

---

## 💻 CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `deciverse doctor` | Run environment, `.env`, provider, and local model diagnostics |
| `deciverse autonomous-demo` | Run deterministic closed-loop demo (verification + replan + artifact export) |
| `deciverse autonomous-run` | Run autonomous policy optimization against multi-dimensional constraints |
| `deciverse run` | Execute traditional sequential simulation pipeline |
| `deciverse runs list` | List prior runs and high-level metadata (slim format) |
| `deciverse runs status <id>` | View full execution status and manifest for a run |
| `deciverse runs export <id>` | Resolve absolute paths to persisted run artifacts |

### Options for `deciverse autonomous-demo`
```text
deciverse autonomous-demo
  --output-dir PATH           Custom directory to store demo artifacts
  --json                      Emit final state and manifest as JSON on stdout
```

### Options for `deciverse autonomous-run`
```text
deciverse autonomous-run
  --goal TEXT                 Natural language policy or organizational goal (required)
  --constraints TEXT          Comma-delimited constraint rules (e.g. "polarization<0.40,conflict<0.30,adoption>0.50")
  --files FILE [FILE ...]     Optional source documents (PDF/MD/TXT) for grounding context
  --max-iterations N          Maximum autonomous control loop iterations (default: 5)
  --output-dir PATH           Custom directory to store run artifacts
  --json                      Emit machine-readable JSON on stdout
```

---

## 🧪 Testing & Validation

```bash
# Run all unit and agentic tests
uv run pytest -m "not integration"

# Run specific agentic test suites
uv run pytest tests/test_agent_planner.py
uv run pytest tests/test_autonomous_controller_e2e.py
uv run pytest tests/test_agent_tools.py
uv run pytest tests/test_provider_failover.py
```

---

## 📁 Repository Structure

```text
app/
  cli.py                  CLI entry point (autonomous-demo, autonomous-run, run, doctor)
  streamlit_app.py        Streamlit UI featuring Autonomous Decision Agent dashboard
  config.py               Configuration loading & validation (.env)
  run_artifacts.py        Immutable RunStore artifact management & demo_trace.md export
  visual_snapshots.py     Pure Python SVG visualization generator
  agent/                  Autonomous Decision Agent Core
    autonomous_controller.py  Central closed-loop orchestrator managing iteration budgets
    state.py                  Persistent DecisionState dataclass hierarchy
    planner.py                ActionUtilityPlanner generating candidate action pools
    replanner.py              Diagnostic engine prescribing state-sensitive mutations
    verifier.py               Constraint parser directly reconciled with SocialImpactModel
    tools.py                  AgentToolRegistry wrapping context, scenario, & simulation
    provider_failover.py      Fallback chain (ollama → claude-cli → codex-cli)
    failure_handler.py        Domain vs technical failure classifier & JSON regex repair
    disruption_engine.py      Disruption injection engine for domain & technical shocks
    human_interaction.py      Human checkpoint manager with directive ingestion
    uncertainty.py            Multi-run variance & 95% confidence interval estimator
  research/               Research-oriented architecture layers
    community_context_engine.py      Community context modeling & ontology extraction
    stakeholder_digital_twin.py      Digital twin persona construction
    scenario_designer.py             Scenario formulation & mutation management
    emergent_behaviour_analyzer.py   Simulation action & sentiment pattern analysis
    social_impact_model.py           Multidimensional social consequence scoring
    decision_comparison_engine.py    Scenario tradeoff comparison & ranking
    decision_intelligence_engine.py  Comprehensive recommendation synthesis
  core/                   WorkbenchSession, TaskManager, ResourceLoader
  resources/              Persistence adapters (projects, documents, graph, simulations, reports)
  tools/                  Composable pipeline steps (ontology, graph, prepare, run, report)
  services/               Business logic (graph builder, simulation runner, report agent)
  utils/                  LLM client, logging, file parsing
scripts/                  OASIS simulation runner scripts (subprocess)
tests/                    Pytest test suite (52 tests passing)
```

---

## ⚠️ Limitations & Honest Boundaries

1. **Simulated Population Approximations**: Multi-agent LLM personas simulate plausible emergent dynamics but do not guarantee real-world predictive fidelity.
2. **Simulation Stochasticity**: Non-deterministic agent interactions are bounded using repeated sampling and 95% confidence intervals (`tool_request_additional_simulation`).
3. **Execution Runtime**: Full multi-round OASIS simulations take 15–45 seconds per round depending on agent count; the deterministic demo mode provides instant end-to-end evaluation for presentations.

---

## 📄 License & Acknowledgments

- **License**: [AGPL-3.0](LICENSE)
- **Foundational Projects**:
  - [MiroFish](https://github.com/666ghj/MiroFish) by 666ghj
  - [OASIS (CAMEL-AI)](https://github.com/camel-ai/oasis)
  - [Kùzu DB](https://kuzudb.com/)
