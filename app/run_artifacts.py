"""Persistent run manifests and artifact helpers for the agent-first CLI."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .config import Config


def _now() -> str:
    return datetime.now().isoformat()


class RunStore:
    """File-backed storage for immutable run artifacts."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir or os.path.join(Config.UPLOAD_FOLDER, "runs"))
        os.makedirs(self.root_dir, exist_ok=True)

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.root_dir, run_id)

    def manifest_path(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "manifest.json")

    def create(
        self,
        requirement: str,
        source_files: Iterable[str],
        project_name: str = "Unnamed Project",
    ) -> Dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = self.run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)
        for rel_path in (
            "input/source_files",
            "graph",
            "simulation",
            "report",
            "visuals",
            "logs",
        ):
            os.makedirs(os.path.join(run_dir, rel_path), exist_ok=True)

        manifest = {
            "run_id": run_id,
            "status": "created",
            "project_name": project_name,
            "requirement": requirement,
            "source_files": [os.path.abspath(path) for path in source_files],
            "project_id": None,
            "graph_id": None,
            "simulation_id": None,
            "report_id": None,
            "graph_build_task_id": None,
            "prepare_task_id": None,
            "report_task_id": None,
            "task_progress": 0,
            "task_message": "",
            "artifacts": {},
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        self.save(manifest)
        return manifest

    def save(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        manifest = dict(manifest)
        manifest["updated_at"] = _now()
        path = self.manifest_path(manifest["run_id"])
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return manifest

    def load(self, run_id: str) -> Dict[str, Any]:
        path = self.manifest_path(run_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Run not found: {run_id}")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def update(self, run_id: str, **changes: Any) -> Dict[str, Any]:
        try:
            manifest = self.load(run_id)
        except FileNotFoundError:
            manifest = {
                "run_id": run_id,
                "type": "autonomous_run",
                "status": "in_progress",
                "project_name": "Autonomous Decision Run",
                "artifacts": {},
                "created_at": _now(),
                "updated_at": _now(),
                "error": None,
            }
        manifest.update(changes)
        return self.save(manifest)

    def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        for item in os.listdir(self.root_dir):
            path = self.manifest_path(item)
            if not os.path.exists(path):
                continue
            try:
                manifests.append(self.load(item))
            except (OSError, json.JSONDecodeError):
                continue
        manifests.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return manifests[:limit]

    def write_json(self, run_id: str, rel_path: str, payload: Any) -> str:
        output_path = os.path.join(self.run_dir(run_id), rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return output_path

    def write_text(self, run_id: str, rel_path: str, content: str) -> str:
        output_path = os.path.join(self.run_dir(run_id), rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return output_path

    def copy_file(self, run_id: str, source_path: str, rel_path: str) -> Optional[str]:
        if not os.path.exists(source_path):
            return None
        output_path = os.path.join(self.run_dir(run_id), rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(source_path, output_path)
        return output_path

    def copy_tree(self, run_id: str, source_dir: str, rel_dir: str) -> Optional[str]:
        if not os.path.exists(source_dir):
            return None
        output_dir = os.path.join(self.run_dir(run_id), rel_dir)
        os.makedirs(output_dir, exist_ok=True)
        for entry in os.listdir(source_dir):
            src = os.path.join(source_dir, entry)
            dst = os.path.join(output_dir, entry)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        return output_dir

    def freeze_source_files(self, run_id: str, source_files: Iterable[str]) -> List[str]:
        copied: List[str] = []
        for index, source_path in enumerate(source_files, start=1):
            absolute = os.path.abspath(source_path)
            if not os.path.exists(absolute):
                continue
            basename = os.path.basename(absolute)
            rel_path = os.path.join("input", "source_files", f"{index:02d}_{basename}")
            output = self.copy_file(run_id, absolute, rel_path)
            if output:
                copied.append(output)
        return copied

    def record_artifact(self, run_id: str, key: str, rel_path: str) -> Dict[str, Any]:
        try:
            manifest = self.load(run_id)
        except FileNotFoundError:
            manifest = {
                "run_id": run_id,
                "type": "autonomous_run",
                "status": "in_progress",
                "project_name": "Autonomous Decision Run",
                "artifacts": {},
                "created_at": _now(),
                "updated_at": _now(),
                "error": None,
            }
        artifacts = dict(manifest.get("artifacts", {}))
        artifacts[key] = rel_path
        manifest["artifacts"] = artifacts
        return self.save(manifest)

    def create_autonomous_run(
        self,
        goal: str = "",
        source_files: Iterable[str] = (),
        constraints: Optional[Dict[str, str]] = None,
        project_name: str = "Autonomous Decision Run",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a dedicated autonomous run directory and initial manifest."""
        run_id = run_id or f"autorun_{uuid.uuid4().hex[:12]}"
        run_dir = self.run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)
        for rel_path in (
            "input/source_files",
            "autonomous_run",
            "visuals",
            "logs",
        ):
            os.makedirs(os.path.join(run_dir, rel_path), exist_ok=True)

        manifest = {
            "run_id": run_id,
            "type": "autonomous_run",
            "status": "created",
            "project_name": project_name,
            "goal": goal,
            "constraints": constraints or {},
            "source_files": [
                os.path.abspath(path)
                for path in source_files
                if isinstance(path, str) and os.path.exists(path)
            ],
            "best_scenario_id": None,
            "best_score": None,
            "artifacts": {},
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        self.save(manifest)
        return manifest

    def persist_autonomous_state(self, run_id: str, state_dict: Dict[str, Any]) -> Dict[str, str]:
        """Save the structured autonomous run files and record artifacts."""
        auto_dir = os.path.join(self.run_dir(run_id), "autonomous_run")
        os.makedirs(auto_dir, exist_ok=True)

        # Ensure manifest exists before recording artifacts
        manifest_path = self.manifest_path(run_id)
        if not os.path.exists(manifest_path):
            manifest = {
                "run_id": run_id,
                "type": "autonomous_run",
                "status": state_dict.get("status", "created"),
                "project_name": state_dict.get("project_name", "Autonomous Decision Run"),
                "goal": state_dict.get("goal", ""),
                "constraints": state_dict.get("constraints", {}),
                "source_files": [
                    os.path.abspath(p)
                    for p in state_dict.get("source_files", [])
                    if isinstance(p, str) and os.path.exists(p)
                ],
                "best_scenario_id": state_dict.get("best_scenario_id"),
                "best_score": state_dict.get("best_score"),
                "artifacts": {},
                "created_at": state_dict.get("created_at", _now()),
                "updated_at": _now(),
                "error": None,
            }
            self.save(manifest)

        # Write core files
        state_file = self.write_json(run_id, "autonomous_run/state.json", state_dict)
        self.record_artifact(run_id, "agent_state", "autonomous_run/state.json")

        decisions_file = self.write_json(
            run_id, "autonomous_run/decisions.json", state_dict.get("decision_history", [])
        )
        self.record_artifact(run_id, "decisions", "autonomous_run/decisions.json")

        tool_events_file = self.write_json(
            run_id, "autonomous_run/tool_events.json", state_dict.get("tool_events", [])
        )
        self.record_artifact(run_id, "tool_events", "autonomous_run/tool_events.json")

        failures_file = self.write_json(run_id, "autonomous_run/failures.json", {
            "system_failures": state_dict.get("system_failures", []),
            "domain_disruptions": state_dict.get("domain_disruptions", []),
        })
        self.record_artifact(run_id, "failures", "autonomous_run/failures.json")

        verification_file = self.write_json(
            run_id, "autonomous_run/verification.json", state_dict.get("verification_history", [])
        )
        self.record_artifact(run_id, "verification", "autonomous_run/verification.json")

        final_data = {
            "run_id": run_id,
            "status": state_dict.get("status", "completed"),
            "goal": state_dict.get("goal", ""),
            "best_scenario_id": state_dict.get("best_scenario_id"),
            "best_scenario_name": state_dict.get("best_scenario_name"),
            "best_score": state_dict.get("best_score"),
            "best_impact_result": state_dict.get("best_impact_result"),
            "final_outcome_summary": state_dict.get("final_outcome_summary"),
            "iteration_count": state_dict.get("iteration_count"),
            "max_iterations": state_dict.get("max_iterations"),
        }
        final_file = self.write_json(run_id, "autonomous_run/final_result.json", final_data)
        self.record_artifact(run_id, "final_result", "autonomous_run/final_result.json")

        # Generate and save demo_trace.md for judges
        trace_md = generate_demo_trace_markdown(state_dict)
        trace_file = self.write_text(run_id, "autonomous_run/demo_trace.md", trace_md)
        self.record_artifact(run_id, "demo_trace", "autonomous_run/demo_trace.md")

        # Update run manifest status
        self.update(
            run_id,
            status=state_dict.get("status", "completed"),
            best_scenario_id=state_dict.get("best_scenario_id"),
            best_score=state_dict.get("best_score"),
        )

        return {
            "state": state_file,
            "decisions": decisions_file,
            "tool_events": tool_events_file,
            "failures": failures_file,
            "verification": verification_file,
            "final_result": final_file,
            "demo_trace": trace_file,
        }


def generate_demo_trace_markdown(state_dict: Dict[str, Any]) -> str:
    """Generate clean, auditable Markdown demonstration trace for judges and evaluation."""
    run_id = state_dict.get("run_id", "unknown_run")
    goal = state_dict.get("goal", "")
    status = state_dict.get("status", "unknown")
    best_name = state_dict.get("best_scenario_name", "None")
    best_score = state_dict.get("best_score", 0.0) or 0.0
    provider = state_dict.get("active_provider", "ollama")
    iterations = state_dict.get("iteration_count", 0)
    max_iter = state_dict.get("max_iterations", 5)
    constraints = state_dict.get("constraints", {})

    lines = [
        f"# 🤖 DECIVERSE Autonomous Decision Agent — Execution Trace",
        f"",
        f"- **Run ID**: `{run_id}`",
        f"- **Timestamp**: `{state_dict.get('updated_at', _now())}`",
        f"- **Active LLM Provider**: `{provider}`",
        f"- **Final Status**: `{str(status).upper()}`",
        f"- **Iterations Completed**: `{iterations} / {max_iter}`",
        f"- **Best Verified Policy**: **{best_name}** (Viability Score: `{best_score:.2f}`)",
        f"",
        f"---",
        f"",
        f"## 1. Goal & Constraints Specification",
        f"",
        f"**Problem Statement / Policy Goal:**",
        f"> {goal}",
        f"",
        f"**Target Verification Constraints:**",
    ]
    for c_name, c_val in constraints.items():
        lines.append(f"- `{c_name}`: **{c_val}**")

    # 2. Decision History & Dynamic Candidate Rankings
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Dynamic Planning & Candidate Action Utility Rankings",
        f"",
    ])
    decisions = state_dict.get("decision_history", [])
    if not decisions:
        lines.append("*No decision cycles recorded.*")
    else:
        for idx, d in enumerate(decisions, start=1):
            selected = d.get("selected_action", {})
            candidates = d.get("candidate_actions", [])
            lines.append(f"### Decision Cycle #{idx} (Iteration {d.get('iteration', idx)})")
            lines.append(f"- **Current State Focus**: {d.get('current_state_summary', 'Policy optimization')}")
            lines.append(f"- **Decision Summary**: {d.get('decision_summary', '')}")
            lines.append(f"- **Selected Action**: `{selected.get('action_name', 'None')}` (Utility Score: `{selected.get('score', 0.0):.2f}`)")
            lines.append(f"")
            lines.append(f"| Rank | Candidate Action | Utility Score | Goal Progress | Constraint Rec. | Info Gain | Cost/Risk | Rationale |")
            lines.append(f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
            for c_idx, c in enumerate(candidates, start=1):
                star = " ⭐" if c.get("action_type") == selected.get("action_type") else ""
                lines.append(
                    f"| #{c_idx} | {c.get('action_name', '')}{star} | **{c.get('score', 0.0):.2f}** | "
                    f"{c.get('goal_progress', 0.0):.2f} | {c.get('constraint_recovery_potential', 0.0):.2f} | "
                    f"{c.get('information_gain', 0.0):.2f} | {(c.get('action_cost', 0.0) + c.get('failure_risk', 0.0)):.2f} | "
                    f"{c.get('rationale', '')} |"
                )
            lines.append(f"")

    # 3. Tool Execution & Simulation Events
    lines.extend([
        f"---",
        f"",
        f"## 3. Tool Execution & Environment Interactions",
        f"",
        f"| Timestamp | Tool Name | Success | Latency (s) | Key Output / Summary |",
        f"| :--- | :--- | :---: | :---: | :--- |",
    ])
    tool_events = state_dict.get("tool_events", [])
    for te in tool_events:
        t_name = te.get("tool_name", "")
        t_succ = "✅ Yes" if te.get("success") else "❌ No"
        t_dur = f"{te.get('duration_seconds', 0.0):.2f}"
        t_time = te.get("timestamp", "").split("T")[-1][:8] if "T" in te.get("timestamp", "") else ""
        t_out = te.get("output_summary") or ""
        if not t_out and isinstance(te.get("output_data"), dict):
            t_out = ", ".join(f"{k}={v}" for k, v in list(te.get("output_data", {}).items())[:2])
        lines.append(f"| {t_time} | `{t_name}` | {t_succ} | {t_dur} | {t_out[:60]} |")

    # 4. Failures, Disruptions & Adaptations
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Failure Recovery, Disruptions & Human Interactions",
        f"",
    ])
    failures = state_dict.get("system_failures", [])
    disruptions = state_dict.get("domain_disruptions", [])
    human_history = state_dict.get("human_interaction_history", [])

    if failures:
        lines.append(f"### ⚙️ Technical Failures & Failover")
        for f in failures:
            lines.append(f"- **Type**: `{f.get('failure_type')}` | **Provider**: `{f.get('provider')}` | **Recovered**: `{f.get('recovered')}`")
            lines.append(f"  - Error: {f.get('error_message')}")
            lines.append(f"  - Fallback: Switched to `{f.get('fallback_provider', 'N/A')}` via `{f.get('recovery_strategy')}`")
        lines.append(f"")

    if disruptions:
        lines.append(f"### ⚡ Domain Environment Disruptions")
        for d in disruptions:
            lines.append(f"- **Disruption**: `{d.get('disruption_type')}` (Severity: `{d.get('severity', 1.0)}`)")
            lines.append(f"  - Injected at: `{d.get('timestamp')}` | Target: `{d.get('target_scenario_id', 'all')}`")
        lines.append(f"")

    if human_history:
        lines.append(f"### 👤 Human Stakeholder Interactions")
        for h in human_history:
            lines.append(f"- **Action**: `{h.get('selected_option', 'Response')}` (Status: `{h.get('status')}`)")
            if h.get("text_input"):
                lines.append(f"  - **Feedback Directive**: *\"{h.get('text_input')}\"*")
        lines.append(f"")

    if not (failures or disruptions or human_history):
        lines.append("*Nominal path: No unrecovered failures or external disruptions recorded.*")
        lines.append(f"")

    # 5. Verification Audits
    lines.extend([
        f"---",
        f"",
        f"## 5. Constraint Verification History",
        f"",
    ])
    verifications = state_dict.get("verification_history", [])
    for v in verifications:
        v_stat = v.get("status", "")
        icon = "🟢" if v_stat == "VERIFIED" else ("🔴" if v_stat == "FAILED" else "🟡")
        lines.append(f"### {icon} Scenario: '{v.get('scenario_name')}' — Status: `{v_stat}`")
        lines.append(f"- **Overall Viability Score**: `{v.get('overall_score', 0.0):.2f}` (Confidence: `{v.get('confidence', 1.0):.2f}`)")
        if v.get("satisfied_constraints"):
            lines.append(f"- **Satisfied Constraints**: {', '.join(f'`{c}`' for c in v.get('satisfied_constraints', []))}")
        if v.get("violated_constraints"):
            lines.append(f"- **Violated Constraints**: {', '.join(f'`{c}`' for c in v.get('violated_constraints', []))}")
        if v.get("uncertain_aspects"):
            lines.append(f"- **Uncertain Aspects**: {', '.join(v.get('uncertain_aspects', []))}")
        lines.append(f"")

    # 6. Final Decision Recommendation
    lines.extend([
        f"---",
        f"",
        f"## 6. Final Verified Decision Intelligence",
        f"",
        f"**Summary Outcome:**",
        f"> {state_dict.get('final_outcome_summary', 'Run finalized.')}",
        f"",
    ])
    best_impact = state_dict.get("best_impact_result")
    if best_impact and isinstance(best_impact, dict):
        lines.append(f"| Dimension | Score | Assessment |")
        lines.append(f"| :--- | :---: | :--- |")
        lines.append(f"| **Overall Viability** | `{best_impact.get('overall_score', 0.0):.2f}` | Composite multi-stakeholder utility |")
        lines.append(f"| **Citizen Acceptance** | `{best_impact.get('acceptance_score', 0.0):.2f}` | Aggregate public approval |")
        lines.append(f"| **Consensus Score** | `{best_impact.get('consensus_score', 0.0):.2f}` | Stakeholder agreement level |")
        lines.append(f"| **Polarization Score** | `{best_impact.get('polarization_score', 0.0):.2f}` | Inter-group ideological divergence |")
        lines.append(f"| **Conflict Index** | `{best_impact.get('conflict_score', 0.0):.2f}` | Active opposition frequency |")
        lines.append(f"| **Adoption Uptake** | `{best_impact.get('adoption_score', 0.0):.2f}` | Projected citizen participation |")
        lines.append(f"| **Equity Balance** | `{best_impact.get('equity_score', 0.0):.2f}` | Distribution across demographics |")
        lines.append(f"| **Long-term Stability** | `{best_impact.get('stability_score', 0.0):.2f}` | Policy durability over rounds |")

    return "\n".join(lines) + "\n"


