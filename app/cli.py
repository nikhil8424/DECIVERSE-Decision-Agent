"""Minimal run-first CLI for DECIVERSE."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import logging

from ._agent_cli import DoctorCheck, doctor_runner
from .utils.logger import get_logger
from .cli_display import PipelineDisplay
from .config import Config
from .core.task_manager import TaskManager, TaskStatus
from .core.workbench_session import WorkbenchSession
from .resources.reports import ReportStore
from .run_artifacts import RunStore
from .services.graph_builder import GraphBuilderService
from .services.graph_db import GraphDatabase
from .services.simulation_manager import SimulationManager
from .services.simulation_runner import RunnerStatus, SimulationRunner
from .visual_snapshots import generate_visual_snapshots

logger = get_logger('mirofish.cli')

DEFAULT_PROJECT_NAME = "DECIVERSE Run"
DEFAULT_PARALLEL_PROFILE_COUNT = 5


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _emit(payload: Any, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)
    return 0


class LocalFileInput:
    """Minimal file wrapper compatible with ProjectManager.save_file_to_project."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.filename = os.path.basename(self.path)

    def save(self, destination: str) -> None:
        shutil.copy2(self.path, destination)


def _require_existing_files(paths: Iterable[str]) -> List[str]:
    resolved = [os.path.abspath(path) for path in paths]
    missing = [path for path in resolved if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing)}")
    return resolved


def _default_project_name(source_files: List[str]) -> str:
    if not source_files:
        return DEFAULT_PROJECT_NAME
    stem = os.path.splitext(os.path.basename(source_files[0]))[0].strip()
    if not stem:
        return DEFAULT_PROJECT_NAME
    if len(source_files) == 1:
        return stem
    return f"{stem} +{len(source_files) - 1}"


def _wait_for_task(
    task_id: str,
    poll_interval: float = 1.0,
    on_update=None,
):
    manager = TaskManager()
    last_snapshot = None
    while True:
        task = manager.get_task(task_id)
        if task is None:
            raise RuntimeError(f"Task not found: {task_id}")
        snapshot = (task.status.value, task.progress, task.message, task.error)
        if snapshot != last_snapshot and on_update is not None:
            on_update(task)
            last_snapshot = snapshot
        if task.status == TaskStatus.COMPLETED:
            return task
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(task.error or task.message or f"Task failed: {task_id}")
        time.sleep(poll_interval)


def _get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    task = TaskManager().get_task(task_id)
    return task.result if task else None


def _wait_for_simulation(simulation_id: str, poll_interval: float = 2.0, on_update=None):
    last_snapshot = None
    while True:
        state = SimulationRunner.get_run_state(simulation_id)
        if state is None:
            raise RuntimeError(f"Simulation run state not found: {simulation_id}")
        snapshot = (state.runner_status.value, state.current_round, state.total_rounds, state.error)
        if snapshot != last_snapshot and on_update is not None:
            on_update(state)
            last_snapshot = snapshot
        if state.runner_status == RunnerStatus.COMPLETED:
            return state
        if state.runner_status == RunnerStatus.FAILED:
            raise RuntimeError(state.error or f"Simulation failed: {simulation_id}")
        time.sleep(poll_interval)


def _write_action_log(output_path: str, actions: List[Any]) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ordered = sorted(actions, key=lambda action: action.timestamp)
    with open(output_path, "w", encoding="utf-8") as handle:
        for action in ordered:
            handle.write(json.dumps(action.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def _top_agents(agent_stats: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    ordered = sorted(agent_stats, key=lambda item: item.get("total_actions", 0), reverse=True)
    return ordered[:limit]


def _simulation_dir(simulation_id: str) -> str:
    return os.path.join(SimulationManager.SIMULATION_DATA_DIR, simulation_id)


def _record_if_copied(store: RunStore, run_id: str, key: str, source_path: str, rel_path: str) -> None:
    copied_path = store.copy_file(run_id, source_path, rel_path)
    if copied_path:
        store.record_artifact(run_id, key, rel_path)


def _resolve_artifact_paths(store: RunStore, manifest: Dict[str, Any]) -> Dict[str, str]:
    run_dir = store.run_dir(manifest["run_id"])
    resolved: Dict[str, str] = {}
    for key, rel_path in manifest.get("artifacts", {}).items():
        absolute_path = os.path.abspath(os.path.join(run_dir, rel_path))
        if os.path.exists(absolute_path):
            resolved[key] = absolute_path
    return resolved


def _refresh_run_manifest(store: RunStore, run_id: str) -> Dict[str, Any]:
    manifest = store.load(run_id)
    changed = False

    if manifest.get("status") == "graph_building" and manifest.get("graph_build_task_id"):
        task = TaskManager().get_task(manifest["graph_build_task_id"])
        if task is not None:
            manifest["task_progress"] = task.progress
            manifest["task_message"] = task.message
            if task.status == TaskStatus.COMPLETED:
                manifest["status"] = "graph_ready"
                manifest["graph_id"] = (task.result or {}).get("graph_id")
            elif task.status == TaskStatus.FAILED:
                manifest["status"] = "failed"
                manifest["error"] = task.error or task.message
            changed = True

    if manifest.get("status") == "simulation_preparing" and manifest.get("prepare_task_id"):
        task = TaskManager().get_task(manifest["prepare_task_id"])
        if task is not None:
            manifest["task_progress"] = task.progress
            manifest["task_message"] = task.message
            if task.status == TaskStatus.COMPLETED:
                manifest["status"] = "simulation_ready"
            elif task.status == TaskStatus.FAILED:
                manifest["status"] = "failed"
                manifest["error"] = task.error or task.message
            changed = True

    if manifest.get("status") == "simulation_running" and manifest.get("simulation_id"):
        state = SimulationRunner.get_run_state(manifest["simulation_id"])
        if state is not None:
            manifest["task_progress"] = state.to_dict().get("progress_percent", 0)
            manifest["task_message"] = f"{state.current_round}/{state.total_rounds} rounds"
            if state.runner_status == RunnerStatus.COMPLETED:
                manifest["status"] = "simulation_completed"
            elif state.runner_status == RunnerStatus.FAILED:
                manifest["status"] = "failed"
                manifest["error"] = state.error
            changed = True

    if manifest.get("status") == "report_generating" and manifest.get("report_task_id"):
        task = TaskManager().get_task(manifest["report_task_id"])
        if task is not None:
            manifest["task_progress"] = task.progress
            manifest["task_message"] = task.message
            if task.status == TaskStatus.COMPLETED:
                manifest["status"] = "completed"
                manifest["report_id"] = (task.result or {}).get("report_id", manifest.get("report_id"))
            elif task.status == TaskStatus.FAILED:
                manifest["status"] = "failed"
                manifest["error"] = task.error or task.message
            changed = True

    if changed:
        manifest = store.save(manifest)
    return manifest


def _generate_verdict(report_markdown: str, requirement: str) -> Dict[str, Any]:
    """Generate a machine-readable verdict from the report for agent consumption."""
    from .utils.llm_client import LLMClient

    try:
        llm = LLMClient()
        result = llm.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract a structured verdict from a simulation report. "
                        "Return JSON with exactly these fields:\n"
                        '- "prediction": one-sentence prediction (max 100 words)\n'
                        '- "confidence": float 0.0-1.0 (how confident the simulation evidence is)\n'
                        '- "key_dynamics": array of 3-5 short strings describing the main dynamics observed\n'
                        '- "signals": array of objects with {"signal": string, "direction": "positive"|"negative"|"mixed", "strength": float 0.0-1.0}\n'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Requirement: {requirement}\n\nReport:\n{report_markdown[:6000]}",
                },
            ],
            temperature=0.2,
        )
        # Ensure required fields exist with defaults
        return {
            "prediction": result.get("prediction", "No prediction generated"),
            "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
            "key_dynamics": result.get("key_dynamics", [])[:5],
            "signals": result.get("signals", [])[:8],
        }
    except Exception as e:
        logger.warning(f"Verdict generation failed (non-fatal): {e}")
        return {
            "prediction": "Verdict generation failed",
            "confidence": 0.0,
            "key_dynamics": [],
            "signals": [],
        }


def _collect_run_outputs(
    store: RunStore,
    manifest: Dict[str, Any],
    graph_data: Dict[str, Any],
    graph_stats: Dict[str, Any],
    timeline: List[Dict[str, Any]],
    agent_stats: List[Dict[str, Any]],
    actions: List[Any],
    report_payload: Optional[Dict[str, Any]],
    report_markdown: str,
) -> Dict[str, Any]:
    run_id = manifest["run_id"]

    store.write_json(run_id, "graph/graph.json", graph_data)
    store.record_artifact(run_id, "graph_json", "graph/graph.json")
    store.write_json(run_id, "graph/graph_summary.json", graph_stats)
    store.record_artifact(run_id, "graph_summary", "graph/graph_summary.json")

    store.write_json(run_id, "simulation/timeline.json", timeline)
    store.record_artifact(run_id, "timeline_json", "simulation/timeline.json")
    store.write_json(run_id, "simulation/top_agents.json", _top_agents(agent_stats))
    store.record_artifact(run_id, "top_agents", "simulation/top_agents.json")
    _write_action_log(os.path.join(store.run_dir(run_id), "simulation", "actions.jsonl"), actions)
    store.record_artifact(run_id, "actions_log", "simulation/actions.jsonl")

    sim_dir = _simulation_dir(manifest["simulation_id"])
    _record_if_copied(store, run_id, "simulation_config", os.path.join(sim_dir, "simulation_config.json"), "simulation/config.json")
    _record_if_copied(store, run_id, "reddit_profiles", os.path.join(sim_dir, "reddit_profiles.json"), "simulation/reddit_profiles.json")
    _record_if_copied(store, run_id, "twitter_profiles", os.path.join(sim_dir, "twitter_profiles.csv"), "simulation/twitter_profiles.csv")
    _record_if_copied(store, run_id, "simulation_log", os.path.join(sim_dir, "simulation.log"), "logs/simulation.log")

    if report_payload is not None:
        store.write_json(run_id, "report/meta.json", report_payload)
        store.record_artifact(run_id, "report_meta", "report/meta.json")
    if report_markdown:
        store.write_text(run_id, "report/report.md", report_markdown)
        store.record_artifact(run_id, "report_markdown", "report/report.md")

    # Generate machine-readable verdict for agent consumption
    if report_markdown:
        verdict = _generate_verdict(report_markdown, manifest.get("requirement", ""))
    else:
        verdict = {"prediction": "No report available", "confidence": 0.0, "key_dynamics": [], "signals": []}
    store.write_json(run_id, "report/verdict.json", verdict)
    store.record_artifact(run_id, "verdict", "report/verdict.json")

    summary = {
        "run_id": manifest["run_id"],
        "project_id": manifest.get("project_id"),
        "graph_id": manifest.get("graph_id"),
        "simulation_id": manifest.get("simulation_id"),
        "report_id": manifest.get("report_id"),
        "node_count": graph_stats.get("node_count", 0),
        "edge_count": graph_stats.get("edge_count", 0),
        "rounds": len(timeline),
        "total_actions": sum(item.get("total_actions", 0) for item in timeline),
        "top_agents": _top_agents(agent_stats, limit=10),
        "verdict": verdict,
    }
    store.write_json(run_id, "report/summary.json", summary)
    store.record_artifact(run_id, "report_summary", "report/summary.json")

    visuals = generate_visual_snapshots(graph_data, timeline, os.path.join(store.run_dir(run_id), "visuals"))
    for key, absolute_path in visuals.items():
        relative = os.path.relpath(absolute_path, store.run_dir(run_id))
        store.record_artifact(run_id, key, relative)

    return store.load(run_id)


def _run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    # Validate agent count if provided
    if hasattr(args, 'agent_count') and args.agent_count is not None:
        if args.agent_count < 5:
            _stderr(f"Warning: Agent count {args.agent_count} is below recommended minimum of 5")
        elif args.agent_count > 500:
            _stderr(f"Warning: Agent count {args.agent_count} exceeds recommended maximum of 500")
    
    source_files = _require_existing_files(args.files)
    project_name = _default_project_name(source_files)
    store = RunStore(root_dir=args.output_dir)
    manifest = store.create(args.requirement, source_files, project_name=project_name)
    run_id = manifest["run_id"]

    provider_label = f"ollama ({Config.OLLAMA_MODEL})" if Config.LLM_PROVIDER == "ollama" else Config.LLM_PROVIDER
    display = PipelineDisplay(
        project_name=project_name,
        run_id=run_id,
        provider=provider_label,
        platform=args.platform,
        json_mode=args.json,
    )
    display.start()

    # Suppress service-layer console noise when rich display is active
    if not args.json:
        from .utils.logger import set_console_level
        set_console_level(logging.WARNING)

    store.write_text(run_id, "input/requirement.txt", args.requirement)
    store.record_artifact(run_id, "requirement", "input/requirement.txt")
    store.freeze_source_files(run_id, source_files)
    store.record_artifact(run_id, "source_files_dir", "input/source_files")

    session = WorkbenchSession.open(metadata={"entrypoint": "cli.run", "run_id": run_id})
    current_step = "ontology"

    try:
        # --- ontology ---
        display.start_step("ontology")
        project_result = session.generate_ontology(
            simulation_requirement=args.requirement,
            uploaded_files=[LocalFileInput(path) for path in source_files],
            project_name=project_name,
        )
        ontology = project_result.get("ontology", {})
        n_entity_types = len(ontology.get("entity_types", []))
        n_edge_types = len(ontology.get("relationship_types", ontology.get("edge_types", [])))
        display.complete_step("ontology", f"{n_entity_types} entity types, {n_edge_types} edge types")

        store.update(run_id, project_id=project_result["project_id"], status="graph_building", task_progress=0, task_message="Ontology generated")
        store.write_json(run_id, "input/ontology.json", ontology)
        store.record_artifact(run_id, "ontology", "input/ontology.json")
        store.write_text(run_id, "input/analysis_summary.txt", project_result.get("analysis_summary", ""))
        store.record_artifact(run_id, "analysis_summary", "input/analysis_summary.txt")

        # --- graph ---
        current_step = "graph"
        display.start_step("graph")
        graph_result = session.start_graph_build(project_id=project_result["project_id"])
        store.update(run_id, graph_build_task_id=graph_result["task_id"], status="graph_building")
        _wait_for_task(
            graph_result["task_id"],
            on_update=lambda task: (
                store.update(run_id, status="graph_building", task_progress=task.progress, task_message=task.message),
                display.update_step("graph", task.message or ""),
            ),
        )
        graph_id = (_get_task_result(graph_result["task_id"]) or {}).get("graph_id")
        if not graph_id:
            raise RuntimeError("Graph build completed without a graph_id")

        graph_builder = GraphBuilderService()
        graph_db = GraphDatabase()
        graph_data = graph_builder.get_graph_data(graph_id)
        graph_stats = graph_db.get_graph_statistics(graph_id)
        n_nodes = graph_stats.get("node_count", 0)
        n_edges = graph_stats.get("edge_count", 0)
        display.complete_step("graph", f"{n_nodes} nodes, {n_edges} edges")
        store.update(run_id, graph_id=graph_id, status="graph_ready", task_progress=100, task_message="Graph ready")

        # --- profiles ---
        current_step = "profiles"
        display.start_step("profiles")
        enable_twitter = args.platform in {"parallel", "twitter"}
        enable_reddit = args.platform in {"parallel", "reddit"}
        simulation_state = session.create_simulation(
            project_id=project_result["project_id"],
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
        )
        simulation_id = simulation_state.simulation_id
        store.update(run_id, simulation_id=simulation_id, status="simulation_preparing", task_progress=0, task_message="Simulation created")

        prepare_result = session.start_simulation_preparation(
            simulation_id=simulation_id,
            use_llm_for_profiles=True,
            parallel_profile_count=DEFAULT_PARALLEL_PROFILE_COUNT,
            agent_count=args.agent_count,
        )
        if prepare_result.get("task_id"):
            store.update(run_id, prepare_task_id=prepare_result["task_id"], status="simulation_preparing")
            _wait_for_task(
                prepare_result["task_id"],
                on_update=lambda task: (
                    store.update(run_id, status="simulation_preparing", task_progress=task.progress, task_message=task.message),
                    display.update_step("profiles", task.message or ""),
                ),
            )

        sim_dir = _simulation_dir(simulation_id)
        agent_count = 0
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                agent_count = len(json.load(f).get("agent_configs", []))
        
        # Display requested vs actual agent count
        requested_count = args.agent_count if hasattr(args, 'agent_count') and args.agent_count else None
        if requested_count:
            display.complete_step("profiles", f"{agent_count} agents (requested: {requested_count})")
        else:
            display.complete_step("profiles", f"{agent_count} agents")
        store.update(run_id, status="simulation_ready", task_progress=100, task_message="Simulation ready")

        _record_if_copied(store, run_id, "frozen_simulation_config", os.path.join(sim_dir, "simulation_config.json"), "input/simulation_config.json")
        _record_if_copied(store, run_id, "frozen_reddit_profiles", os.path.join(sim_dir, "reddit_profiles.json"), "input/reddit_profiles.json")
        _record_if_copied(store, run_id, "frozen_twitter_profiles", os.path.join(sim_dir, "twitter_profiles.csv"), "input/twitter_profiles.csv")

        # --- simulation ---
        current_step = "simulation"
        display.start_step("simulation")
        session.start_simulation_run(
            simulation_id=simulation_id,
            platform=args.platform,
            max_rounds=args.max_rounds,
            enable_graph_memory_update=False,
        )
        _wait_for_simulation(
            simulation_id,
            on_update=lambda state: (
                store.update(run_id, status="simulation_running", task_progress=state.to_dict().get("progress_percent", 0), task_message=f"{state.current_round}/{state.total_rounds} rounds"),
                display.update_step("simulation", f"round {state.current_round}/{state.total_rounds}"),
            ),
        )

        timeline = SimulationRunner.get_timeline(simulation_id)
        agent_stats = SimulationRunner.get_agent_stats(simulation_id)
        actions = SimulationRunner.get_all_actions(simulation_id)
        total_actions = sum(item.get("total_actions", 0) for item in timeline)
        display.complete_step("simulation", f"{len(timeline)} rounds, {total_actions} actions")
        store.update(run_id, status="simulation_completed", task_progress=100, task_message="Simulation completed")

        # --- report ---
        current_step = "report"
        display.start_step("report")
        report_payload = None
        report_markdown = ""
        report_store = ReportStore()

        report_result = session.start_report_generation(simulation_id=simulation_id)
        report_id = report_result.get("report_id")
        if report_result.get("task_id"):
            store.update(run_id, report_id=report_id, report_task_id=report_result["task_id"], status="report_generating")
            report_task = _wait_for_task(
                report_result["task_id"],
                on_update=lambda task: (
                    store.update(run_id, status="report_generating", task_progress=task.progress, task_message=task.message),
                    display.update_step("report", task.message or ""),
                ),
            )
            report_id = (report_task.result or {}).get("report_id", report_id)
        if report_id:
            report = report_store.get(report_id)
            if report is not None:
                report_payload = report.to_dict()
                report_markdown = report.markdown_content
        display.complete_step("report", "done")

        # --- visuals ---
        current_step = "visuals"
        display.start_step("visuals")
        final_manifest = store.update(run_id, report_id=report_id, status="completed", task_progress=100, task_message="Run completed")
        final_manifest = _collect_run_outputs(
            store=store,
            manifest=final_manifest,
            graph_data=graph_data,
            graph_stats=graph_stats,
            timeline=timeline,
            agent_stats=agent_stats,
            actions=actions,
            report_payload=report_payload,
            report_markdown=report_markdown,
        )
        visual_keys = {"swarm_overview", "cluster_map", "timeline", "platform_split"}
        n_visuals = sum(1 for k in final_manifest.get("artifacts", {}) if k in visual_keys)
        display.complete_step("visuals", f"{n_visuals} snapshots")

        display.finish()
        return final_manifest
    except Exception as exc:
        display.fail_step(current_step, str(exc)[:80])
        display.finish()
        store.update(run_id, status="failed", error=str(exc), task_message=str(exc))
        raise


def cmd_doctor() -> int:
    """Run environment/config diagnostics for deciverse."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(repo_root, ".env")

    valid_providers = ("claude-cli", "codex-cli", "ollama")

    def provider_set() -> bool:
        return bool(os.environ.get("LLM_PROVIDER", "").strip())

    def provider_valid() -> bool:
        return Config.LLM_PROVIDER in valid_providers

    # .env is optional — warn but don't fail. Emit before the checks table so
    # the output order reads top-to-bottom: warnings, then checks, then summary.
    if not os.path.exists(env_path):
        print(
            f"warning: .env file not found at {env_path} "
            "(using process environment only)",
            file=sys.stderr,
        )

    checks: list[DoctorCheck] = [
        DoctorCheck(
            name="LLM_PROVIDER set",
            check=provider_set,
            hint="export LLM_PROVIDER=ollama (or claude-cli / codex-cli) in .env",
        ),
        DoctorCheck(
            name="LLM_PROVIDER valid",
            check=provider_valid,
            hint=f"must be one of {valid_providers}; got '{Config.LLM_PROVIDER}'",
        ),
    ]

    if Config.LLM_PROVIDER in ("claude-cli", "codex-cli"):
        provider = Config.LLM_PROVIDER

        def provider_binary_on_path() -> bool:
            return shutil.which(provider) is not None

        checks.append(
            DoctorCheck(
                name=f"{provider} binary on PATH",
                check=provider_binary_on_path,
                hint=f"install the {provider} binary or add it to PATH",
            )
        )
    elif Config.LLM_PROVIDER == "ollama":
        import urllib.request

        def ollama_reachable() -> bool:
            try:
                url = f"{Config.OLLAMA_BASE_URL}/api/tags"
                with urllib.request.urlopen(url, timeout=3) as res:
                    return res.status == 200
            except Exception:
                return False

        def ollama_model_installed() -> bool:
            try:
                url = f"{Config.OLLAMA_BASE_URL}/api/tags"
                with urllib.request.urlopen(url, timeout=3) as res:
                    data = json.loads(res.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                target = Config.OLLAMA_MODEL
                return any(target == m or target == m.split(":")[0] or target in m for m in models)
            except Exception:
                return False

        checks.append(
            DoctorCheck(
                name=f"Ollama reachable at {Config.OLLAMA_BASE_URL}",
                check=ollama_reachable,
                hint=f"start Ollama service or check OLLAMA_BASE_URL={Config.OLLAMA_BASE_URL}",
            )
        )
        checks.append(
            DoctorCheck(
                name=f"Ollama model '{Config.OLLAMA_MODEL}' installed",
                check=ollama_model_installed,
                hint=f"run `ollama pull {Config.OLLAMA_MODEL}` to install the model",
            )
        )

    return doctor_runner(checks, exit_on_fail=False)


def _handle_command(args: argparse.Namespace) -> Dict[str, Any]:
    if args.command == "runs" and args.runs_command == "list":
        store = RunStore(root_dir=args.output_dir)
        manifests = [_refresh_run_manifest(store, item["run_id"]) for item in store.list(limit=args.limit)]
        slim = [
            {
                "run_id": m.get("run_id"),
                "status": m.get("status"),
                "created_at": m.get("created_at"),
                "artifact_count": len(m.get("artifacts", {})),
            }
            for m in manifests
        ]
        return {"runs": slim, "count": len(slim)}
    if args.command == "runs" and args.runs_command == "status":
        store = RunStore(root_dir=args.output_dir)
        return _refresh_run_manifest(store, args.run_id)
    if args.command == "runs" and args.runs_command == "export":
        store = RunStore(root_dir=args.output_dir)
        manifest = _refresh_run_manifest(store, args.run_id)
        artifacts = _resolve_artifact_paths(store, manifest)
        if args.artifact:
            if args.artifact not in artifacts:
                raise FileNotFoundError(f"Artifact not found for run {args.run_id}: {args.artifact}")
            return {
                "run_id": args.run_id,
                "artifact": args.artifact,
                "path": artifacts[args.artifact],
            }
        return {
            "run_id": args.run_id,
            "count": len(artifacts),
            "artifacts": artifacts,
        }
    if args.command == "run":
        return _run_pipeline(args)
    if args.command == "autonomous-run":
        return _run_autonomous(args)
    if args.command == "autonomous-demo":
        return _run_autonomous_demo(args)
    raise RuntimeError("Unknown command")


def _run_autonomous(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the full autonomous decision control loop."""
    from .agent import (
        AutonomousDecisionController,
        DecisionState,
        DisruptionEngine,
        DomainDisruptionType,
        ProviderFailoverManager,
        TechnicalDisruptionType,
    )

    source_files = _require_existing_files(args.files)
    goal = args.goal
    store = RunStore(root_dir=args.output_dir)

    # Parse constraints (comma-separated or dict)
    constraints = {}
    if args.constraints:
        for item in args.constraints.split(","):
            item = item.strip()
            if not item:
                continue
            if "<=" in item:
                k, v = item.split("<=", 1)
                constraints[k.strip()] = f"<={v.strip()}"
            elif ">=" in item:
                k, v = item.split(">=", 1)
                constraints[k.strip()] = f">={v.strip()}"
            elif "<" in item:
                k, v = item.split("<", 1)
                constraints[k.strip()] = f"<{v.strip()}"
            elif ">" in item:
                k, v = item.split(">", 1)
                constraints[k.strip()] = f">{v.strip()}"
            elif "=" in item:
                k, v = item.split("=", 1)
                constraints[k.strip()] = f"=={v.strip()}"
            else:
                constraints[item] = "<0.35"
    else:
        constraints = {"polarization": "<0.35", "conflict": "<0.30", "adoption": ">0.60"}

    # Initialize run store
    manifest = store.create_autonomous_run(
        goal=goal,
        source_files=source_files,
        constraints=constraints,
        project_name=args.project_name or "Autonomous Decision Run",
    )
    run_id = manifest["run_id"]
    store.freeze_source_files(run_id, source_files)

    # Setup provider and disruptions
    provider_mgr = ProviderFailoverManager(primary_provider=args.provider or Config.LLM_PROVIDER)
    disruption_engine = DisruptionEngine()

    if getattr(args, "inject_domain_disruption", None):
        try:
            dtype = DomainDisruptionType(args.inject_domain_disruption)
            disruption_engine.inject_domain_disruption(dtype)
        except Exception as e:
            _stderr(f"Warning: Unknown domain disruption type: {e}")

    if getattr(args, "inject_technical_disruption", None):
        try:
            ttype = TechnicalDisruptionType(args.inject_technical_disruption)
            disruption_engine.inject_technical_disruption(ttype)
        except Exception as e:
            _stderr(f"Warning: Unknown technical disruption type: {e}")

    # Trace callback for live terminal output
    def _trace(phase: str, message: str, payload: Optional[Dict[str, Any]] = None):
        if not getattr(args, "json", False):
            _stderr(f"[{phase}] {message}")

    state = DecisionState(
        run_id=run_id,
        goal=goal,
        problem_statement=goal,
        source_files=source_files,
        constraints=constraints,
        max_iterations=args.max_iterations or 5,
        primary_provider=provider_mgr.primary_provider,
        active_provider=provider_mgr.active_provider,
    )

    controller = AutonomousDecisionController(
        state=state,
        provider_manager=provider_mgr,
        disruption_engine=disruption_engine,
        on_trace_callback=_trace,
    )

    _trace("START", f"Launching Autonomous Decision Controller for run {run_id}")
    final_state = controller.run_until_completion()

    # Persist artifacts
    saved_artifacts = store.persist_autonomous_state(run_id, final_state.to_dict())

    return {
        "run_id": run_id,
        "status": final_state.status.value,
        "goal": final_state.goal,
        "best_scenario_id": final_state.best_scenario_id,
        "best_scenario_name": final_state.best_scenario_name,
        "best_score": final_state.best_score,
        "final_outcome_summary": final_state.final_outcome_summary,
        "iterations_completed": final_state.iteration_count,
        "max_iterations": final_state.max_iterations,
        "decisions_count": len(final_state.decision_history),
        "tool_events_count": len(final_state.tool_events),
        "verification_history": [v.to_dict() for v in final_state.verification_history],
        "artifacts": saved_artifacts,
    }


def _run_autonomous_demo(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute a real, deterministic, closed-loop demonstration of the Autonomous Decision Agent.

    Demonstrates:
    1. Goal & Constraint Ingestion
    2. Context Observation & Stakeholder Entity Grounding
    3. Action Utility Planning & Transparent Candidate Ranking
    4. Multi-Agent Simulation & Social Consequence Measurement
    5. Constraint Failure & Root-Cause Replanning Diagnostics
    6. Human Checkpoint & Stakeholder Directive Injection
    7. Policy Mutation & Dynamic Re-planning
    8. Environment Domain Disruption Handling
    9. Technical LLM Timeout & Provider Failover Recovery
    10. Final Constraint Verification & Audit Artifact Persistence
    """
    from .agent import (
        AgentStatus,
        AutonomousDecisionController,
        DecisionState,
        DisruptionEngine,
        DomainDisruptionType,
        ProviderFailoverManager,
        TechnicalDisruptionType,
    )

    # Determine files
    files = list(args.files) if getattr(args, "files", None) else ["demo_transit_policy.md"]
    for f in files:
        if not os.path.exists(f) and os.path.basename(f) == "demo_transit_policy.md":
            with open(f, "w", encoding="utf-8") as demo_f:
                demo_f.write("# Metropolitan Transit Authority Fare Modernization & Policy\n\nPolicy provisions for fare restructuring, off-peak discounts, and commuter incentives.\n")
    source_files = _require_existing_files(files)
    goal = getattr(args, "goal", None) or "Find the most socially viable transit policy that eliminates ideological polarization and maximizes commuter adoption"
    store = RunStore(root_dir=getattr(args, "output_dir", None))

    constraints = {
        "polarization": "<0.35",
        "conflict": "<0.30",
        "adoption": ">0.50",
    }

    manifest = store.create_autonomous_run(
        goal=goal,
        source_files=source_files,
        constraints=constraints,
        project_name="DECIVERSE Hackathon Demo Run",
    )
    run_id = manifest["run_id"]
    store.freeze_source_files(run_id, source_files)

    provider_mgr = ProviderFailoverManager(primary_provider="ollama", fallback_chain=["ollama", "claude-cli", "codex-cli"])
    disruption_engine = DisruptionEngine()

    def _demo_trace(phase: str, message: str, payload: Optional[Dict[str, Any]] = None):
        if not getattr(args, "json", False):
            icons = {
                "GOAL": "🎯",
                "OBSERVE": "🔍",
                "PLAN": "📋",
                "ACTION": "⚡",
                "EVALUATE": "📊",
                "VERIFY": "🔎",
                "VERIFIED": "🎉",
                "FAILED": "⚠️",
                "REPLAN": "🔄",
                "HUMAN": "👤",
                "RECOVERY": "🛡️",
                "FINALIZE": "🏁",
                "UNRESOLVED": "🛑",
            }
            icon = icons.get(phase.upper(), "📌")
            _stderr(f"{icon} [{phase}] {message}")

    state = DecisionState(
        run_id=run_id,
        goal=goal,
        problem_statement=goal,
        source_files=source_files,
        constraints=constraints,
        max_iterations=8,
        primary_provider=provider_mgr.primary_provider,
        active_provider=provider_mgr.active_provider,
    )

    controller = AutonomousDecisionController(
        state=state,
        provider_manager=provider_mgr,
        disruption_engine=disruption_engine,
        on_trace_callback=_demo_trace,
    )

    _stderr("\n" + "=" * 80)
    _stderr("🌐 DECIVERSE AUTONOMOUS DECISION AGENT — LIVE CLOSED-LOOP DEMO")
    _stderr("=" * 80 + "\n")
    _demo_trace("GOAL", f"Objective: {goal}")
    _demo_trace("GOAL", f"Target Constraints: {', '.join(f'{k} {v}' for k, v in constraints.items())}")

    # Step 1: Initialize run & observe context
    controller.initialize_run(
        goal=goal,
        problem_statement=goal,
        source_files=source_files,
        constraints=constraints,
        max_iterations=8,
    )

    # Step 2: Formulate baseline and simulate
    _stderr("\n--- [Phase 1/4: Initial Planning & Social Simulation] ---")
    # Step 1: Observe context
    controller.execute_next_step()
    # Step 2: Create baseline scenario
    controller.execute_next_step()
    # Step 3: Simulate baseline and verify (fails strict constraints)
    controller.execute_next_step()

    # Step 3: Human-in-the-Loop guidance on tradeoff
    _stderr("\n--- [Phase 2/4: Human-in-the-Loop Policy Guidance] ---")
    human_directive = "Provide tiered student/senior bus discounts and cap peak congestion toll at $3.00."
    _demo_trace("HUMAN", f"Human policymaker rejected baseline due to stakeholder friction and provided directive: '{human_directive}'")
    controller.resume_after_human(approval=False, text_input=human_directive)

    # Step 4: Inject Domain Disruption & Technical Provider Timeout
    _stderr("\n--- [Phase 3/4: Adversarial Disruptions & Provider Failover] ---")
    _demo_trace("ACTION", "Injecting controlled domain disruption: Fuel Price Surge / Merchant Pushback")
    disruption_engine.inject_domain_disruption(DomainDisruptionType.INCREASE_STAKEHOLDER_OPPOSITION)

    _demo_trace("ACTION", "Injecting technical failure: Primary LLM Provider (ollama) Timeout")
    provider_mgr.inject_provider_failure("ollama", "Connection timeout after 30s")

    # Step 5: Resume closed loop until verification
    _stderr("\n--- [Phase 4/4: Adaptive Replanning & Final Verification] ---")
    controller.run_until_completion(max_steps=10)

    # Persist all artifacts including demo_trace.md
    saved_artifacts = store.persist_autonomous_state(run_id, state.to_dict())

    _stderr("\n" + "=" * 80)
    _stderr("🏁 DEMO EXECUTION COMPLETE — AUDIT EVIDENCE PERSISTED")
    _stderr("=" * 80)
    _stderr(f"📄 Run Directory: {store.run_dir(run_id)}")
    _stderr(f"📄 Human-Readable Trace: {saved_artifacts.get('demo_trace')}")
    _stderr(f"📊 Final Policy: {state.best_scenario_name} (Viability: {state.best_score:.2f})")
    _stderr(f"🟢 Final Status: {state.status.value.upper()}\n")

    return {
        "run_id": run_id,
        "status": state.status.value,
        "goal": state.goal,
        "best_scenario_id": state.best_scenario_id,
        "best_scenario_name": state.best_scenario_name,
        "best_score": state.best_score,
        "final_outcome_summary": state.final_outcome_summary,
        "iterations_completed": state.iteration_count,
        "decisions_count": len(state.decision_history),
        "tool_events_count": len(state.tool_events),
        "verification_history": [v.to_dict() for v in state.verification_history],
        "artifacts": saved_artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deciverse", description="Minimal run-first CLI for DECIVERSE")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full workflow and persist artifacts")
    run_parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more source files (pdf/md/txt) used to ground the ontology and profiles",
    )
    run_parser.add_argument(
        "--requirement",
        required=True,
        help="Plain-English simulation requirement (e.g. 'How would voters react to X?')",
    )
    run_parser.add_argument("--platform", choices=("parallel", "twitter", "reddit"), default="parallel")
    run_parser.add_argument("--max-rounds", type=int)
    run_parser.add_argument("--agent-count", type=int, help="Number of agents to simulate (5-500)")
    run_parser.add_argument("--wait", action="store_true", help="Accepted for consistency; end-to-end run waits by default")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--json", action="store_true")

    # Autonomous Run Command
    auto_parser = subparsers.add_parser("autonomous-run", help="Execute closed-loop autonomous decision agent")
    auto_parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Source documents grounding the community decision problem",
    )
    auto_parser.add_argument(
        "--goal",
        required=True,
        help="High-level objective / policy goal (e.g. 'Find the most socially viable transport policy')",
    )
    auto_parser.add_argument(
        "--constraints",
        help="Comma-separated constraint thresholds (e.g. 'polarization<0.35,conflict<0.30,adoption>0.60')",
    )
    auto_parser.add_argument("--project-name", help="Optional project label")
    auto_parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iteration budget")
    auto_parser.add_argument("--provider", choices=("ollama", "claude-cli", "codex-cli"), help="Primary LLM provider")
    auto_parser.add_argument("--inject-domain-disruption", help="Controlled domain disruption for demo")
    auto_parser.add_argument("--inject-technical-disruption", help="Controlled technical disruption for demo")
    auto_parser.add_argument("--output-dir", help="Run storage directory")
    auto_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # Autonomous Demo Command
    demo_parser = subparsers.add_parser(
        "autonomous-demo",
        help="Run real, deterministic closed-loop demo showcasing planning, human-in-the-loop, disruptions & failover",
    )
    demo_parser.add_argument(
        "--files",
        nargs="*",
        default=["demo_transit_policy.md"],
        help="Source documents for demo",
    )
    demo_parser.add_argument(
        "--goal",
        default="Find the most socially viable transit policy that eliminates ideological polarization and maximizes commuter adoption",
        help="High-level policy goal",
    )
    demo_parser.add_argument("--output-dir", help="Run storage directory")
    demo_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")


    runs_parser = subparsers.add_parser("runs", help="Inspect persisted runs")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List run manifests")
    runs_list.add_argument("--limit", type=int, default=20)
    runs_list.add_argument("--output-dir")
    runs_list.add_argument("--json", action="store_true")
    runs_status = runs_subparsers.add_parser("status", help="Show run status")
    runs_status.add_argument("run_id")
    runs_status.add_argument("--output-dir")
    runs_status.add_argument("--json", action="store_true")
    runs_export = runs_subparsers.add_parser("export", help="Resolve artifact paths for a run")
    runs_export.add_argument("run_id")
    runs_export.add_argument("--artifact")
    runs_export.add_argument("--output-dir")
    runs_export.add_argument("--json", action="store_true")

    subparsers.add_parser(
        "doctor",
        help="Run environment/config diagnostics (LLM_PROVIDER, provider binary, .env)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `doctor` must always run — it exists precisely to diagnose bad config.
    if args.command == "doctor":
        return cmd_doctor()

    config_errors = Config.validate()
    if config_errors:
        for err in config_errors:
            _stderr(f"config error: {err}")
        _stderr("hint: run `deciverse doctor` for full diagnostics")
        return 1

    try:
        payload = _handle_command(args)
        return _emit(payload, getattr(args, "json", False))
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            _stderr(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
