"""Streamlit UI for DECIVERSE - AI-Powered Social Decision Simulator."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import json
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
from app.core.workbench_session import WorkbenchSession
from app.core.task_manager import TaskManager, TaskStatus
from app.run_artifacts import RunStore
from app.services.simulation_runner import SimulationRunner, RunnerStatus
from app.services.simulation_manager import SimulationManager
from app.visual_snapshots import generate_visual_snapshots
from app.research import (
    ScenarioManager,
    EmergentBehaviourAnalyzer,
    SocialImpactModel,
    DecisionComparisonEngine,
    DecisionIntelligenceEngine,
)
from app.agent import (
    AutonomousDecisionController,
    DecisionState,
    AgentStatus,
    ProviderFailoverManager,
    DisruptionEngine,
    DomainDisruptionType,
    TechnicalDisruptionType,
    HumanInteractionManager,
    Verifier,
)


# Page configuration
st.set_page_config(
    page_title="DECIVERSE — Autonomous Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: bold;
        color: #1D1D1D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E6DED2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .trace-card {
        padding: 0.8rem;
        border-radius: 0.4rem;
        margin-bottom: 0.6rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .status-running {
        color: #E36414;
    }
    .status-completed {
        color: #6A994E;
    }
    .status-failed {
        color: #BC4749;
    }
    .status-pending {
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'autonomous_agent'
    if 'selected_run_id' not in st.session_state:
        st.session_state.selected_run_id = None
    if 'active_run_id' not in st.session_state:
        st.session_state.active_run_id = None
    if 'pipeline_stage' not in st.session_state:
        st.session_state.pipeline_stage = None
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []  # Store temporary file paths for cleanup
    if 'scenario_manager' not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()
    if 'current_problem_id' not in st.session_state:
        st.session_state.current_problem_id = None
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = None
    if 'agent_controller' not in st.session_state:
        st.session_state.agent_controller = None
    if 'agent_traces' not in st.session_state:
        st.session_state.agent_traces = []
    if 'disruption_engine' not in st.session_state:
        st.session_state.disruption_engine = DisruptionEngine()
    if 'provider_manager' not in st.session_state:
        st.session_state.provider_manager = ProviderFailoverManager()


init_session_state()


# Navigation
def render_navigation():
    """Render sidebar navigation."""
    with st.sidebar:
        st.title("🌐 DECIVERSE")
        st.caption("Agentic AI Social Decision Simulator")
        st.markdown("---")
        
        # Prominently feature the Autonomous Agent page
        if st.button("🤖 Autonomous Decision Agent", key="nav_autonomous_agent", type="primary", use_container_width=True):
            st.session_state.current_page = "autonomous_agent"
            st.rerun()
            
        st.markdown("---")
        st.caption("Simulator Research Modules")
        pages = [
            ("Dashboard", "dashboard"),
            ("Community Problem", "community_problem"),
            ("Stakeholder Config", "stakeholder_config"),
            ("Candidate Decisions", "candidate_decisions"),
            ("Simulation Parameters", "simulation_params"),
            ("Run Simulation", "run_simulation"),
            ("Emergent Behaviour", "emergent_behaviour"),
            ("Social Impact", "social_impact"),
            ("Scenario Comparison", "scenario_comparison"),
            ("Recommendation", "recommendation"),
            ("Run History", "history"),
            ("Explore Run", "explore_run"),
            ("About", "about"),
        ]
        
        for label, page_id in pages:
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown("---")
        
        # Configuration info
        st.subheader("LLM Provider & Health")
        p_mgr: ProviderFailoverManager = st.session_state.provider_manager
        active_p = p_mgr.active_provider
        status_map = p_mgr.get_status_summary()
        
        for p_name, p_stat in status_map.items():
            icon = "🟢" if p_stat == "available" else ("🟡" if p_stat == "degraded" else "🔴")
            active_badge = " *(Active)*" if p_name == active_p else ""
            st.text(f"{icon} {p_name}: {p_stat}{active_badge}")
        
        st.text(f"Data: {Config.DATA_DIR}")






# Dashboard page
def render_dashboard():
    """Render dashboard page."""
    st.markdown('<div class="main-header">DECIVERSE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Social Decision Simulator</div>', unsafe_allow_html=True)
    
    st.markdown("Turn real-world evidence into a simulated social world for decision support.")
    
    # Quick action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 New Decision Analysis", type="primary", use_container_width=True):
            st.session_state.current_page = 'community_problem'
            st.rerun()
    with col2:
        if st.button("📊 View Recent Runs", use_container_width=True):
            st.session_state.current_page = 'history'
            st.rerun()
    
    st.markdown("---")
    
    # Research pipeline visualization
    st.subheader("Research Pipeline")
    
    pipeline_steps = [
        ("Community Problem", "�"),
        ("Community Context", "🏛️"),
        ("Stakeholder Digital Twin", "�"),
        ("Scenario Designer", "🎯"),
        ("Social Interaction Core", "🎭"),
        ("Emergent Behaviour", "📊"),
        ("Social Impact", "⚖️"),
        ("Decision Comparison", "🔀"),
        ("Decision Intelligence", "🧠"),
    ]
    
    cols = st.columns(len(pipeline_steps))
    for i, (label, icon) in enumerate(pipeline_steps):
        with cols[i]:
            st.markdown(f"<div style='text-align: center; padding: 0.5rem; font-size: 0.8rem;'>"
                       f"{icon}<br><small>{label}</small></div>", 
                       unsafe_allow_html=True)
            if i < len(pipeline_steps) - 1:
                st.markdown("↓", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent runs
    st.subheader("Recent Decision Analyses")
    store = RunStore()
    runs = store.list(limit=5)
    
    if runs:
        for run in runs:
            run_id = run.get('run_id', '')
            status = run.get('status', 'unknown')
            created_at = run.get('created_at', '')
            artifact_count = len(run.get('artifacts', {}))
            
            # Format status
            status_class = f"status-{status}"
            status_display = status.replace('_', ' ').title()
            
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            col1.text(run_id[:12] + "...")
            col2.markdown(f"<span class='{status_class}'>{status_display}</span>", unsafe_allow_html=True)
            col3.text(created_at[:19] if created_at else "")
            col4.text(f"{artifact_count} artifacts")
            
            if col5.button("View", key=f"view_{run_id}"):
                st.session_state.selected_run_id = run_id
                st.session_state.current_page = 'explore_run'
                st.rerun()
    else:
        st.info("No decision analyses yet. Start by defining a community problem!")


# New simulation page
def render_new_simulation():
    """Render new simulation page."""
    st.markdown('<div class="main-header">New Simulation</div>', unsafe_allow_html=True)
    
    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, MD, TXT)",
        type=['pdf', 'md', 'txt'],
        accept_multiple_files=True,
        help="Upload documents to ground the ontology and agent profiles"
    )
    
    # Requirement
    st.subheader("Simulation Requirement")
    requirement = st.text_area(
        "What do you want to simulate?",
        placeholder="e.g., What would happen if the city introduced fare-free weekend buses?",
        height=100,
        help="Describe the scenario you want to simulate"
    )
    
    # Configuration
    with st.expander("Configuration", expanded=False):
        st.subheader("Platform")
        platform = st.selectbox(
            "Social Platform",
            ["parallel", "twitter", "reddit"],
            help="Choose which social media platform to simulate"
        )
        
        st.subheader("Simulation Parameters")
        max_rounds = st.number_input(
            "Max Rounds",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum number of simulation rounds"
        )
        
        parallel_profile_count = st.number_input(
            "Parallel Profile Generation",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of agent profiles to generate in parallel"
        )
        
        agent_count = st.number_input(
            "Number of Agents to Simulate",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
            help="Number of agents to launch in the simulation (5-500)"
        )
    
    # Validation
    can_run = uploaded_files and requirement and len(requirement.strip()) > 0
    
    # Agent count validation (warning only, don't block)
    if agent_count < 5:
        st.warning("Minimum 5 agents recommended for meaningful simulation")
    elif agent_count > 500:
        st.warning("Maximum 500 agents recommended for performance reasons")
    
    # Run button
    if st.button("🚀 Run Simulation", type="primary", disabled=not can_run, use_container_width=True):
        if not can_run:
            st.error("Please upload documents and enter a requirement")
        else:
            # Start simulation
            with st.spinner("Starting simulation..."):
                try:
                    # Create temporary files from uploads
                    temp_file_paths = []
                    
                    for uploaded_file in uploaded_files:
                        # Create temporary file
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
                        temp_file.write(uploaded_file.read())
                        temp_file.close()
                        temp_file_paths.append(temp_file.name)
                        st.session_state.temp_files.append(temp_file.name)
                    
                    # Create run
                    source_files = temp_file_paths
                    project_name = uploaded_files[0].name.rsplit('.', 1)[0]
                    
                    store = RunStore()
                    manifest = store.create(requirement, source_files, project_name=project_name)
                    run_id = manifest["run_id"]
                    
                    st.session_state.active_run_id = run_id
                    st.session_state.pipeline_stage = 'starting'
                    
                    # Store configuration in session state
                    st.session_state.sim_config = {
                        'requirement': requirement,
                        'platform': platform,
                        'max_rounds': max_rounds,
                        'parallel_profile_count': parallel_profile_count,
                        'agent_count': agent_count,
                        'source_files': source_files,
                        'project_name': project_name
                    }
                    
                    st.success(f"Simulation started! Run ID: {run_id}")
                    st.session_state.current_page = 'pipeline_progress'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to start simulation: {str(e)}")
                    # Cleanup temp files
                    for temp_path in temp_file_paths:
                        try:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                        except:
                            pass


# Pipeline progress page
def render_pipeline_progress():
    """Render pipeline progress page."""
    run_id = st.session_state.active_run_id
    if not run_id:
        st.error("No active run")
        st.session_state.current_page = 'dashboard'
        st.rerun()
        return
    
    st.markdown('<div class="main-header">Pipeline Progress</div>', unsafe_allow_html=True)
    st.info(f"Run ID: {run_id}")
    
    store = RunStore()
    manifest = store.load(run_id)
    status = manifest.get('status', 'unknown')
    
    # Check if run is already completed or failed
    if status in ['completed', 'failed']:
        st.session_state.current_page = 'explore_run'
        st.session_state.selected_run_id = run_id
        st.rerun()
        return
    
    # Progress display
    progress_container = st.container()
    
    with progress_container:
        # Create status placeholders for each step
        steps = ['ontology', 'graph', 'profiles', 'simulation', 'report', 'visuals']
        step_status = {}
        
        for step in steps:
            step_status[step] = st.empty()
        
        # Overall progress
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Execute pipeline
    try:
        config = st.session_state.sim_config
        
        # Create session
        session = WorkbenchSession.open(metadata={"entrypoint": "streamlit", "run_id": run_id})
        
        # Save input files
        store.write_text(run_id, "input/requirement.txt", config['requirement'])
        store.record_artifact(run_id, "requirement", "input/requirement.txt")
        store.freeze_source_files(run_id, config['source_files'])
        store.record_artifact(run_id, "source_files_dir", "input/source_files")
        
        # --- Ontology ---
        step_status['ontology'].info("🔍 Generating ontology...")
        progress_bar.progress(10)
        status_text.text("Processing documents and generating ontology...")
        
        # Create proper file inputs for the backend
        class LocalFileInput:
            """Minimal file wrapper compatible with ProjectManager.save_file_to_project."""
            def __init__(self, path: str):
                self.path = os.path.abspath(path)
                self.filename = os.path.basename(path)
            def save(self, destination: str) -> None:
                shutil.copy2(self.path, destination)
        
        file_inputs = [LocalFileInput(f) for f in config['source_files']]
        
        project_result = session.generate_ontology(
            simulation_requirement=config['requirement'],
            uploaded_files=file_inputs,
            project_name=config['project_name'],
        )
        ontology = project_result.get("ontology", {})
        n_entity_types = len(ontology.get("entity_types", []))
        n_edge_types = len(ontology.get("relationship_types", ontology.get("edge_types", [])))
        
        step_status['ontology'].success(f"✅ Ontology: {n_entity_types} entity types, {n_edge_types} edge types")
        store.update(run_id, project_id=project_result["project_id"], status="graph_building", task_progress=20, task_message="Ontology generated")
        store.write_json(run_id, "input/ontology.json", ontology)
        store.record_artifact(run_id, "ontology", "input/ontology.json")
        
        progress_bar.progress(20)
        
        # --- Graph ---
        step_status['graph'].info("🕸️ Building knowledge graph...")
        status_text.text("Building knowledge graph from ontology...")
        
        graph_result = session.start_graph_build(project_id=project_result["project_id"])
        store.update(run_id, graph_build_task_id=graph_result["task_id"], status="graph_building")
        
        # Wait for graph build
        task_manager = TaskManager()
        while True:
            task = task_manager.get_task(graph_result["task_id"])
            if task is None:
                raise RuntimeError(f"Task not found: {graph_result['task_id']}")
            
            if task.status == TaskStatus.COMPLETED:
                break
            if task.status == TaskStatus.FAILED:
                raise RuntimeError(task.error or task.message or f"Task failed: {graph_result['task_id']}")
            
            progress = 20 + (task.progress * 0.3)  # 20-50% range
            progress_bar.progress(int(progress))
            status_text.text(f"Building graph: {task.message}")
            step_status['graph'].info(f"🕸️ Building graph: {task.progress}% - {task.message}")
            time.sleep(1)
        
        graph_id = (task.result or {}).get("graph_id")
        if not graph_id:
            raise RuntimeError("Graph build completed without a graph_id")
        
        from app.services.graph_builder import GraphBuilderService
        from app.services.graph_db import GraphDatabase
        
        graph_builder = GraphBuilderService()
        graph_db = GraphDatabase()
        graph_data = graph_builder.get_graph_data(graph_id)
        graph_stats = graph_db.get_graph_statistics(graph_id)
        n_nodes = graph_stats.get("node_count", 0)
        n_edges = graph_stats.get("edge_count", 0)
        
        step_status['graph'].success(f"✅ Graph: {n_nodes} nodes, {n_edges} edges")
        store.update(run_id, graph_id=graph_id, status="graph_ready", task_progress=50, task_message="Graph ready")
        progress_bar.progress(50)
        
        # --- Profiles ---
        step_status['profiles'].info("👥 Generating agent profiles...")
        status_text.text("Generating agent profiles...")
        
        enable_twitter = config['platform'] in {"parallel", "twitter"}
        enable_reddit = config['platform'] in {"parallel", "reddit"}
        
        simulation_state = session.create_simulation(
            project_id=project_result["project_id"],
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
        )
        simulation_id = simulation_state.simulation_id
        store.update(run_id, simulation_id=simulation_id, status="simulation_preparing", task_progress=50, task_message="Simulation created")
        
        prepare_result = session.start_simulation_preparation(
            simulation_id=simulation_id,
            use_llm_for_profiles=True,
            parallel_profile_count=config['parallel_profile_count'],
            agent_count=config.get('agent_count', None),
        )
        
        if prepare_result.get("task_id"):
            store.update(run_id, prepare_task_id=prepare_result["task_id"], status="simulation_preparing")
            
            while True:
                task = task_manager.get_task(prepare_result["task_id"])
                if task is None:
                    raise RuntimeError(f"Task not found: {prepare_result['task_id']}")
                
                if task.status == TaskStatus.COMPLETED:
                    break
                if task.status == TaskStatus.FAILED:
                    raise RuntimeError(task.error or task.message or f"Task failed: {prepare_result['task_id']}")
                
                progress = 50 + (task.progress * 0.2)  # 50-70% range
                progress_bar.progress(int(progress))
                status_text.text(f"Generating profiles: {task.message}")
                step_status['profiles'].info(f"👥 Generating profiles: {task.progress}% - {task.message}")
                time.sleep(1)
        
        sim_dir = os.path.join(SimulationManager.SIMULATION_DATA_DIR, simulation_id)
        agent_count = 0
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                agent_count = len(json.load(f).get("agent_configs", []))
        
        # Display requested vs actual agent count
        requested_count = config.get('agent_count', 'auto')
        if requested_count != 'auto':
            step_status['profiles'].success(f"✅ Profiles: {agent_count} agents (requested: {requested_count})")
        else:
            step_status['profiles'].success(f"✅ Profiles: {agent_count} agents")
        store.update(run_id, status="simulation_ready", task_progress=70, task_message="Simulation ready")
        progress_bar.progress(70)
        
        # --- Simulation ---
        step_status['simulation'].info("🎭 Running social simulation...")
        status_text.text("Running OASIS social simulation...")
        
        session.start_simulation_run(
            simulation_id=simulation_id,
            platform=config['platform'],
            max_rounds=config['max_rounds'],
            enable_graph_memory_update=False,
        )
        
        # Wait for simulation
        while True:
            state = SimulationRunner.get_run_state(simulation_id)
            if state is None:
                raise RuntimeError(f"Simulation run state not found: {simulation_id}")
            
            if state.runner_status == RunnerStatus.COMPLETED:
                break
            if state.runner_status == RunnerStatus.FAILED:
                raise RuntimeError(state.error or f"Simulation failed: {simulation_id}")
            
            progress = 70 + (state.to_dict().get("progress_percent", 0) * 0.25)  # 70-95% range
            progress_bar.progress(int(progress))
            status_text.text(f"Simulation: round {state.current_round}/{state.total_rounds}")
            step_status['simulation'].info(f"🎭 Simulation: round {state.current_round}/{state.total_rounds} ({state.twitter_actions_count + state.reddit_actions_count} actions)")
            time.sleep(2)
        
        timeline = SimulationRunner.get_timeline(simulation_id)
        agent_stats = SimulationRunner.get_agent_stats(simulation_id)
        actions = SimulationRunner.get_all_actions(simulation_id)
        total_actions = sum(item.get("total_actions", 0) for item in timeline)
        
        step_status['simulation'].success(f"✅ Simulation: {len(timeline)} rounds, {total_actions} actions")
        store.update(run_id, status="simulation_completed", task_progress=95, task_message="Simulation completed")
        progress_bar.progress(95)
        
        # --- Report ---
        step_status['report'].info("📝 Generating report...")
        status_text.text("Generating analysis report...")
        
        report_payload = None
        report_markdown = ""
        
        from app.resources.reports import ReportStore
        report_store = ReportStore()
        
        report_result = session.start_report_generation(simulation_id=simulation_id)
        report_id = report_result.get("report_id")
        
        if report_result.get("task_id"):
            store.update(run_id, report_id=report_id, report_task_id=report_result["task_id"], status="report_generating")
            
            while True:
                task = task_manager.get_task(report_result["task_id"])
                if task is None:
                    raise RuntimeError(f"Task not found: {report_result['task_id']}")
                
                if task.status == TaskStatus.COMPLETED:
                    break
                if task.status == TaskStatus.FAILED:
                    raise RuntimeError(task.error or task.message or f"Task failed: {report_result['task_id']}")
                
                progress = 95 + (task.progress * 0.05)  # 95-100% range
                progress_bar.progress(int(progress))
                status_text.text(f"Generating report: {task.message}")
                step_status['report'].info(f"📝 Generating report: {task.progress}% - {task.message}")
                time.sleep(1)
            
            report_id = (task.result or {}).get("report_id", report_id)
        
        if report_id:
            report = report_store.get(report_id)
            if report is not None:
                report_payload = report.to_dict()
                report_markdown = report.markdown_content
        
        step_status['report'].success("✅ Report generated")
        
        # --- Visuals ---
        step_status['visuals'].info("🎨 Generating visualizations...")
        status_text.text("Generating visual snapshots...")
        
        final_manifest = store.update(run_id, report_id=report_id, status="completed", task_progress=100, task_message="Run completed")
        
        # Collect outputs
        final_manifest = collect_run_outputs(
            store=store,
            manifest=final_manifest,
            graph_data=graph_data,
            graph_stats=graph_stats,
            timeline=timeline,
            agent_stats=agent_stats,
            actions=actions,
            report_payload=report_payload,
            report_markdown=report_markdown,
            run_id=run_id
        )
        
        visual_keys = {"swarm_overview", "cluster_map", "timeline", "platform_split"}
        n_visuals = sum(1 for k in final_manifest.get("artifacts", {}) if k in visual_keys)
        
        step_status['visuals'].success(f"✅ Visuals: {n_visuals} snapshots")
        progress_bar.progress(100)
        status_text.text("Pipeline completed successfully!")
        
        st.success(f"Simulation completed successfully! Run ID: {run_id}")
        
        # Cleanup temp files
        for temp_path in st.session_state.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
        st.session_state.temp_files = []
        
        # Navigate to results
        time.sleep(2)
        st.session_state.selected_run_id = run_id
        st.session_state.current_page = 'explore_run'
        st.session_state.active_run_id = None
        st.rerun()
        
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        store.update(run_id, status="failed", error=str(e), task_message=str(e))
        
        # Cleanup temp files
        for temp_path in st.session_state.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
        st.session_state.temp_files = []


def _generate_verdict(report_markdown: str, requirement: str) -> Dict[str, Any]:
    """Generate a machine-readable verdict from the report for agent consumption."""
    from app.utils.llm_client import LLMClient

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
        return {
            "prediction": "Verdict generation failed",
            "confidence": 0.0,
            "key_dynamics": [],
            "signals": [],
        }


def _top_agents(agent_stats: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Get top agents by activity."""
    ordered = sorted(agent_stats, key=lambda item: item.get("total_actions", 0), reverse=True)
    return ordered[:limit]


def _write_action_log(output_path: str, actions: List[Any]) -> str:
    """Write action log to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ordered = sorted(actions, key=lambda action: action.timestamp)
    with open(output_path, "w", encoding="utf-8") as handle:
        for action in ordered:
            handle.write(json.dumps(action.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def _simulation_dir(simulation_id: str) -> str:
    """Get simulation directory."""
    return os.path.join(SimulationManager.SIMULATION_DATA_DIR, simulation_id)


def _record_if_copied(store: RunStore, run_id: str, key: str, source_path: str, rel_path: str) -> None:
    """Record file if successfully copied."""
    copied_path = store.copy_file(run_id, source_path, rel_path)
    if copied_path:
        store.record_artifact(run_id, key, rel_path)


def collect_run_outputs(store, manifest, graph_data, graph_stats, timeline, agent_stats, actions, report_payload, report_markdown, run_id):
    """Collect and store run outputs (adapted from CLI)."""
    
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
    
    # Generate verdict
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


# Explore run page
def render_explore_run():
    """Render run exploration page."""
    run_id = st.session_state.selected_run_id
    if not run_id:
        st.error("No run selected")
        st.session_state.current_page = 'history'
        st.rerun()
        return
    
    store = RunStore()
    try:
        manifest = store.load(run_id)
    except FileNotFoundError:
        st.error(f"Run not found: {run_id}")
        st.session_state.selected_run_id = None
        st.session_state.current_page = 'history'
        st.rerun()
        return
    
    st.markdown('<div class="main-header">Explore Run</div>', unsafe_allow_html=True)
    st.info(f"Run ID: {run_id}")
    
    # Status
    status = manifest.get('status', 'unknown')
    status_class = f"status-{status}"
    status_display = status.replace('_', ' ').title()
    st.markdown(f"**Status:** <span class='{status_class}'>{status_display}</span>", unsafe_allow_html=True)
    
    # Tabs
    tabs = st.tabs(["Overview", "Verdict", "Report", "Visuals", "Artifacts"])
    
    with tabs[0]:
        render_overview_tab(manifest, store)
    
    with tabs[1]:
        render_verdict_tab(manifest, store)
    
    with tabs[2]:
        render_report_tab(manifest, store)
    
    with tabs[3]:
        render_visuals_tab(manifest, store)
    
    with tabs[4]:
        render_artifacts_tab(manifest, store)


def render_overview_tab(manifest, store):
    """Render overview tab."""
    st.subheader("Run Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    run_id = manifest.get('run_id', '')
    created_at = manifest.get('created_at', '')
    status = manifest.get('status', 'unknown')
    artifacts = manifest.get('artifacts', {})
    
    col1.metric("Run ID", run_id[:12] + "..." if len(run_id) > 12 else run_id)
    col2.metric("Status", status.replace('_', ' ').title())
    col3.metric("Created", created_at[:19] if created_at else "N/A")
    col4.metric("Artifacts", len(artifacts))
    
    st.markdown("---")
    
    # Try to load summary
    try:
        summary_path = os.path.join(store.run_dir(run_id), "report/summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            st.subheader("Simulation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nodes", summary.get('node_count', 0))
            col2.metric("Edges", summary.get('edge_count', 0))
            col3.metric("Rounds", summary.get('rounds', 0))
            col4.metric("Actions", summary.get('total_actions', 0))
            
            # Display agent count
            top_agents = summary.get('top_agents', [])
            if top_agents:
                st.markdown(f"**Agents Generated:** {len(top_agents)}")
                st.markdown(f"*Top {min(10, len(top_agents))} most active agents:*")
                
                agent_data = []
                for agent in top_agents[:10]:
                    agent_data.append({
                        "Agent ID": agent.get("agent_id", "N/A"),
                        "Name": agent.get("agent_name", "Unknown"),
                        "Actions": agent.get("total_actions", 0)
                    })
                
                st.dataframe(agent_data, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load summary: {str(e)}")


def render_verdict_tab(manifest, store):
    """Render verdict tab."""
    st.subheader("Simulation Verdict")
    
    try:
        verdict_path = os.path.join(store.run_dir(manifest['run_id']), "report/verdict.json")
        if os.path.exists(verdict_path):
            with open(verdict_path, 'r') as f:
                verdict = json.load(f)
            
            # Prediction
            st.markdown("### Prediction")
            st.write(verdict.get('prediction', 'No prediction available'))
            
            # Confidence
            confidence = verdict.get('confidence', 0.0)
            st.markdown(f"### Confidence: {confidence * 100:.1f}%")
            st.progress(confidence)
            
            # Key dynamics
            key_dynamics = verdict.get('key_dynamics', [])
            if key_dynamics:
                st.markdown("### Key Dynamics")
                for dynamic in key_dynamics:
                    st.markdown(f"• {dynamic}")
            
            # Signals
            signals = verdict.get('signals', [])
            if signals:
                st.markdown("### Signals")
                for signal in signals:
                    direction = signal.get('direction', 'unknown')
                    strength = signal.get('strength', 0.0)
                    signal_text = signal.get('signal', '')
                    emoji = "📈" if direction == "positive" else "📉" if direction == "negative" else "➡️"
                    st.markdown(f"{emoji} **{signal_text}** ({direction}, strength: {strength:.2f})")
        else:
            st.info("No verdict available for this run.")
    except Exception as e:
        st.error(f"Could not load verdict: {str(e)}")


def render_report_tab(manifest, store):
    """Render report tab."""
    st.subheader("Analysis Report")
    
    try:
        report_path = os.path.join(store.run_dir(manifest['run_id']), "report/report.md")
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.markdown(report_content)
        else:
            st.info("No report available for this run.")
    except Exception as e:
        st.error(f"Could not load report: {str(e)}")


def render_visuals_tab(manifest, store):
    """Render visuals tab."""
    st.subheader("Visual Snapshots")
    
    run_dir = store.run_dir(manifest['run_id'])
    visuals_dir = os.path.join(run_dir, "visuals")
    
    if not os.path.exists(visuals_dir):
        st.info("No visualizations available for this run.")
        return
    
    visual_files = [f for f in os.listdir(visuals_dir) if f.endswith('.svg')]
    
    if not visual_files:
        st.info("No visualizations available for this run.")
        return
    
    for visual_file in visual_files:
        st.markdown(f"### {visual_file.replace('.svg', '').replace('_', ' ').title()}")
        try:
            # Read SVG file
            with open(os.path.join(visuals_dir, visual_file), 'r', encoding='utf-8') as f:
                svg_content = f.read()
            st.markdown(svg_content, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not load {visual_file}: {str(e)}")


def render_artifacts_tab(manifest, store):
    """Render artifacts tab."""
    st.subheader("Run Artifacts")
    
    artifacts = manifest.get('artifacts', {})
    run_dir = store.run_dir(manifest['run_id'])
    
    if not artifacts:
        st.info("No artifacts available for this run.")
        return
    
    for key, rel_path in artifacts.items():
        full_path = os.path.join(run_dir, rel_path)
        
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.text(key)
        col2.text(rel_path)
        
        if os.path.exists(full_path):
            if col3.button("Download", key=f"download_{key}"):
                with open(full_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {key}",
                        data=f.read(),
                        file_name=os.path.basename(rel_path),
                        mime="application/octet-stream"
                    )
        else:
            col3.text("Missing")


# Run history page
def render_history():
    """Render run history page."""
    st.markdown('<div class="main-header">Run History</div>', unsafe_allow_html=True)
    
    store = RunStore()
    runs = store.list(limit=50)
    
    if not runs:
        st.info("No runs yet. Create your first simulation!")
        return
    
    # Search/filter
    search = st.text_input("Search runs by ID", placeholder="Enter run ID...")
    
    if search:
        runs = [r for r in runs if search.lower() in r.get('run_id', '').lower()]
    
    # Display runs
    for run in runs:
        run_id = run.get('run_id', '')
        status = run.get('status', 'unknown')
        created_at = run.get('created_at', '')
        artifact_count = len(run.get('artifacts', {}))
        project_name = run.get('project_name', 'Unnamed')
        
        # Format status
        status_class = f"status-{status}"
        status_display = status.replace('_', ' ').title()
        
        with st.expander(f"{run_id[:12]}... - {project_name} ({status_display})"):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            col1.text(f"Run ID: {run_id}")
            col2.markdown(f"<span class='{status_class}'>{status_display}</span>", unsafe_allow_html=True)
            col3.text(f"Created: {created_at[:19] if created_at else 'N/A'}")
            col4.text(f"Artifacts: {artifact_count}")
            
            if col5.button("View", key=f"view_history_{run_id}"):
                st.session_state.selected_run_id = run_id
                st.session_state.current_page = 'explore_run'
                st.rerun()


# About page
def render_about():
    """Render about page."""
    st.markdown('<div class="main-header">About DECIVERSE</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## What is DECIVERSE?
    
    DECIVERSE is an AI-powered social decision simulator that transforms 
    real-world evidence into simulated social worlds for decision support. It helps 
    communities and decision-makers explore possible outcomes of different interventions 
    by simulating heterogeneous stakeholder reactions and emergent social consequences.
    
    ## Research Contribution
    
    DECIVERSE builds on foundational technologies (MiroFish and OASIS) to provide:
    
    - **Community Digital Twin**: Structured representation of community context and stakeholders
    - **Scenario-Based Decision Simulation**: Evaluate multiple candidate decisions
    - **Emergent Behaviour Analysis**: Identify community-level behaviour patterns
    - **Social Impact Evaluation**: Quantify acceptance, consensus, polarization, and other metrics
    - **Multi-Scenario Decision Comparison**: Compare alternative decisions side-by-side
    - **Decision Intelligence**: Evidence-based recommendations with uncertainty quantification
    
    ## Research Pipeline
    
    1. **Community Problem**: Define the decision problem and context
    2. **Community Context Engine**: Extract entities, relationships, and stakeholder information
    3. **Stakeholder Digital Twin**: Generate heterogeneous agent personas
    4. **Scenario Designer**: Define candidate decisions/interventions
    5. **Social Interaction Core**: Run OASIS-powered multi-agent simulations
    6. **Emergent Behaviour Analyzer**: Derive community-level behaviour patterns
    7. **Social Impact Model**: Evaluate social consequences across multiple dimensions
    8. **Decision Comparison Engine**: Compare scenarios across impact metrics
    9. **Decision Intelligence**: Generate comprehensive recommendations
    
    ## Foundational Technologies
    
    - **MiroFish**: Multi-agent social simulation framework
    - **OASIS**: Social media simulation environment (camel-oasis==0.2.5, camel-ai==0.2.78)
    
    ## Configuration
    
    DECIVERSE uses environment variables for configuration:
    
    - `LLM_PROVIDER`: claude-cli, codex-cli, or ollama
    - `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
    - `OLLAMA_MODEL`: Ollama model name (default: qwen3:8b)
    
    ## Important Note
    
    Results are based on simulated agent behaviour and should be interpreted as 
    scenario-analysis evidence for decision support, not as guaranteed real-world predictions.
    Simulated agents are not equivalent to actual community members.
    
    ## License
    
    AGPL-3.0
    """)


# Research workflow pages
def render_community_problem():
    """Render community problem definition page."""
    st.markdown('<div class="main-header">Community Problem</div>', unsafe_allow_html=True)
    st.markdown("Define the community decision problem you want to analyze.")
    
    problem_statement = st.text_area(
        "Problem Statement",
        placeholder="e.g., Should the community introduce a new traffic restriction policy?",
        height=150,
        help="Describe the community problem or decision that needs to be addressed"
    )
    
    st.subheader("Community Context Documents")
    uploaded_files = st.file_uploader(
        "Upload context documents (PDF, MD, TXT)",
        type=['pdf', 'md', 'txt'],
        accept_multiple_files=True,
        help="Upload documents that provide context about the community and the problem"
    )
    
    if st.button("Save Problem & Continue", type="primary"):
        if problem_statement and uploaded_files:
            st.session_state.problem_statement = problem_statement
            st.session_state.uploaded_files = uploaded_files
            
            # Create problem ID
            import uuid as _uuid
            problem_id = f"problem_{_uuid.uuid4().hex[:12]}"
            st.session_state.current_problem_id = problem_id
            
            st.success("Problem saved! Continue to Stakeholder Configuration.")
            st.session_state.current_page = 'stakeholder_config'
            st.rerun()
        else:
            st.error("Please provide a problem statement and upload at least one document")


def render_stakeholder_config():
    """Render stakeholder configuration page."""
    st.markdown('<div class="main-header">Stakeholder Configuration</div>', unsafe_allow_html=True)
    st.markdown("Configure the stakeholder digital twin parameters.")
    
    if 'problem_statement' not in st.session_state:
        st.warning("Please define a community problem first.")
        if st.button("Go to Problem Definition"):
            st.session_state.current_page = 'community_problem'
            st.rerun()
        return
    
    st.info(f"Problem: {st.session_state.problem_statement}")
    
    agent_count = st.number_input(
        "Number of Stakeholders to Simulate",
        min_value=5,
        max_value=500,
        value=50,
        step=5,
        help="Number of simulated community members (5-500)"
    )
    
    st.subheader("Stakeholder Entity Types")
    st.info("Entity types will be extracted from your documents during context analysis.")
    
    if st.button("Continue to Candidate Decisions"):
        st.session_state.agent_count = agent_count
        st.session_state.current_page = 'candidate_decisions'
        st.rerun()


def render_candidate_decisions():
    """Render candidate decisions page."""
    st.markdown('<div class="main-header">Candidate Decisions</div>', unsafe_allow_html=True)
    st.markdown("Define the candidate decisions/interventions to compare.")
    
    if 'current_problem_id' not in st.session_state:
        st.warning("Please define a community problem first.")
        if st.button("Go to Problem Definition"):
            st.session_state.current_page = 'community_problem'
            st.rerun()
        return
    
    st.info(f"Problem ID: {st.session_state.current_problem_id}")
    
    num_decisions = st.number_input(
        "Number of Candidate Decisions",
        min_value=1,
        max_value=5,
        value=2,
        help="How many alternative decisions do you want to compare?"
    )
    
    decisions = []
    for i in range(num_decisions):
        with st.expander(f"Decision {i+1}", expanded=True):
            name = st.text_input(f"Decision Name", key=f"decision_name_{i}")
            description = st.text_area(f"Description", key=f"decision_desc_{i}", height=100)
            intervention = st.text_area(f"Intervention/Action", key=f"decision_intervention_{i}", height=80)
            
            decisions.append({
                'name': name,
                'description': description,
                'intervention': intervention
            })
    
    if st.button("Save Decisions & Continue", type="primary"):
        st.session_state.candidate_decisions = decisions
        st.session_state.current_page = 'simulation_params'
        st.rerun()


def render_simulation_params():
    """Render simulation parameters page."""
    st.markdown('<div class="main-header">Simulation Parameters</div>', unsafe_allow_html=True)
    st.markdown("Configure simulation parameters for all scenarios.")
    
    platform = st.selectbox(
        "Social Platform",
        ["parallel", "twitter", "reddit"],
        help="Choose which social media platform to simulate"
    )
    
    max_rounds = st.number_input(
        "Max Simulation Rounds",
        min_value=1,
        max_value=100,
        value=10,
        help="Maximum number of simulation rounds per scenario"
    )
    
    parallel_profile_count = st.number_input(
        "Parallel Profile Generation",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of agent profiles to generate in parallel"
    )
    
    if st.button("Continue to Run Simulation"):
        st.session_state.platform = platform
        st.session_state.max_rounds = max_rounds
        st.session_state.parallel_profile_count = parallel_profile_count
        st.session_state.current_page = 'run_simulation'
        st.rerun()


def render_run_simulation():
    """Render run simulation page."""
    st.markdown('<div class="main-header">Run Simulation</div>', unsafe_allow_html=True)
    st.markdown("Execute simulations for all candidate decisions.")
    
    if 'candidate_decisions' not in st.session_state:
        st.warning("Please define candidate decisions first.")
        if st.button("Go to Candidate Decisions"):
            st.session_state.current_page = 'candidate_decisions'
            st.rerun()
        return
    
    st.info(f"Running {len(st.session_state.candidate_decisions)} scenario simulations...")
    
    # For now, redirect to the existing new_simulation workflow
    # This will be enhanced to run multiple scenarios
    st.info("Multi-scenario simulation will be implemented. For now, using single simulation workflow.")
    
    if st.button("Launch Single Simulation (Current Workflow)", type="primary"):
        st.session_state.current_page = 'new_simulation'
        st.rerun()


def render_emergent_behaviour():
    """Render emergent behaviour analysis page."""
    st.markdown('<div class="main-header">Emergent Behaviour Analysis</div>', unsafe_allow_html=True)
    st.markdown("Analyze community-level behaviour patterns from simulation results.")
    
    st.info("Emergent behaviour analysis requires completed simulation results.")
    st.info("This page will display:")
    st.markdown("""
    - Consensus metrics
    - Polarization analysis
    - Sentiment patterns
    - Conflict intensity
    - Information diffusion
    - Agent influence distribution
    - Community/group behaviour
    - Adoption and support patterns
    """)


def render_social_impact():
    """Render social impact evaluation page."""
    st.markdown('<div class="main-header">Social Impact Evaluation</div>', unsafe_allow_html=True)
    st.markdown("Evaluate social consequences across multiple dimensions.")
    
    st.info("Social impact evaluation requires completed simulation results.")
    st.info("This page will display:")
    st.markdown("""
    - Acceptance score
    - Consensus score
    - Polarization score
    - Conflict score
    - Equity score
    - Adoption score
    - Stability score
    - Overall social viability score
    - Risk assessment
    """)


def render_scenario_comparison():
    """Render scenario comparison page."""
    st.markdown('<div class="main-header">Scenario Comparison</div>', unsafe_allow_html=True)
    st.markdown("Compare candidate decisions across impact metrics.")
    
    st.info("Scenario comparison requires multiple completed simulations.")
    st.info("This page will display:")
    st.markdown("""
    - Comparison table across all dimensions
    - Rankings by overall score
    - Rankings by acceptance
    - Rankings by consensus
    - Rankings by conflict (low to high)
    - Side-by-side metric visualization
    """)


def render_autonomous_agent():
    """Render the full Autonomous Decision Agent page."""
    st.markdown('<div class="main-header">🤖 Autonomous Decision Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Closed-Loop Swarm-Intelligence Optimizer with Transparent Action-Utility Planning, '
        'SocialImpactModel Verifier, and Resilient Provider Failover</div>',
        unsafe_allow_html=True
    )

    # Architectural Loop Overview banner
    with st.expander("ℹ️ Autonomous Agent Control Architecture & Closed Loop", expanded=False):
        st.markdown("""
        ```text
        GOAL & CONSTRAINTS ──> OBSERVE ──> GENERATE CANDIDATE ACTIONS ──> ACTION-UTILITY RANKING
                                                                                   │
        ┌──────────────────────────────────────────────────────────────────────────┘
        ▼
        DECIDE ──> ACT (Tools) ──> OBSERVE RESULT ──> EVALUATE (Emergent Impact)
                                                                │
        ┌───────────────────────────────────────────────────────┘
        ▼
        VERIFY (SocialImpactModel vs Constraints)
          ├── [VERIFIED]  ──> FINALIZE (Optimal Policy Selected)
          └── [FAILED / UNCERTAIN] ──> REPLAN (Diagnose Deficits ──> Policy Adaptation ──> ACT)
        ```
        """)

    # 1. Inputs & Configuration Layout
    st.subheader("1. Goal, Constraints & Execution Configuration")
    
    col_preset, col_file = st.columns([1, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Demo Scenario Presets",
            [
                "Metro City Transit: Fare-Free Weekend Bus Policy",
                "Downtown Clean Air & Congestion Charge Zone",
                "High-Density Affordable Housing Re-zoning",
                "Custom Goal Definition",
            ],
            index=0,
        )
    
    # Presets mapping
    preset_goals = {
        "Metro City Transit: Fare-Free Weekend Bus Policy": "Find the most socially viable transit policy that minimizes public polarization and conflict while maximizing adoption across commuters, businesses, and taxpayers.",
        "Downtown Clean Air & Congestion Charge Zone": "Design a low-emission traffic congestion pricing scheme that reduces emissions without unfairly penalizing low-income commuters or small business delivery logistics.",
        "High-Density Affordable Housing Re-zoning": "Formulate a municipal housing expansion policy that increases affordable units while addressing neighborhood density and traffic concerns.",
        "Custom Goal Definition": "Optimize community policy intervention to maximize social adoption and equity while keeping polarization below 0.35.",
    }

    default_goal = preset_goals.get(preset_choice, "")

    with col_file:
        source_mode = st.radio("Source Documents", ["Use Repository Sample (demo_transit_policy.md)", "Upload Custom Files"], horizontal=True)
        uploaded_doc_paths = []
        if source_mode == "Upload Custom Files":
            uploaded = st.file_uploader("Upload policy documents (PDF, MD, TXT)", accept_multiple_files=True)
            if uploaded:
                for uf in uploaded:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}")
                    tfile.write(uf.getvalue())
                    tfile.close()
                    uploaded_doc_paths.append(tfile.name)
        else:
            sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../demo_transit_policy.md"))
            if os.path.exists(sample_path):
                uploaded_doc_paths.append(sample_path)

    goal_input = st.text_area("Target Policy Goal", value=default_goal, height=75)

    # Constraints configuration
    st.markdown("**Constraint Thresholds (Evaluated via SocialImpactModel):**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        polarization_thresh = st.slider("Max Polarization (<)", 0.10, 0.90, 0.35, 0.05)
    with c2:
        conflict_thresh = st.slider("Max Conflict (<)", 0.10, 0.90, 0.30, 0.05)
    with c3:
        adoption_thresh = st.slider("Min Adoption (>)", 0.10, 0.90, 0.60, 0.05)
    with c4:
        acceptance_thresh = st.slider("Min Acceptance (>)", 0.10, 0.90, 0.65, 0.05)

    parsed_constraints = {
        "polarization": f"<{polarization_thresh:.2f}",
        "conflict": f"<{conflict_thresh:.2f}",
        "adoption": f">{adoption_thresh:.2f}",
        "acceptance": f">{acceptance_thresh:.2f}",
    }

    c_iter, c_prov, c_hitl = st.columns([1, 1, 1])
    with c_iter:
        max_iterations = st.number_input("Max Iteration Budget", min_value=1, max_value=15, value=5)
    with c_prov:
        primary_prov = st.selectbox("Primary LLM Provider", ["ollama", "claude-cli", "codex-cli"], index=0)
    with c_hitl:
        st.write("")
        st.write("")
        enable_hitl = st.checkbox("Enable Human-in-the-Loop Checkpoints", value=True)

    # 2. Controlled Disruption Demo Injection Panel
    with st.expander("⚡ Demo Disruption & Failure Injection Framework", expanded=False):
        st.markdown("Test agent robustness by injecting domain disruptions or technical failures into runtime execution:")
        d_col1, d_col2, d_col3 = st.columns([1, 1, 1])
        
        with d_col1:
            domain_disp = st.selectbox(
                "Domain Disruption",
                [
                    "None",
                    "Increase Polarization Surge (+0.35)",
                    "Increase Stakeholder Opposition (-0.30 Acceptance)",
                    "Reduce Budget Constraint (-0.25 Adoption)",
                ]
            )
        with d_col2:
            tech_disp = st.selectbox(
                "Technical/System Failure",
                [
                    "None",
                    "Simulate Primary LLM Timeout (Triggers Provider Failover)",
                    "Simulate Provider Unavailable (Connection Refused)",
                    "Return Malformed Tool Response (Corrupted JSON)",
                    "Simulate OASIS Malformed Execution Output",
                    "Raise Controlled Tool Exception",
                ]
            )
        with d_col3:
            st.write("")
            st.write("")
            if st.button("Apply Injected Disruption", use_container_width=True):
                disp_eng: DisruptionEngine = st.session_state.disruption_engine
                prov_mgr: ProviderFailoverManager = st.session_state.provider_manager
                
                # Handle Domain
                if "Polarization" in domain_disp:
                    disp_eng.inject_domain_disruption(DomainDisruptionType.INCREASE_POLARIZATION, {"polarization_boost": 0.35})
                    st.success("Injected: Polarization surge disruption active!")
                elif "Opposition" in domain_disp:
                    disp_eng.inject_domain_disruption(DomainDisruptionType.INCREASE_STAKEHOLDER_OPPOSITION, {"opposition_penalty": 0.30})
                    st.success("Injected: Stakeholder opposition disruption active!")
                elif "Budget" in domain_disp:
                    disp_eng.inject_domain_disruption(DomainDisruptionType.REDUCE_BUDGET, {"budget_penalty": 0.25})
                    st.success("Injected: Budget reduction disruption active!")
                
                # Handle Tech
                if "Timeout" in tech_disp:
                    prov_mgr.inject_provider_failure(primary_prov, "timeout")
                    disp_eng.inject_technical_disruption(TechnicalDisruptionType.FORCE_LLM_TIMEOUT)
                    st.warning("Injected: LLM Timeout on primary provider active!")
                elif "Unavailable" in tech_disp:
                    prov_mgr.inject_provider_failure(primary_prov, "unavailable")
                    disp_eng.inject_technical_disruption(TechnicalDisruptionType.FORCE_PROVIDER_UNAVAILABLE)
                    st.warning("Injected: Provider unavailable active!")
                elif "Malformed" in tech_disp:
                    disp_eng.inject_technical_disruption(TechnicalDisruptionType.RETURN_MALFORMED_TOOL_RESPONSE)
                    st.warning("Injected: Malformed tool response active!")
                elif "OASIS" in tech_disp:
                    disp_eng.inject_technical_disruption(TechnicalDisruptionType.FORCE_OASIS_MALFORMED_OUTPUT)
                    st.warning("Injected: OASIS execution corruption active!")
                elif "Exception" in tech_disp:
                    disp_eng.inject_technical_disruption(TechnicalDisruptionType.RAISE_TOOL_EXCEPTION)
                    st.warning("Injected: Controlled tool exception active!")

    # 3. Action Execution Controls
    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

    with btn_col1:
        start_clicked = st.button("▶ Start Autonomous Decision Run", type="primary", use_container_width=True)
    with btn_col2:
        step_clicked = st.button("⏭ Step Forward (1 Action)", use_container_width=True)
    with btn_col3:
        reset_clicked = st.button("🔄 Reset Agent State", use_container_width=True)

    if reset_clicked:
        st.session_state.agent_state = None
        st.session_state.agent_controller = None
        st.session_state.agent_traces = []
        st.session_state.disruption_engine = DisruptionEngine()
        st.session_state.provider_manager = ProviderFailoverManager()
        st.success("Agent state reset.")
        st.rerun()

    # Trace collector helper
    def _ui_trace_collector(phase: str, msg: str, payload: Dict[str, Any]):
        st.session_state.agent_traces.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "phase": phase,
            "message": msg,
            "payload": payload,
        })

    # Start or Step logic
    if start_clicked or step_clicked:
        if st.session_state.agent_controller is None or st.session_state.agent_state is None:
            # Initialize fresh run in RunStore
            store = RunStore()
            manifest = store.create_autonomous_run(
                goal=goal_input,
                source_files=uploaded_doc_paths,
                constraints=parsed_constraints,
                project_name="Autonomous Decision Run",
            )
            run_id = manifest["run_id"]
            if uploaded_doc_paths:
                store.freeze_source_files(run_id, uploaded_doc_paths)

            # Initialize fresh controller and state
            state = DecisionState(
                run_id=run_id,
                goal=goal_input,
                problem_statement=goal_input,
                source_files=uploaded_doc_paths,
                constraints=parsed_constraints,
                max_iterations=int(max_iterations),
                primary_provider=primary_prov,
                active_provider=primary_prov,
            )
            controller = AutonomousDecisionController(
                state=state,
                provider_manager=st.session_state.provider_manager,
                disruption_engine=st.session_state.disruption_engine,
                on_trace_callback=_ui_trace_collector,
            )
            st.session_state.agent_state = state
            st.session_state.agent_controller = controller
            st.session_state.agent_traces = []
            controller.initialize_run(
                goal=goal_input,
                problem_statement=goal_input,
                source_files=uploaded_doc_paths,
                constraints=parsed_constraints,
                max_iterations=int(max_iterations),
                primary_provider=primary_prov,
            )

        controller = st.session_state.agent_controller
        controller.on_trace_callback = _ui_trace_collector

        if start_clicked:
            with st.spinner("Autonomous Decision Agent is executing the closed-loop optimization cycle..."):
                controller.run_until_completion()
        elif step_clicked:
            with st.spinner("Agent executing single action step..."):
                controller.execute_next_step()

    # 4. Human-in-the-Loop Checkpoint Card (if pending)
    state: Optional[DecisionState] = st.session_state.agent_state
    if state and state.human_interaction_pending and state.human_interaction_request:
        st.markdown("### 🛑 Human Stakeholder Checkpoint Required")
        req_data = state.human_interaction_request
        
        st.warning(f"**{req_data.get('title', 'Confirmation')}**\n\n{req_data.get('prompt', '')}")
        
        human_guidance_input = st.text_input(
            "📝 Human Policy Guidance / Tradeoff Directive (Optional):",
            placeholder="e.g., Add a bus discount and cap peak toll at $3.00",
            key="human_guidance_directive",
        )

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            if st.button("✅ Approve Policy Modification & Continue", type="primary", use_container_width=True):
                controller = st.session_state.agent_controller
                controller.resume_after_human(approval=True, text_input=human_guidance_input or None)
                st.rerun()
        with h_col2:
            if st.button("❌ Reject & Force Re-planning Alternative", use_container_width=True):
                controller = st.session_state.agent_controller
                controller.resume_after_human(approval=False, text_input=human_guidance_input or None)
                st.rerun()

    # 5. Agent Dashboard & Metrics (if active)
    if state:
        st.markdown("---")
        st.subheader("2. Real-Time Agent State & KPI Overview")
        
        m1, m2, m3, m4 = st.columns(4)
        status_val = state.status.value if isinstance(state.status, AgentStatus) else state.status
        status_color = {
            "verified": "🟢 VERIFIED",
            "planning": "🟣 PLANNING",
            "executing": "🟠 EXECUTING",
            "evaluating": "🔵 EVALUATING",
            "replanning": "🟡 REPLANNING",
            "unresolved": "🔴 UNRESOLVED",
            "failed": "🔴 FAILED",
            "paused_for_human": "🛑 PAUSED (HUMAN)",
        }.get(status_val, f"⚪ {status_val.upper()}")

        with m1:
            st.metric("Iteration Budget", f"{state.iteration_count} / {state.max_iterations}")
        with m2:
            st.metric("Agent Status", status_color)
        with m3:
            best_s = f"{state.best_score:.2f}" if state.best_score >= 0 else "N/A"
            st.metric("Best Viability Score", best_s)
        with m4:
            st.metric("Active LLM Provider", state.active_provider.upper())

        # Constraints Checklist
        st.markdown("**Constraint Verification Status:**")
        chk_col1, chk_col2 = st.columns(2)
        with chk_col1:
            st.markdown("##### 🎯 Target Constraints")
            for c_name, c_expr in state.constraints.items():
                st.markdown(f"- `{c_name}`: **{c_expr}**")
        with chk_col2:
            st.markdown("##### 📊 Latest Verification Outcome")
            if state.verification_history:
                latest_v = state.verification_history[-1]
                for s in latest_v.satisfied_constraints:
                    st.markdown(f"🟢 **{s}**")
                for v in latest_v.violated_constraints:
                    st.markdown(f"🔴 **{v}**")
                if latest_v.uncertain_aspects:
                    for u in latest_v.uncertain_aspects:
                        st.markdown(f"🟡 *{u}*")
            else:
                st.info("No verification completed yet.")

        # 6. Live Agent Trace Display
        st.markdown("---")
        st.subheader("3. Live Autonomous Execution Trace")
        
        trace_container = st.container()
        with trace_container:
            for trace in st.session_state.agent_traces:
                phase = trace.get("phase", "EVENT").upper()
                msg = trace.get("message", "")
                ts = trace.get("timestamp", "")
                payload = trace.get("payload", {})
                
                badge_style = {
                    "GOAL": "background:#e8f5e9; color:#2e7d32; border-left: 4px solid #2e7d32;",
                    "OBSERVE": "background:#e3f2fd; color:#1565c0; border-left: 4px solid #1565c0;",
                    "PLAN": "background:#f3e5f5; color:#6a1b9a; border-left: 4px solid #6a1b9a;",
                    "ACTION": "background:#fff3e0; color:#e65100; border-left: 4px solid #e65100;",
                    "EVALUATE": "background:#e0f7fa; color:#00838f; border-left: 4px solid #00838f;",
                    "VERIFY": "background:#fbe9e7; color:#d84315; border-left: 4px solid #d84315;",
                    "VERIFIED": "background:#e8f5e9; color:#1b5e20; border-left: 5px solid #1b5e20; font-weight:bold;",
                    "FAILED": "background:#ffebee; color:#c62828; border-left: 4px solid #c62828;",
                    "REPLAN": "background:#fffde7; color:#f57f17; border-left: 4px solid #f57f17;",
                    "RECOVERY": "background:#e1f5fe; color:#0277bd; border-left: 4px solid #0277bd;",
                    "HUMAN": "background:#fff8e1; color:#ff8f00; border-left: 4px solid #ff8f00;",
                    "FINALIZE": "background:#e8f5e9; color:#1b5e20; border-left: 5px solid #1b5e20; font-weight:bold;",
                    "UNRESOLVED": "background:#ffebee; color:#b71c1c; border-left: 5px solid #b71c1c;",
                }.get(phase, "background:#fafafa; color:#333; border-left: 4px solid #aaa;")

                st.markdown(
                    f'<div class="trace-card" style="{badge_style}"><strong>[{ts}] {phase}</strong>: {msg}</div>',
                    unsafe_allow_html=True,
                )

                # If Planner payload, show candidate ranking table
                if phase == "PLAN" and "candidates" in payload:
                    with st.expander(f"🔍 Action Utility Ranking Breakdown (Round {trace.get('timestamp')})"):
                        st.caption("Transparent multi-action ranking based on Action Utility formula:")
                        cand_rows = []
                        for c in payload["candidates"]:
                            cand_rows.append({
                                "Rank Score": f"{c.get('score', 0.0):.2f}",
                                "Candidate Action": c.get("action_name", ""),
                                "Goal Progress": f"{c.get('goal_progress', 0.0):.2f}",
                                "Constraint Recovery": f"{c.get('constraint_recovery_potential', 0.0):.2f}",
                                "Info Gain": f"{c.get('information_gain', 0.0):.2f}",
                                "Cost/Risk": f"{(c.get('action_cost', 0.0) + c.get('failure_risk', 0.0)):.2f}",
                                "Rationale": c.get("rationale", ""),
                            })
                        st.table(cand_rows)

        # 7. Candidate Scenarios Explored
        if state.candidate_scenarios:
            st.markdown("---")
            st.subheader("4. Candidate Policy Scenarios Explored")
            scen_table = []
            for s_id, s_data in state.candidate_scenarios.items():
                s_impact = state.social_impact_results.get(s_id, {})
                scen_table.append({
                    "Scenario ID": s_id,
                    "Policy Name": s_data.get("name", ""),
                    "Intervention Summary": s_data.get("intervention", "")[:80] + "...",
                    "Viability Score": f"{s_impact.get('overall_score', 0.0):.2f}" if s_impact else "Pending",
                    "Polarization": f"{s_impact.get('polarization_score', 0.0):.2f}" if s_impact else "Pending",
                    "Conflict": f"{s_impact.get('conflict_score', 0.0):.2f}" if s_impact else "Pending",
                    "Adoption": f"{s_impact.get('adoption_score', 0.0):.2f}" if s_impact else "Pending",
                })
            st.dataframe(scen_table, use_container_width=True)

        # 8. Final Outcome & Persistence Export
        if state.status in (AgentStatus.VERIFIED, AgentStatus.UNRESOLVED, AgentStatus.FAILED):
            st.markdown("---")
            st.subheader("5. Final Outcome & Persistent Audit Artifacts")
            
            if state.status == AgentStatus.VERIFIED:
                st.success(f"🎉 **OPTIMAL POLICY VERIFIED**: {state.final_outcome_summary}")
            else:
                st.error(f"⚠️ **SAFE STOP**: {state.final_outcome_summary}")

            # Save to Store
            store = RunStore()
            persisted = store.persist_autonomous_state(state.run_id, state.to_dict())
            
            st.markdown(f"**Run Artifacts Persisted under `uploads/runs/{state.run_id}/autonomous_run/`:**")
            for art_key, art_path in persisted.items():
                st.text(f"📄 {art_key}: {art_path}")

            st.download_button(
                "📥 Download Autonomous Run State (JSON)",
                data=json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
                file_name=f"{state.run_id}_state.json",
                mime="application/json",
            )

            if "demo_trace" in persisted and os.path.exists(persisted["demo_trace"]):
                with open(persisted["demo_trace"], "r", encoding="utf-8") as _trace_f:
                    _trace_data = _trace_f.read()
                st.download_button(
                    "📥 Download Human-Readable Audit Trace (demo_trace.md)",
                    data=_trace_data,
                    file_name=f"{state.run_id}_demo_trace.md",
                    mime="text/markdown",
                )


# Main app
def main():
    """Main Streamlit application."""
    render_navigation()
    
    # Route to appropriate page
    page = st.session_state.current_page
    
    if page == 'autonomous_agent':
        render_autonomous_agent()
    elif page == 'dashboard':
        render_dashboard()
    elif page == 'community_problem':
        render_community_problem()
    elif page == 'stakeholder_config':
        render_stakeholder_config()
    elif page == 'candidate_decisions':
        render_candidate_decisions()
    elif page == 'simulation_params':
        render_simulation_params()
    elif page == 'run_simulation':
        render_run_simulation()
    elif page == 'emergent_behaviour':
        render_emergent_behaviour()
    elif page == 'social_impact':
        render_social_impact()
    elif page == 'scenario_comparison':
        render_scenario_comparison()
    elif page == 'recommendation':
        render_recommendation()
    elif page == 'new_simulation':
        render_new_simulation()
    elif page == 'pipeline_progress':
        render_pipeline_progress()
    elif page == 'explore_run':
        render_explore_run()
    elif page == 'history':
        render_history()
    elif page == 'about':
        render_about()
    else:
        render_autonomous_agent()


if __name__ == "__main__":
    main()