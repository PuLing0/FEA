"""Minimal LangGraph runtime skeleton for fig_edit_agent."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.evaluator_agent import EvaluatorAgent
from agents.execute_agent import ExecuteAgent
from agents.plan_agent import PlanAgent
from schema import (
    ArtifactIndex,
    ArtifactKind,
    ImageArtifact,
    InstructionArtifact,
    SessionPhase,
    SessionState,
    ToolInvocationRecord,
    ToolName,
    UnderstandArgs,
    UnderstandingArtifact,
)

from .prompts import BOOTSTRAP_UNDERSTAND_QUESTION_TEMPLATE
from .config import default_max_execute_acts
from .artifact_context import (
    add_to_session_working_set,
    register_artifact_in_session_pool,
)
from .run_logger import (
    create_run_logger,
    log_event,
    summarize_artifact,
    summarize_decision,
    summarize_image_index,
    summarize_operation,
    summarize_task_state,
)
from .state import RuntimeState
from .tool_runner import ToolRunner
from tools.registry import build_default_tool_registry


PLAN_AGENT = PlanAgent()
EXECUTE_AGENT = ExecuteAgent()
EVALUATOR_AGENT = EvaluatorAgent()
TOOL_REGISTRY = build_default_tool_registry()
TOOL_RUNNER = ToolRunner(TOOL_REGISTRY)


def _initial_session(session_id: str) -> SessionState:
    return SessionState(session_id=session_id, phase=SessionPhase.UNDERSTANDING)


def register_and_understand(state: RuntimeState) -> RuntimeState:
    runtime_input = state["input"]
    runtime_input.setdefault("use_llm", False)
    if "image_uris" not in runtime_input or not runtime_input["image_uris"]:
        runtime_input["image_uris"] = [runtime_input["image_uri"]]
    max_task_loops = state.get("max_task_loops", 2)
    max_execute_acts = state.get("max_execute_acts", default_max_execute_acts())
    max_evaluator_checkpoints = state.get("max_evaluator_checkpoints", 3)
    max_tool_failures = state.get("max_tool_failures", 3)
    logger = create_run_logger(runtime_input["session_id"])
    session = _initial_session(runtime_input["session_id"])
    session.artifact_index = ArtifactIndex(by_type={})
    state = {
        "input": runtime_input,
        "session": session,
        "artifacts": {},
        "plans": {},
        "tasks": {},
        "task_act_records": [],
        "task_loops": [],
        "operations": [],
        "max_task_loops": max_task_loops,
        "max_execute_acts": max_execute_acts,
        "max_evaluator_checkpoints": max_evaluator_checkpoints,
        "max_tool_failures": max_tool_failures,
        "run_id": logger.run_id,
        "run_log_uri": logger.uri,
        "run_logger": logger,
    }
    log_event(
        state,
        "run_start",
        log_uri=logger.uri,
        image_uris=list(runtime_input["image_uris"]),
        use_llm=runtime_input.get("use_llm", False),
        max_task_loops=max_task_loops,
        max_execute_acts=max_execute_acts,
        max_evaluator_checkpoints=max_evaluator_checkpoints,
        max_tool_failures=max_tool_failures,
    )
    log_event(state, "node_start", node="register_and_understand")
    root_instruction = InstructionArtifact(
        id="art_instruction_session_root_001",
        summary=runtime_input["instruction_text"],
        payload={"instruction_text": runtime_input["instruction_text"]},
        created_by="user",
        role="session_root_instruction",
        scope="session",
    )
    register_artifact_in_session_pool(state, root_instruction)
    add_to_session_working_set(
        state,
        root_instruction.id,
        usage="session root instruction",
        selection_reason="created from the user instruction at session start",
    )
    log_event(state, "artifact_created", **summarize_artifact(root_instruction))
    for index, image_uri in enumerate(runtime_input["image_uris"], start=1):
        image = ImageArtifact(
            id=f"art_img_input_{index:03d}",
            uri=image_uri,
            payload={"role": "initial_input", "slot_index": index},
            created_by="user",
            role="initial_input",
            scope="session",
        )
        register_artifact_in_session_pool(state, image)
        add_to_session_working_set(
            state,
            image.id,
            usage="session input image",
            selection_reason="provided by the user at session start",
        )
        log_event(state, "artifact_created", **summarize_artifact(image))

        if not runtime_input.get("use_llm", False):
            understanding = UnderstandingArtifact(
                id=f"art_understanding_bootstrap_{index:03d}",
                summary=f"Input image slot {index}: {image_uri}",
                payload={
                    "image_ref": image.id,
                    "task_instruction": runtime_input["instruction_text"],
                    "summary": f"Input image slot {index}: {image_uri}",
                },
                source_ids=[image.id],
                created_by=ToolName.UNDERSTAND.value,
                role="bootstrap_understanding",
                scope="session",
            )
            register_artifact_in_session_pool(state, understanding)
            invocation = ToolInvocationRecord(
                    id=f"op_understand_bootstrap_{index:03d}",
                    task_id="bootstrap",
                    loop_index=0,
                    tool_name=ToolName.UNDERSTAND,
                    args={
                        "image_ref": image.id,
                        "question": BOOTSTRAP_UNDERSTAND_QUESTION_TEMPLATE.format(index=index),
                    },
                    status="succeeded",
                    output_refs=[understanding.id],
                    result_payload={
                        "image_ref": image.id,
                        "summary": understanding.payload["summary"],
                        "understanding_ref": understanding.id,
                    },
                    raw_output_uri=f"runs/bootstrap/{ToolName.UNDERSTAND.value}.json",
                )
            state["operations"].append(invocation)
            image.summary = understanding.payload["summary"]
            image.role = "initial_input"
            log_event(state, "operation_succeeded", **summarize_operation(invocation))
            log_event(state, "artifact_created", **summarize_artifact(understanding))
            continue

        understand_execution = TOOL_RUNNER.run(
            state,
            ToolName.UNDERSTAND,
            task_id="bootstrap",
            loop_index=0,
            args=UnderstandArgs(
                image_ref=image.id,
                question=BOOTSTRAP_UNDERSTAND_QUESTION_TEMPLATE.format(index=index),
            ),
        )
        state["operations"].append(understand_execution.invocation)
        event = "operation_failed" if understand_execution.invocation.status == "failed" else "operation_succeeded"
        log_event(state, event, **summarize_operation(understand_execution.invocation))
        for artifact in understand_execution.artifacts:
            register_artifact_in_session_pool(state, artifact)
            if artifact.payload.get("summary"):
                image.summary = artifact.payload["summary"]
            image.role = "initial_input"
            log_event(state, "artifact_created", **summarize_artifact(artifact))
    state["session"].phase = SessionPhase.PLANNING
    log_event(state, "node_end", node="register_and_understand", image_artifact_ids=summarize_image_index(state))
    return state


def plan(state: RuntimeState) -> RuntimeState:
    log_event(state, "node_start", node="plan")
    before_plan_ids = set(state.get("plans", {}).keys())
    before_task_ids = set(state.get("tasks", {}).keys())
    result = PLAN_AGENT.run(state)
    new_plan_ids = [plan_id for plan_id in result.get("plans", {}) if plan_id not in before_plan_ids]
    new_task_ids = [task_id for task_id in result.get("tasks", {}) if task_id not in before_task_ids]
    log_event(
        result,
        "plan_created",
        plan_ids=new_plan_ids,
        task_ids=new_task_ids,
        current_plan_id=result["session"].current_plan_id,
        current_task_id=result["session"].current_task_id,
    )
    log_event(result, "node_end", node="plan")
    return result


def execute_current_task(state: RuntimeState) -> RuntimeState:
    log_event(state, "node_start", node="execute")
    before_operation_count = len(state.get("operations", []))
    before_artifact_ids = set(state.get("artifacts", {}).keys())
    task_id = state["session"].current_task_id
    result = EXECUTE_AGENT.run(state)
    for operation in result.get("operations", [])[before_operation_count:]:
        event = "operation_failed" if operation.status == "failed" else "operation_succeeded"
        log_event(result, event, **summarize_operation(operation))
    for artifact_id, artifact in result.get("artifacts", {}).items():
        if artifact_id not in before_artifact_ids:
            log_event(result, "artifact_created", **summarize_artifact(artifact))
    task_state = result["session"].task_states.get(task_id) if task_id else None
    log_event(
        result,
        "execute_checkpoint",
        task_state=summarize_task_state(task_state, result),
        decision=summarize_decision(result.get("decision")),
    )
    log_event(result, "node_end", node="execute")
    return result


def evaluate_checkpoint(state: RuntimeState) -> RuntimeState:
    log_event(state, "node_start", node="evaluate")
    before_operation_count = len(state.get("operations", []))
    before_artifact_ids = set(state.get("artifacts", {}).keys())
    task_id = state["session"].current_task_id
    result = EVALUATOR_AGENT.run(state)
    for operation in result.get("operations", [])[before_operation_count:]:
        event = "operation_failed" if operation.status == "failed" else "operation_succeeded"
        log_event(result, event, **summarize_operation(operation))
    for artifact_id, artifact in result.get("artifacts", {}).items():
        if artifact_id not in before_artifact_ids:
            log_event(result, "artifact_created", **summarize_artifact(artifact))
    task_state = result["session"].task_states.get(task_id) if task_id else None
    decision = result.get("decision")
    log_event(
        result,
        "evaluate_decision",
        decision_route=getattr(decision.route, "value", decision.route) if decision else None,
        task_state=summarize_task_state(task_state, result),
        decision=summarize_decision(decision),
    )
    log_event(result, "node_end", node="evaluate")
    return result


def build_runtime_graph(*, stop_after_plan: bool = False):
    graph = StateGraph(RuntimeState)
    graph.add_node("register_and_understand", register_and_understand)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute_current_task)
    graph.add_node("evaluate", evaluate_checkpoint)

    graph.add_edge(START, "register_and_understand")
    graph.add_edge("register_and_understand", "plan")
    if stop_after_plan:
        graph.add_edge("plan", END)
        return graph.compile()

    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {
            "execute": "execute",
            "evaluate": "evaluate",
            "plan": "plan",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "execute": "execute",
            "plan": "plan",
            "end": END,
        },
    )
    return graph.compile()


def _route_after_execute(state: RuntimeState) -> str:
    current_task_id = state["session"].current_task_id
    if current_task_id is None:
        if state["session"].phase == SessionPhase.PLANNING:
            log_event(state, "route_next", route="plan")
            return "plan"
        log_event(state, "run_end", route="end")
        log_event(state, "route_next", route="end")
        return "end"

    task_state = state["session"].task_states[current_task_id]
    if task_state.latest_execute_checkpoint == "passed":
        log_event(state, "route_next", route="evaluate")
        return "evaluate"
    if task_state.latest_execute_checkpoint == "retry":
        log_event(state, "route_next", route="execute")
        return "execute"
    if task_state.latest_execute_checkpoint == "failed":
        log_event(state, "route_next", route="plan")
        return "plan"
    log_event(state, "run_end", route="end")
    log_event(state, "route_next", route="end")
    return "end"


def _route_after_evaluate(state: RuntimeState) -> str:
    if state["session"].phase == SessionPhase.EXECUTING:
        log_event(state, "route_next", route="execute")
        return "execute"
    if state["session"].phase == SessionPhase.PLANNING:
        log_event(state, "route_next", route="plan")
        return "plan"
    log_event(state, "run_end", route="end")
    log_event(state, "route_next", route="end")
    return "end"
