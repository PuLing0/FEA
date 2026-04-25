"""Centralized prompt text for the fig edit agent runtime."""

from __future__ import annotations

from typing import Any


INPUT_THINKING_SYSTEM_PROMPT = (
    "You are preparing inputs for an image-editing task. "
    "Think only about what categories of images are needed before execution starts."
)

INPUT_SELECTOR_SYSTEM_PROMPT = (
    "You are a stateless image input selector. "
    "Given candidate image descriptions and an input-thinking note, choose only the images needed for the task."
)

INPUT_VALIDATION_SYSTEM_PROMPT = (
    "You validate selected task input images before execution starts. "
    "Confirm whether the selected set is sufficient and coherent."
)

PLAN_SYSTEM_PROMPT = (
    "You are a planning agent for image editing. "
    "Return structured plan output only. "
    "plan_instruction must be a single concise sentence. "
    "tasks must be a non-empty array. "
    "When replan context is present, keep retained prefix tasks fixed and generate only new suffix tasks. "
    "Do not put task JSON into plan_instruction."
)

EXECUTE_STRATEGY_SYSTEM_PROMPT = (
    "You are an execution planner for an image-editing agent. "
    "Choose only the next single tool for the next act. "
    "Available tools: prompt_reconstruct, grounding, segment, crop, understand, collage, edit. "
    "Tool guidance: use grounding when you need a coarse location or bbox-style candidate region before a more precise local step; "
    "use segment when you need to isolate or localize a target region; "
    "use crop when you need a focused preview of a segmented area; "
    "use understand when you need to inspect what an image or preview actually contains; "
    "use collage when multiple reference images should first be organized into one unified reference image; "
    "use prompt_reconstruct when the current prompt is vague, underspecified, or uses ambiguous image references, "
    "especially before an edit if a stronger prompt would help; "
    "use edit when you are ready to apply the actual image transformation. "
    "These are options, not a fixed workflow. Do not follow a rigid path if the current state suggests otherwise. "
    "Do not schedule multiple tools. "
    "If there are multiple image candidates, choose the most appropriate base_image_artifact_id from the resolved input artifact ids."
)

EXECUTE_OBSERVE_SYSTEM_PROMPT = (
    "You are the observe step of an image-editing execute agent. "
    "Decide whether the current loop should continue or whether the current results are good enough to stop the execute checkpoint with success. "
    "Also generate one short semantic summary for each newly produced artifact. "
    "Return only structured output."
)

EXECUTION_HISTORY_SUMMARY_SYSTEM_PROMPT = (
    "You summarize earlier execution history for an image-editing task. "
    "Summarize only confirmed facts from the provided records. "
    "Do not invent new requirements. Return one concise paragraph only."
)

UNDERSTAND_SYSTEM_PROMPT = (
    "You understand an input image for an image-editing agent. "
    "Return one concise factual summary focused on what is visible and relevant to the task."
)

GROUNDING_SYSTEM_PROMPT = (
    "You are an image grounding assistant. "
    "Given an image and a semantic grounding query, return up to the requested number of candidates. "
    "Each candidate must include a short label and a bbox [x1, y1, x2, y2]. "
    "You may optionally include score, positive_points, and negative_points. "
    "Use bbox as the primary localization output. "
    "Do not return any prose outside the structured schema."
)

COLLAGE_LAYOUT_SYSTEM_PROMPT = (
    "You are a layout planner for hard image collages. "
    "Return only the structured layout. Use every provided artifact exactly once. "
    "Do not invent artifact ids. Keep all layers fully inside the canvas."
)

PROMPT_RECONSTRUCT_SYSTEM_PROMPT = (
    "You rewrite execution prompts for an image editing task. "
    "Make the instruction clearer, more specific, and replace ambiguous image references "
    "with explicit artifact ids when available. Output only the final rewritten instruction text."
)

EVALUATE_SYSTEM_PROMPT = (
    "You are a strict image-editing evaluator. Return only structured output. "
    "Judge both whether the edit is satisfactory and the 0-5 score dimensions. "
    "If the result needs more work, provide concrete issues and a refined edit prompt."
)

EVALUATE_SCORE_RUBRIC = """
All score dimensions use this 0-5 scale:
0: Not applicable or impossible to judge from the provided images.
1: Severe failure. The dimension is essentially wrong and should trigger replan.
2: Major issue. The result is mostly unsatisfactory for this dimension.
3: Partial success. The core idea is visible, but important problems remain.
4: Good. Minor issues remain, but this dimension is mostly successful.
5: Excellent. This dimension is fully satisfied with no meaningful issue.

Dimensions:
- instruction_success: Whether the candidate fulfills the final edit instruction.
- reference_consistency: Whether the candidate uses the task input/reference images correctly.
- overediting: Whether the candidate avoids changing content that should be preserved.
- naturalness: Whether lighting, perspective, scale, composition, and blending look natural.
- artifacts: Whether the image avoids distortions, broken anatomy, blurred faces, watermarks, damaged edges, or texture artifacts.
""".strip()

REAL_AGENT_SMOKE_DEFAULT_INSTRUCTION = (
    "Generate a photo of this person wearing the provided top and skirt in the "
    "provided background. Preserve the face identity and keep the result natural."
)

BOOTSTRAP_UNDERSTAND_QUESTION_TEMPLATE = "understand image slot {index} for the user request"
CANDIDATE_UNDERSTAND_QUESTION_TEMPLATE = "understand candidate image {image_id} for task execution"
EXECUTE_GROUNDING_QUERY = "Locate the primary edit region relevant to the current task."
EXECUTE_UNDERSTAND_QUESTION = "understand the current preview for this step and summarize what it shows"
EXECUTE_COLLAGE_LAYOUT_GOAL = "organize multiple references into one clear reference board for the next edit"
DEFAULT_UNDERSTAND_QUESTION = "Summarize the visible contents relevant to the task."


def build_input_thinking_user_prompt(*, task: Any, candidates_text: str) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Task depends_on: {task.depends_on}\n"
        f"Static task inputs: {task.input_artifact_ids}\n"
        f"Candidate images:\n{candidates_text}\n"
        "Briefly explain what image inputs are needed for this task before execution starts."
    )


def build_input_selector_user_prompt(*, task: Any, thinking: str, candidates_text: str) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Task depends_on: {task.depends_on}\n"
        f"Input thinking:\n{thinking}\n"
        f"Candidate images:\n{candidates_text}\n"
        "Return only selected_artifact_ids."
    )


def build_input_validation_user_prompt(*, task: Any, selected_artifact_ids: list[str], input_thinking: str | None, candidates_text: str) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Selected artifact ids: {selected_artifact_ids}\n"
        f"Input thinking: {input_thinking}\n"
        f"Candidate images:\n{candidates_text}\n"
        "Briefly validate whether the selected image set is appropriate."
    )


def build_plan_user_prompt(
    *,
    instruction_text: str,
    replan_context: str,
    retained_prefix_plan: str,
    available_artifacts: str,
    image_artifact_ids: list[str],
    understanding_summaries: list[dict[str, Any]],
) -> str:
    return (
        f"Instruction: {instruction_text}\n"
        f"{replan_context}"
        f"{retained_prefix_plan}"
        f"{available_artifacts}"
        f"Image artifact ids: {image_artifact_ids}\n"
        f"Understanding summaries: {understanding_summaries}\n"
        "Each task must include: id, type, instruction, input_artifact_ids, "
        "depends_on, acceptance_criteria. "
        "New tasks must use only the provided available new task ids when present. "
        "depends_on may reference retained prefix task ids and newly generated task ids. "
        "Do not regenerate retained prefix tasks. "
        "For replan tasks, Available artifacts for planning are context only. "
        "Default each new replan task's input_artifact_ids to an empty array, because runtime will select real image inputs at task start. "
        "Only include input_artifact_ids when the task absolutely requires fixed static images from Available artifacts for planning. "
        "Do not invent future artifact ids in input_artifact_ids; use depends_on for future task outputs."
    )


def build_execute_observe_user_prompt(
    *,
    task: Any,
    active_instruction: str,
    retry_context: str | None,
    selected_tool: str,
    tool_args: dict[str, Any],
    source_lines: list[str],
    new_artifact_lines: list[str],
) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Acceptance criteria: {task.acceptance_criteria}\n"
        f"Current active instruction: {active_instruction}\n"
        f"Retry context: {retry_context}\n"
        f"Tool name: {selected_tool}\n"
        f"Tool args: {tool_args}\n"
        f"Source artifacts:\n{chr(10).join(source_lines) or '(none)'}\n"
        f"New artifacts:\n{chr(10).join(new_artifact_lines) or '(none)'}\n"
        "Return outcome as either 'continue' or 'success'. "
        "Choose success only if the current tool result is enough to stop the execute checkpoint and hand off to evaluator. "
        "For each new artifact, write a concise summary explaining what it is and what role it plays in the current task."
    )


def build_execute_strategy_user_prompt(
    *,
    task: Any,
    user_instruction: str,
    resolved_ids: list[str],
    resolved_image_summaries: str,
    retry_context: str | None,
    active_instruction: str,
    task_artifact_context: str,
    latest_candidate_refs: list[str],
) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"User instruction: {user_instruction}\n"
        f"Resolved input artifact ids: {resolved_ids}\n"
        f"Resolved image candidates:\n{resolved_image_summaries}\n"
        f"Retry context: {retry_context}\n"
        f"Current active instruction: {active_instruction}\n"
        f"Task artifact summary:\n{task_artifact_context}\n"
        f"Latest candidate refs: {latest_candidate_refs}\n"
        "Return reasoning, selected_tools, and base_image_artifact_id. selected_tools should contain exactly one next tool name."
    )


def build_execution_history_summary_user_prompt(*, task_instruction: str, raw_records: str) -> str:
    return (
        f"Task instruction: {task_instruction}\n"
        f"Earlier thinking-act-observe rounds:\n{raw_records}"
    )


def build_understand_user_prompt(*, task_instruction: str, question: str | None) -> str:
    return (
        f"Task instruction: {task_instruction}\n"
        f"Question: {question or DEFAULT_UNDERSTAND_QUESTION}\n"
        "Return only the summary."
    )


def build_grounding_user_prompt(*, grounding_query: str, task_instruction: str, width: int, height: int, top_k: int) -> str:
    return (
        f"Grounding query: {grounding_query}\n"
        f"Task context: {task_instruction}\n"
        f"Image size: {width} x {height}\n"
        f"Return at most {top_k} candidate(s)."
    )


def build_collage_layout_user_prompt(
    *,
    task_instruction: str,
    layout_goal: str,
    source_context: str,
    max_canvas_width: int,
    max_canvas_height: int,
    max_canvas_pixels: int,
) -> str:
    return (
        f"Task instruction: {task_instruction}\n"
        f"Layout goal: {layout_goal}\n\n"
        f"Source images:\n{source_context}\n\n"
        "Create a clear collage reference image for a downstream image-editing model. "
        "The collage is a hard composition of existing images, not a generative edit. "
        f"Canvas must be <= {max_canvas_width}x{max_canvas_height} and <= "
        f"{max_canvas_pixels} pixels. "
        "For each item, x/y are the top-left paste coordinates after resize/rotation, "
        "width/height are the resized dimensions before rotation, and opacity is within [0, 1]."
    )


def build_evaluate_user_prompt(
    *,
    reference_text: str,
    input_refs: list[str],
    candidate_ref: str,
    instruction: str,
    checks: list[str],
    score_rubric: str = EVALUATE_SCORE_RUBRIC,
) -> str:
    return (
        f"{reference_text}\n"
        f"Task input refs: {input_refs}\n"
        f"Candidate ref: {candidate_ref}\n"
        f"Final edit instruction: {instruction}\n"
        f"Acceptance checks: {checks}\n\n"
        f"Scoring rubric:\n{score_rubric}\n\n"
        "Set is_satisfied=true only when the candidate is ready to pass without further editing. "
        "When is_satisfied=false, explain the main issues and provide new_rewritten_prompt."
    )
