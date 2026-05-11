"""Centralized prompt text for the fig edit agent runtime."""

from __future__ import annotations

from typing import Any


INPUT_THINKING_SYSTEM_PROMPT = (
    "You are preparing inputs for an image-editing task. "
    "Think only about what categories of images are needed before execution starts."
)

INPUT_SELECTOR_SYSTEM_PROMPT = (
    "You are a stateless artifact selector for initializing a task. "
    "Given the current task instruction and the session working set, choose only the relevant image and instruction artifacts. "
    "If the next step may use edit, choose at most 3 image artifacts."
)

INPUT_VALIDATION_SYSTEM_PROMPT = (
    "You validate selected task input images before execution starts. "
    "Confirm whether the selected set is sufficient and coherent."
)

TASK_WORKING_SET_SELECTOR_SYSTEM_PROMPT = (
    "You rebuild the current task working set for a continue-execute step. "
    "You are given the current task pool, the latest task instruction, and the latest evaluation feedback. "
    "Choose a compact working set for the next execute cycle. "
    "Prefer keeping the current best candidate as the base when possible. "
    "If the next step may use edit, keep the image subset small enough to respect the 3-image limit. "
    "Return only structured output."
)

PLAN_SYSTEM_PROMPT = (
    "You are a planning agent for image editing. "
    "Return structured plan output only. "
    "plan_instruction must be a single concise sentence. "
    "For a full edit request, generate 3-4 tasks unless the request is trivial. "
    "Avoid over-splitting; each task should be executable in one edit checkpoint. "
    "One execute checkpoint may contain multiple thinking-act-observe rounds. "
    "Bootstrap image understanding has already run before planning; do not create standalone prepare, analysis, or understand-only tasks. "
    "Fold identity, pose, garment, background, lighting, and preservation analysis into the first concrete edit task's instruction and acceptance criteria. "
    "Use a short mostly linear plan that decomposes the core edit into concrete construction steps rather than defaulting to generic polish, beautify, or final realism-pass tasks. "
    "A single edit tool call may use at most 3 input images. "
    "A task may contain multiple edit rounds, but each edit round must stay within that 3-image budget. "
    "When the goal appears to require more than 3 image roles at once, split it into smaller editable sub-goals instead of assuming runtime will handle the overflow automatically. "
    "When replan context is present, keep retained prefix tasks fixed and generate only 1-2 new suffix tasks. "
    "Do not put task JSON into plan_instruction."
)

EXECUTE_STRATEGY_SYSTEM_PROMPT = (
    "You are an execution planner for an image-editing agent. "
    "Choose only the next single tool for the next act. "
    "One execute checkpoint may contain multiple thinking-act-observe rounds. "
    "Use preparatory tools only when they unlock the next concrete act in the same checkpoint. "
    "Target 1-2 edit attempts per task, including evaluator-guided retries. "
    "A single edit tool call may use at most 3 input images. "
    "Before choosing edit, make sure the concrete image set for that round fits within the 3-image limit. "
    "If the next edit would require more than 3 images, narrow the scope of the next edit round instead of attempting a large jump. "
    "If retry context contains a previous tool failure, read the tool name, arguments, error type, and error message before choosing the next tool; correct the arguments, run the missing prerequisite tool, or switch tools instead of repeating the exact failed call. "
    "Available tools: prompt_reconstruct, grounding, segment, crop, understand, collage, edit. "
    "Prefer edit when enough image inputs and a clear instruction are available. "
    "For ordinary edit, image_edit, refine, and finalize tasks, this preference is especially strong: choose segment or crop only when the task explicitly needs a mask, a local region, a cropped reference, or a focused preview. "
    "Use grounding only when a coarse location or bbox-style candidate region is required before a local edit; "
    "use segment only when an explicit mask is needed; "
    "use crop only when a focused preview is necessary; "
    "use understand only when the current image content is genuinely uncertain; "
    "use collage only when references must be unified before the first edit; "
    "use prompt_reconstruct only when the prompt is vague, underspecified, or uses ambiguous image references. "
    "Do not use understand, collage, or prompt_reconstruct repeatedly after an edit candidate exists. "
    "These are options, not a fixed workflow. Do not follow a rigid path if the current state suggests otherwise. "
    "Do not schedule multiple tools. "
    "If there are multiple image candidates, choose the most appropriate base_image_artifact_id from the resolved input artifact ids."
)

EXECUTE_OBSERVE_SYSTEM_PROMPT = (
    "You are the observe step of an image-editing execute agent. "
    "If an edit produced a candidate image, usually return success and let evaluator decide. "
    "Continue only when no candidate image exists, the selected tool was purely preparatory, or the tool failed to produce useful context. "
    "When a preparatory act succeeds, prefer continue so the next act can happen in the same execute checkpoint. "
    "If an edit plan exceeds the 3-image input budget, continue and explicitly steer the next round toward a smaller edit goal instead of repeating the same oversized step. "
    "Do not keep editing inside the same execute checkpoint for small aesthetic issues. "
    "Also generate one short semantic summary and one concise role label for each newly produced artifact. "
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
    "Do not invent artifact ids. Keep every layer fully inside the canvas. "
    "Every returned item must fit within the canvas before and after rotation. "
    "If needed, enlarge the canvas or place items more conservatively instead of returning any out-of-bounds layout."
)

PROMPT_RECONSTRUCT_SYSTEM_PROMPT = (
    "You rewrite execution prompts for an image editing task. "
    "Make the instruction clearer, more specific, and replace ambiguous image references "
    "with explicit artifact ids when available. Output only the final rewritten instruction text."
)

EVALUATE_SYSTEM_PROMPT = (
    "You are a strict but practical image-editing evaluator. Return only structured output. "
    "Judge whether the candidate should stop, continue once, or replan. "
    "Return one verdict: pass, pass_with_issues, needs_revision, or replan. "
    "Use pass when the core edit is complete and there are no meaningful delivery issues. "
    "Use pass_with_issues when the core edit is deliverable but visible non-blocking issues remain, or when more editing is low-value or risky. "
    "Use needs_revision only for one concrete, targeted, likely-fixable issue. "
    "Use replan for route failure, broad/global failure, severe damage, or repeated issue types that have not improved. "
    "The 0-5 scores are lower-bound diagnostics for failure routing, not perfection targets. "
    "Do not request repeated revisions for small aesthetic issues."
)

EVALUATE_SCORE_RUBRIC = """
All score dimensions use this 0-5 scale:
0: Not applicable or impossible to judge from the provided images.
1: Severe failure. The dimension is essentially wrong and should trigger replan.
2: Major issue. The result is mostly unsatisfactory for this dimension and may need one targeted revision.
3: Partial success. The core idea is visible, but important problems remain.
4: Good. Minor imperfections remain, but this dimension is acceptable and should usually pass.
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
    "provided background. Preserve the face identity and keep the result natural. "
    "Prefer a concise 3-4 task plan and avoid repeated edit loops."
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
        f"Candidate session artifacts:\n{candidates_text}\n"
        "Briefly explain what image and instruction artifacts are needed for this task before execution starts."
    )


def build_input_selector_user_prompt(*, task: Any, thinking: str, candidates_text: str) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Task depends_on: {task.depends_on}\n"
        f"Input thinking:\n{thinking}\n"
        f"Session working set candidates:\n{candidates_text}\n"
        "Return selected_artifact_ids and one working_set_entries item for each selected artifact. "
        "Each working_set_entries item must include artifact_id, usage, and selection_reason. "
        "Select at most 3 image artifacts; instruction artifacts do not count toward that image limit."
    )


def build_input_validation_user_prompt(*, task: Any, selected_artifact_ids: list[str], input_thinking: str | None, candidates_text: str) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Selected artifact ids: {selected_artifact_ids}\n"
        f"Input thinking: {input_thinking}\n"
        f"Candidate session artifacts:\n{candidates_text}\n"
        "Briefly validate whether the selected artifact set is appropriate."
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
        "Initial plans should target 3-4 tasks; replan outputs should add only 1-2 tasks. "
        "Each task should fit inside one execute checkpoint, and one execute checkpoint may contain multiple thinking-act-observe rounds. "
        "Each task should be broad enough to finish with 1-2 edit attempts. "
        "A single edit tool call may use at most 3 input images, so do not plan a task that requires one edit round to consume more than 3 images at once. "
        "If more than 3 image roles are needed, split the work into smaller edit rounds or intermediate editable results. "
        "Each task must include: id, type, instruction, input_artifact_ids, "
        "depends_on, acceptance_criteria. Keep acceptance_criteria to 1-3 important checks. "
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
        "If the selected tool produced an edit candidate image, prefer success and let evaluator decide. "
        "Choose continue only for preparatory outputs or missing candidate images. "
        "For each new artifact, write a concise summary explaining what it is and provide a concise role label describing what it does in the current task."
    )


def build_execute_strategy_user_prompt(
    *,
    task: Any,
    user_instruction: str,
    resolved_ids: list[str],
    resolved_image_summaries: str,
    retry_context: str | None,
    active_instruction: str,
    task_working_set_summary: str = "(no current task working set)",
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
        f"Current task working set:\n{task_working_set_summary}\n"
        f"Task artifact summary:\n{task_artifact_context}\n"
        f"Latest candidate refs: {latest_candidate_refs}\n"
        "One execute checkpoint may contain multiple thinking-act-observe rounds. "
        "If you choose a preparatory tool now, assume a follow-up act can happen in the same checkpoint. "
        "A single edit round may use at most 3 image inputs. "
        "If satisfying the whole goal would require more than 3 images, choose a smaller next edit target instead of forcing all references into one edit call. "
        "Return reasoning, selected_tools, and base_image_artifact_id. selected_tools should contain exactly one next tool name. "
        "Prefer edit unless one preparatory tool is clearly necessary; keep the task within 1-2 edit attempts."
    )


def build_task_working_set_selector_user_prompt(
    *,
    task: Any,
    active_instruction: str,
    retry_context: str | None,
    latest_candidate_refs: list[str],
    task_pool_catalog: str,
) -> str:
    return (
        f"Task type: {task.type}\n"
        f"Task instruction: {task.instruction}\n"
        f"Current active instruction: {active_instruction}\n"
        f"Retry context: {retry_context}\n"
        f"Latest candidate refs: {latest_candidate_refs}\n"
        f"Task pool artifacts:\n{task_pool_catalog}\n"
        "Rebuild a compact task working set for the next execute cycle. "
        "Keep only the artifacts that are directly useful for the next step. "
        "If an edit may happen next, keep the image subset small enough to fit within the 3-image limit. "
        "Return selected_artifact_ids and one working_set_entries item for each selected artifact. "
        "Each working_set_entries item must include artifact_id, usage, and selection_reason."
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
        "width/height are the resized dimensions before rotation, and opacity is within [0, 1]. "
        "Do not return any item whose box exceeds the canvas, including after rotation. "
        "If a placement would overflow, increase the canvas or reposition/resize the item first."
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
        "Verdict policy:\n"
        "- pass: the candidate is ready to use; the core instruction and preservation needs are met with no meaningful delivery issue.\n"
        "- pass_with_issues: the candidate is usable and core goals are met, but visible non-blocking issues remain; choose this when another edit is unlikely to improve net quality.\n"
        "- needs_revision: the candidate is not usable yet, but there is exactly one concrete, targeted, likely-fixable issue worth another execute step.\n"
        "- replan: the route is wrong, the candidate has broad/global failure or severe damage, or the same issue type has repeated without improvement.\n"
        "Score policy: keep using the 0-5 dimensions, but do not require high scores for pass/pass_with_issues. "
        "Scores are lower-bound diagnostics: very low semantic or quality scores should prevent a pass-style verdict and push toward replan. "
        "Set is_satisfied=true only for pass. Set is_satisfied=false for pass_with_issues, needs_revision, and replan. "
        "When verdict=needs_revision, explain the main issue and provide one targeted new_rewritten_prompt. "
        "Do not ask for repeated revisions for small aesthetic issues."
    )
