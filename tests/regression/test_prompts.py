from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_prompt_module_builds_plan_prompt() -> None:
    from runtime.prompts import PLAN_SYSTEM_PROMPT, build_plan_user_prompt

    prompt = build_plan_user_prompt(
        instruction_text="把人物放到背景里",
        replan_context="Replan mode: split_task\n",
        retained_prefix_plan="Retained prefix task ids: ['task_001']\n",
        available_artifacts="Available artifacts for planning:\n- art_image_001\n",
        image_artifact_ids=["art_img_input_001"],
        understanding_summaries=[{"summary": "人物参考图"}],
    )

    assert "planning agent" in PLAN_SYSTEM_PROMPT
    assert "3-4 tasks" in PLAN_SYSTEM_PROMPT
    assert "at most 3 input images" in PLAN_SYSTEM_PROMPT
    assert "multiple thinking-act-observe rounds" in PLAN_SYSTEM_PROMPT
    assert "Avoid over-splitting" in PLAN_SYSTEM_PROMPT
    assert "Bootstrap image understanding has already run before planning" in PLAN_SYSTEM_PROMPT
    assert "do not create standalone prepare" in PLAN_SYSTEM_PROMPT
    assert "core edit into concrete construction steps" in PLAN_SYSTEM_PROMPT
    assert "generic polish" in PLAN_SYSTEM_PROMPT
    assert "Instruction: 把人物放到背景里" in prompt
    assert "Available artifacts for planning" in prompt
    assert "1-2 edit attempts" in prompt
    assert "at most 3 input images" in prompt
    assert "multiple thinking-act-observe rounds" in prompt
    assert "Do not invent future artifact ids" in prompt


def test_prompt_module_builds_evaluate_prompt_with_verdict_policy() -> None:
    from runtime.prompts import build_evaluate_user_prompt

    prompt = build_evaluate_user_prompt(
        reference_text="Image 1 is reference. Image 2 is candidate.",
        input_refs=["art_img_input_001"],
        candidate_ref="art_image_001",
        instruction="保持人物身份",
        checks=["身份一致"],
    )

    assert "Candidate ref: art_image_001" in prompt
    assert "pass_with_issues" in prompt
    assert "Return only verdict and reason" in prompt
    assert "Do not produce numeric scores" in prompt
    assert "same issue type has repeated" in prompt


def test_prompt_module_includes_execute_convergence_guidance() -> None:
    from runtime.prompts import (
        EXECUTE_OBSERVE_SYSTEM_PROMPT,
        EXECUTE_STRATEGY_SYSTEM_PROMPT,
        build_execute_observe_user_prompt,
        build_execute_strategy_user_prompt,
    )

    task = Task(
        id="task_prompt",
        plan_id="plan_prompt",
        type="reference_edit",
        instruction="把人物放到背景里",
    )
    strategy_prompt = build_execute_strategy_user_prompt(
        task=task,
        user_instruction="生成最终图",
        resolved_ids=["art_img_input_001"],
        resolved_image_summaries="- art_img_input_001: 人物",
        retry_context=None,
        active_instruction="把人物放到背景里",
        task_artifact_context="(none)",
        latest_candidate_refs=[],
    )
    observe_prompt = build_execute_observe_user_prompt(
        task=task,
        active_instruction="把人物放到背景里",
        retry_context=None,
        selected_tool="edit",
        tool_args={"image_refs": ["art_img_input_001"]},
        source_lines=[],
        new_artifact_lines=["- art_image_001 | kind=image"],
    )

    assert "multiple thinking-act-observe rounds" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "Target 1-2 edit attempts per task" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "at most 3 input images" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "Prefer edit" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "For ordinary edit, image_edit, refine, and finalize tasks" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "choose segment or crop only when the task explicitly needs" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "let evaluator decide" in EXECUTE_OBSERVE_SYSTEM_PROMPT
    assert "same execute checkpoint" in EXECUTE_OBSERVE_SYSTEM_PROMPT
    assert "3-image input budget" in EXECUTE_OBSERVE_SYSTEM_PROMPT
    assert "multiple thinking-act-observe rounds" in strategy_prompt
    assert "1-2 edit attempts" in strategy_prompt
    assert "at most 3 image inputs" in strategy_prompt
    assert "prefer success and let evaluator decide" in observe_prompt


def test_collage_prompt_emphasizes_in_bounds_layout() -> None:
    from runtime.prompts import (
        COLLAGE_LAYOUT_SYSTEM_PROMPT,
        build_collage_layout_user_prompt,
    )

    prompt = build_collage_layout_user_prompt(
        task_instruction="整理参考图",
        layout_goal="make a clean reference board",
        source_context="1. artifact_id=art_face_001; size=20x30",
        max_canvas_width=4096,
        max_canvas_height=4096,
        max_canvas_pixels=16_777_216,
    )

    assert "after rotation" in COLLAGE_LAYOUT_SYSTEM_PROMPT
    assert "out-of-bounds layout" in COLLAGE_LAYOUT_SYSTEM_PROMPT
    assert "including after rotation" in prompt
    assert "increase the canvas or reposition/resize the item first" in prompt
