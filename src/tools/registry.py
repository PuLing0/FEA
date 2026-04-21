"""Minimal tool registry for runtime use."""

from __future__ import annotations

from typing import TYPE_CHECKING

from schema import ToolName

from .crop_tool import CropTool
from .collage_tool import CollageTool
from .edit_tool import EditTool
from .evaluate_tool import EvaluateTool
from .grounding_tool import GroundingTool
from .prompt_reconstruct_tool import PromptReconstructTool
from .segment_tool import SegmentTool
from .understand_tool import UnderstandTool

if TYPE_CHECKING:
    from .base import ToolExecutionResult


class ToolRegistry:
    """Name-to-tool registry."""

    def __init__(self) -> None:
        self._tools = {
            ToolName.UNDERSTAND: UnderstandTool(),
            ToolName.GROUNDING: GroundingTool(),
            ToolName.SEGMENT: SegmentTool(),
            ToolName.CROP: CropTool(),
            ToolName.COLLAGE: CollageTool(),
            ToolName.PROMPT_RECONSTRUCT: PromptReconstructTool(),
            ToolName.EDIT: EditTool(),
            ToolName.EVALUATE: EvaluateTool(),
        }

    def get(self, name: ToolName):
        return self._tools[name]


def build_default_tool_registry() -> ToolRegistry:
    return ToolRegistry()
