"""Minimal tool registry for runtime use."""

from __future__ import annotations

from schema import ToolName
from .base import BaseTool

from .crop_tool import CropTool
from .collage_tool import CollageTool
from .edit_tool import EditTool
from .evaluate_tool import EvaluateTool
from .grounding_tool import GroundingTool
from .prompt_reconstruct_tool import PromptReconstructTool
from .segment_tool import SegmentTool
from .understand_tool import UnderstandTool

class ToolRegistry:
    """Name-to-tool registry."""

    def __init__(self, tools: dict[ToolName, BaseTool] | None = None) -> None:
        self._tools: dict[ToolName, BaseTool] = tools if tools is not None else {
            ToolName.UNDERSTAND: UnderstandTool(),
            ToolName.GROUNDING: GroundingTool(),
            ToolName.SEGMENT: SegmentTool(),
            ToolName.CROP: CropTool(),
            ToolName.COLLAGE: CollageTool(),
            ToolName.PROMPT_RECONSTRUCT: PromptReconstructTool(),
            ToolName.EDIT: EditTool(),
            ToolName.EVALUATE: EvaluateTool(),
        }

    def get(self, name: ToolName) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"tool is not registered: {name}")
        return self._tools[name]


def build_default_tool_registry() -> ToolRegistry:
    return ToolRegistry()
