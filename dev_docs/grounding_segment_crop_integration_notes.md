# Grounding Segment Crop Integration Notes

本文记录本轮围绕 `grounding / segment / crop` 三个 tool 的真实接入过程中，已经遇到的问题、明确的设计结论、以及尚未完全打通的阻塞点。重点是把这轮讨论里沉淀下来的工程判断固定下来，避免后续重复走弯路。

## 1. Grounding 不是纯检测，而是语义定位

### 问题

最初的 `GroundingTool` 只是返回硬编码 bbox：

- 不看图
- 不理解语义
- 不支持后续 `segment` 和 `crop` 真正消费

同时，关于 grounding 输入语义，最开始有过几种拆分方案：

- `target_description`
- `grounding_instruction`
- `task_instruction`

这几项语义边界不清，容易重复。

### 结论

将外部输入统一收敛为：

```python
GroundingArgs(
    image_ref: str,
    grounding_query: str,
    top_k: int | None = 1,
)
```

其中：

- `grounding_query` 是唯一对外语义输入
- 内部 prompt 可以额外使用 task 上下文，但不再拆成多个外部字段

### 当前实现

- `src/schema/tools.py`
- `src/tools/grounding_tool.py`

### 结果

grounding 现在是：

- bbox-first
- 可选 positive / negative points
- 多模态真实看图
- 结果写入 `GeometryArtifact`

并且已经用真实图片跑通过一次：

- 输入 `examples/fig1.jpg`
- 返回 `GeometryArtifact`
- payload 中包含真实 `bbox`、`score`、`positive_points`

## 2. Grounding 输出里哪些字段是必要的

### 问题

最开始设计过更丰富的 candidate payload，例如：

- `label`
- `bbox`
- `score`
- `rationale`
- `positive_points`
- `negative_points`

以及 artifact payload 中还考虑过：

- `target_description`
- `grounding_instruction`
- `task_instruction`

这些字段里有些是调试友好，但对当前链路并不是必须。

### 结论

grounding candidate 保留：

- `label`
- `bbox`
- `score`
- `positive_points`
- `negative_points`

去掉：

- `rationale`

artifact payload 只保留：

```python
{
    "image_artifact_id": ...,
    "grounding_query": ...,
    "candidates": [...],
}
```

去掉：

- `target_description`
- `grounding_instruction`
- `task_instruction`

### 原因

- `rationale` 主要是调试信息，不服务当前主链路
- `bbox` 是后续 `crop` 和 `segment` 的关键几何信息
- `positive_points / negative_points` 对后续 `segment` 很有价值
- payload 要尽量简洁，避免重复写 prompt 语义

## 3. Crop 的两条分支语义必须分开

### 问题

最开始尝试把 `mask` 和 `grounding` 都统一抽象成 “bbox -> crop”。

这个思路对 `grounding` 分支成立，但对 `mask` 分支是不准确的：

- `grounding + crop` 的确是矩形裁剪
- `mask + crop` 的目标不是矩形裁剪，而是不规则 cutout

### 结论

`CropTool` 必须有两条明确分支：

#### 1. `mask_ref` 分支

- 先从 mask 非空区域求最小包围矩形
- 该矩形只用于确定输出范围
- 最终用 mask 作为 alpha 通道
- 输出透明背景的不规则 cutout 图

#### 2. `grounding_ref` 分支

- 直接使用 grounding bbox
- 输出普通矩形 crop preview 图

### 当前实现

- `src/schema/tools.py`
- `src/tools/crop_tool.py`
- `src/agents/execute_agent.py`

### 结果

`CropArgs` 已变为：

```python
CropArgs(
    image_ref,
    mask_ref=None,
    grounding_ref=None,
    padding=0,
)
```

并且：

- `mask_ref` 和 `grounding_ref` 至少一个存在
- 两者都在时，优先 `mask_ref`
- 输出图片在 agent 运行中会保存到该 run 目录的 `artifacts/crop/`；直接调用工具且未提供 `output_dir` 时回退到 `generated/crop/`
- 结果注册为新的 `ImageArtifact`

## 4. Crop 和 Grounding / Mask 的来源校验必须存在

### 问题

如果不校验来源图，可能出现：

- 用 A 图的 mask 去裁 B 图
- 用 A 图的 grounding 去裁 B 图

这种错误不会立刻报崩，但结果一定是错的。

### 结论

在 `CropTool` 中必须校验：

#### 对 `mask_ref`

```python
mask.payload["image_ref"] == args.image_ref
```

#### 对 `grounding_ref`

```python
geometry.payload["image_artifact_id"] == args.image_ref
```

### 当前实现

- `src/tools/crop_tool.py`

### 结果

来源不匹配会直接报错，不再允许“默默裁错图”。

## 5. Segment 不应该是写死 SAM3 的单一工具

### 问题

用户明确要求：

- 当前接入 SAM 3.1
- 后续要预留多模型接口

如果把 `SegmentTool` 写成完全 SAM3 特化，就会在后面扩模型时返工。

### 结论

`SegmentTool` 要做成：

- 工具层统一编排
- 后端层可切换
- 当前默认后端是 `sam31`

对外接口：

```python
SegmentArgs(
    image_ref,
    target,
    grounding_ref,
    backend_name=None,
)
```

内部接口：

- `_predict_candidates(...)`
- `_predict_candidates_via_sam31(...)`

后续别的模型只需要接到 `_predict_candidates(...)` 的统一候选协议，不需要重写 `SegmentTool`。

### 当前实现

- `src/schema/tools.py`
- `src/tools/segment_tool.py`

### 结果

当前 `SegmentTool` 已经具备：

- `grounding_ref` 输入
- `backend_name` 端口
- 默认 `"sam31"` 路径
- 工具内统一候选筛选

## 6. Segment 里 points 是主提示，bbox 是辅助 ROI

### 问题

关于 `segment` 如何消费 grounding，曾经有两种方向：

- bbox-first
- points-first

如果只用 bbox，分割质量会受限；而 reference `image_segment.py` 的实现本质上是 point-guided segmentation。

### 结论

对当前项目，采用：

- `positive_points / negative_points` 作为主提示
- `bbox` 作为 ROI hint

这和 grounding 的最终输出设计是一致的：

- grounding 负责语义定位
- segment 负责精确 mask

### 当前实现

- `src/tools/segment_tool.py`

### 结果

segment 读取 grounding 时会取：

- `bbox`
- `positive_points`
- `negative_points`

其中：

- points 用于主约束
- bbox 用于后端定位辅助

## 7. GrabCut 不再作为开关，而是内置质量路径

### 问题

最开始参考 `fig-edit-agent` 时，`refinement_mode` 有：

- `grabcut`
- `none`

但这轮讨论里已经明确：

- 用户不希望把 refinement 暴露成二选一开关
- 用户已有经验表明 GrabCut 提升明显

### 结论

`SegmentTool` 的 refinement 固定为：

```text
SAM 3.1 候选 -> GrabCut refinement -> 候选筛选 -> 最终 mask
```

也就是说：

- 不对外暴露 `refinement_mode`
- 不支持 `none`
- GrabCut 直接作为默认质量路径内置

### 当前实现

- `src/tools/segment_tool.py`

### 结果

当前代码已经按这个原则组织：

- 候选先经过 `_predict_candidates`
- 再经过 `_refine_candidates_with_grabcut`
- 再进入 `_select_best_candidate`

## 8. Segment 输出 payload 应该尽量轻

### 问题

一开始曾经考虑在 `MaskArtifact.payload` 中保留更多字段：

- `backend_name`
- `refinement`
- `selected_candidate_name`
- `bbox`

但这些字段对当前主链路不是必需，且会使 payload 变重。

### 结论

最终 `MaskArtifact.payload` 只保留：

```python
{
    "image_ref": args.image_ref,
    "grounding_ref": args.grounding_ref,
    "target": args.target,
    "positive_points": [...],
    "negative_points": [...],
    "mask_score": 0.93,
}
```

其中：

- 实际 mask 图像放在 `MaskArtifact.uri`
- 其他几何信息如 bbox 后续需要时可重新从 mask 文件推导

### 当前实现

- `src/tools/segment_tool.py`

## 9. 多模态链路的真实阻塞不在 LLM，而在中间产物是否为真实本地图

### 问题

在最初接入多模态 `understand / evaluate / observe` 之后，完整 graph 运行很快暴露出一个核心问题：

- `understand` 可以看本地输入图
- 但后续 `edit / crop / segment` 如果只产出 `store://generated/...`
- 多模态模型就无法继续读取这些中间结果

最早暴露出的错误是：

```text
Observe image artifact uri is not a local file path: store://generated/...
```

### 结论

只要一个 tool 生成了“后续需要被看图”的图像结果，它就必须：

- 真实写本地文件
- 在 artifact `uri` 中给出本地路径

### 当前实现状态

- `GroundingTool`: 已真实运行
- `CropTool`: 已真实落本地图片
- `SegmentTool`: 已真实落本地 mask 文件
- `EditTool`: 仍然是 stub，是目前完整链路继续往前走的主要阻塞点

## 10. 环境阻塞与依赖问题

### 问题

这轮真实接入过程中，环境层面出现过多次真实阻塞：

- 机器默认 Python 只有 3.10，而项目要求 `>=3.11`
- `uv` 初始未安装
- `.venv` 需要重新创建
- `pytest`、`pytest-mock` 最初未进入 `.venv`
- `Pillow` 最初未装，导致真实 `CropTool` 无法读取图片
- `numpy` 最初未装，导致真实 `SegmentTool` 无法导入
- 真正 SAM 3.1 路径还缺：
  - `torch`
  - `torchvision`
  - `opencv-python-headless`

### 当前状态

已完成：

- `.venv` 创建
- `uv` 安装
- `pytest` 环境可用
- `Pillow` / `numpy` / `huggingface_hub` / `opencv-python-headless` 已写入项目依赖
- grounding / crop / segment 的接口级测试已可运行

仍未完全完成：

- `torch`
- `torchvision`

这两个是切换到真实 SAM 3.1 的硬阻塞。

## 11. 循环导入问题是真实的工程问题，不是测试假象

### 问题

在做 `GroundingTool` / `CropTool` / `SegmentTool` 的局部 smoke test 时，多次遇到：

```text
ImportError: partially initialized module ...
```

原因是仓库里存在：

```text
tools -> runtime -> agents -> tools
```

以及包级 `__init__` 做了过度重导出。

### 解决方案

已经做了这些收敛：

- `agent.py` 直接从 `runtime.graph` 导 graph builder
- `runtime.graph` 直接从具体 `agents.*` 导入
- `runtime.input_selector` 直接从 `tools.registry` 导入
- `agents` 侧直接依赖 `tools.registry`
- `runtime.__init__` 不再包级导出 `build_runtime_graph`
- 测试里改为从 `runtime.graph` 直接导入

### 结果

局部导入 smoke test 已恢复，例如：

```python
from tools.crop_tool import CropTool
from tools.grounding_tool import GroundingTool
```

可以成功导入。

## 12. 目前的真实完成度

### 已经真实可用

- `GroundingTool`
  - 多模态真实看图
  - 返回真实 geometry artifact

- `CropTool`
  - mask 分支输出不规则 cutout
  - grounding 分支输出矩形 preview
  - 输出本地图片文件

- `SegmentTool`
  - 接 grounding
  - 产出本地 mask 文件
  - 内置候选筛选和 GrabCut 路径

### 仍未完全真实打通

- `SegmentTool` 当前默认路径已经预留真实 SAM 3.1 接口，但缺：
  - `torch`
  - `torchvision`
  - 实际 checkpoint 运行环境确认

- `EditTool` 仍然是 stub
  - 这是完整链路里最大的剩余功能缺口

## 13. 这轮沉淀下来的几个关键工程判断

### 1. artifact 要尽量轻，文件本体走 `uri`

- mask/image 等重内容不要塞进 payload
- payload 只留必要 linkage 和轻量 metadata

### 2. 真正需要被多模态模型“看”的图，必须落本地文件

- `store://...` 只适合 stub 阶段
- 一旦进入真实多模态链路，必须写真实图片文件

### 3. grounding / segment / crop 三者的职责边界要明确

- grounding：语义定位
- segment：像素级 mask
- crop：基于 mask 或 grounding 产出局部图像

### 4. 工具链不要把“质量路径”交给调用者决策

- GrabCut 如果已经明确有价值，就直接内置
- 不要过早暴露很多模式开关

### 5. 后端要抽象，但 payload 不要为了抽象而膨胀

- `backend_name` 这种可作为 tool 内部扩展口
- 不一定需要落到 artifact payload 里

## 14. 下一步最合理的工作

按当前状态，后续最值得继续推进的是：

1. 完成真实 SAM 3.1 运行依赖安装
2. 跑一条真实 `grounding -> segment -> crop`
3. 再推进 `EditTool` 的真实接入

如果这三步完成，当前项目就会从“runtime skeleton + 半真实工具”进入“真正能跑出中间产物”的阶段。
