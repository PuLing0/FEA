# Runtime Replan And Input Selection Development Notes

本文记录本轮开发中遇到的主要问题、判断过程和最终解决方案。重点覆盖 schema 简化、execute agent 的 thinking-act-observe 循环、artifact summary、replan、以及 task 启动前的输入图片选择。

## 1. Planner 的任务 schema 太松

### 问题

最开始 `PlanLLMOutput.tasks` 是 `list[dict[str, Any]]`。

这个设计的问题是：

- LLM 只知道 `tasks` 是一组字典，但不知道每个 task 必须包含哪些字段。
- 复杂规划或 replan 时，模型容易把 task JSON 塞进 `plan_instruction`。
- task 字段不稳定，后续 runtime 很难可靠 materialize 成 `Task`。

### 解决方案

增加明确的嵌套 schema：

- `PlanTaskSpec`
  - `id`
  - `type`
  - `instruction`
  - `input_artifact_ids`
  - `depends_on`
  - `acceptance_criteria`

并将：

```python
PlanLLMOutput.tasks: list[dict[str, Any]]
```

改为：

```python
PlanLLMOutput.tasks: list[PlanTaskSpec]
```

### 当前实现

- `src/schema/runtime.py`
- `src/schema/__init__.py`
- `src/agents/plan_agent.py`

### 结果

Planner 输出更稳定，runtime 可以直接校验和 materialize task。

## 2. `target_artifact_ids` 和 `input_artifact_ids` 语义重复

### 问题

原来的 `Task` 同时包含：

- `input_artifact_ids`
- `target_artifact_ids`

实际使用中，`target_artifact_ids` 经常被当作“base image”或“未来产物引用”，导致语义混乱：

- target artifact 本应是生成结果，但在执行前结果尚不存在。
- execute agent 用它判断 base image，会和 runtime 输入选择冲突。
- replan 中也容易提前绑定未来 artifact。

### 解决方案

删除 `Task.target_artifact_ids`，只保留：

```python
Task.input_artifact_ids
```

但更重要的是：task 的真实执行输入不依赖 planner 预填，而是在 task 开始前由 `prepare_task_inputs(...)` 统一选择。

### 当前实现

- `src/schema/runtime.py`
- `src/agents/execute_agent.py`
- `src/agents/plan_agent.py`
- `tests/regression/`

## 3. Replan 不应该提前绑定 task 输入

### 问题

replan fallback 之前会把 `preserve_artifact_ids` 写进新 task 的 `input_artifact_ids`。

这会导致：

- replan 和 initial plan 的行为不一致。
- planner 提前决定每个 task 用哪些图，而不是让 task 启动前根据当前 artifact 池选择。
- preserved artifact 可能被所有新 task 重复绑定，造成输入过重。

### 解决方案

replan suffix task 默认：

```python
input_artifact_ids = []
```

`Plan.input_artifact_ids` 保留 plan 级可见 artifact 池，例如：

- 原始输入图
- preserved image artifacts

真实 task 输入在 task 开始前统一通过 `prepare_task_inputs(...)` 从候选池中选择。

### 当前实现

- `src/agents/plan_agent.py`
  - `_build_replan_fallback_tasks(...)`
  - `_build_plan_with_llm(...)`
  - `_validate_llm_plan_output(...)`

### 额外约束

LLM 如果仍然返回非空 `input_artifact_ids`：

- 空列表允许。
- 非空时必须属于当前可见 artifact pool。
- 非法 artifact id 会触发 fallback。

对应测试：

- `test_plan_agent_replan_rejects_illegal_llm_input_artifact_id`

## 4. Replan 需要保留已通过 task，而不是重建整个 plan

### 问题

失败后如果完全重新规划，容易丢掉已经完成且通过 evaluator 的成果。

例如：

- `task_001` 已经成功合成人物主体。
- `task_002` 入景失败。
- 重新规划时不应该重新生成人物主体。

### 解决方案

replan 流程改为：

1. 找到旧 plan。
2. 机械筛选出 `PASSED` task 作为 retained prefix。
3. failed/running task 标记为 `REPLANNED`。
4. pending future task 标记为 `ABANDONED`。
5. LLM 只生成新的 suffix tasks。
6. 新 plan = retained prefix tasks + new suffix tasks。

### 当前实现

- `src/agents/plan_agent.py`
  - `_apply_replan_plan(...)`
  - `_build_retained_prefix_task_ids(...)`
  - `_mark_old_plan_tasks_for_replan(...)`
  - `_materialize_tasks_from_specs(...)`

### 结果

replan 后的 plan 仍然保留旧成果，同时替换失败路径。

## 5. Replan prompt 缺少失败原因和 artifact summary

### 问题

如果 replan prompt 只告诉 LLM “重新规划”，但不告诉它：

- 哪个 task 失败了
- 为什么失败
- 哪些问题需要规避
- 当前有哪些可复用 artifact
- 每个 artifact 是什么

那么 replan 没有实际意义，模型容易生成和旧路线类似的 plan。

### 解决方案

replan prompt 中加入：

- replan mode
- replan reason
- replaced task id
- decision issues
- source execution outcome
- preserve artifact ids
- retained prefix tasks
- available new task ids
- available artifacts with summary

其中 artifact summary 来自 `Artifact.summary`。

### 当前实现

- `src/agents/plan_agent.py`
  - `_build_plan_with_llm(...)`
  - `_render_replan_artifact_catalog(...)`

## 6. Artifact 需要稳定 summary，不能每次临时理解

### 问题

任务输入选择、replan、execute strategy 都需要知道 artifact 的语义。

如果每次都临时调用 understand：

- 成本高。
- 慢。
- 同一 artifact 的理解可能不稳定。
- replan prompt 构造复杂。

### 解决方案

在 `ArtifactBase` 上增加轻量字段：

```python
summary: str | None = None
```

这个字段由 LLM 在 observe 阶段生成并注入 artifact。

### 当前实现

- `src/schema/artifacts.py`
- `src/schema/runtime.py`
- `src/agents/execute_agent.py`

## 7. Artifact summary 应该在 observe 中生成

### 问题

summary 如果在 tool 内部生成，会导致：

- 每个 tool 都要接 LLM。
- 工具层变重。
- 同一个 tool 是否需要 summary 的逻辑分散。

### 解决方案

把 summary 生成放到 execute agent 的 observe 阶段。

每轮 act 后，observe 一次性返回：

- `outcome`
  - `continue`
  - `success`
- `observation`
- `artifact_summaries`

这样一次 LLM 请求同时完成：

- 判断当前 execute loop 是否可以结束。
- 生成新 artifact 的语义 summary。

### 当前实现

新增 schema：

- `ObserveArtifactSummary`
- `ObserveLLMOutput`

执行逻辑：

- `ExecuteAgent._observe_with_llm(...)`
- `ExecuteAgent._apply_observe_artifact_summaries(...)`

对应测试：

- `test_observe_llm_output_construct`
- `test_execute_agent_observe_injects_artifact_summary`

## 8. ExecuteAgent 的 stop 条件不能等同于“edit 出图”

### 问题

之前逻辑是：只要 `edit` 产出图片，就认为 execute 成功。

这不符合 thinking-act-observe 的模型：

- edit 之后也可能还需要 crop、understand、prompt reconstruct 或继续修正。
- 非 edit tool 也可能产出关键中间 artifact。
- 是否停止应该由 observe 判断。

### 解决方案

execute loop 中每轮只执行一个 tool。

每轮顺序是：

```text
thinking -> act(one tool) -> observe
```

是否结束由 observe 的 `outcome` 决定：

- `continue`：继续下一轮。
- `success`：execute checkpoint 通过，交给 evaluator。

如果超过最大轮数，则由系统强制进入 execute failure，并触发 replan。

### 当前实现

- `src/agents/execute_agent.py`

## 9. Base image 选择不能依赖已删除的 target artifact

### 问题

删除 `target_artifact_ids` 后，execute agent 仍然需要知道 edit 时谁是 base image。

如果简单取第一个输入图，容易选错：

- 人物入景任务中，背景图通常是 base image。
- retry 时，上次失败候选图应该优先作为 base。

### 解决方案

增加 `ExecuteLLMOutput.base_image_artifact_id`。

base image 选择优先级：

1. evaluator retry 指定的 `base_candidate_artifact_id`
2. LLM strategy 返回的 `base_image_artifact_id`
3. resolved input 中第一张 image fallback

同时把 resolved image summaries 提供给 execute strategy prompt，让 LLM 能判断哪张图更适合作为 base。

### 当前实现

- `src/schema/runtime.py`
- `src/agents/execute_agent.py`
  - `_resolve_base_image_ref(...)`
  - `_resolve_retry_base_candidate_ref(...)`
  - `_build_resolved_image_summaries(...)`

对应测试：

- `test_execute_agent_uses_llm_selected_base_image_artifact_id`
- `test_execute_agent_retry_candidate_has_priority_as_base_image`

## 10. Task 输入选择需要统一从 artifact pool 中选择

### 问题

如果每个 task 靠 planner 的 `input_artifact_ids` 固定输入，会出现：

- replan 后新 task 输入过重。
- dependency 输出和 plan-level preserved artifacts 不好统一。
- task 运行前无法根据最新 artifact summary 选择。

### 解决方案

统一使用 `prepare_task_inputs(...)`：

1. 如果 evaluator retry 已提供压缩输入，则优先使用 retry input。
2. 如果已有 resolved input，则直接复用。
3. 构建候选图片池。
4. 确保候选图片有 understanding。
5. 把候选图片转成 compact text。
6. LLM 先生成 input thinking。
7. LLM 基于 thinking 选择 artifact ids。
8. LLM 验证选择是否合理。
9. 写入 `TaskState.resolved_input_artifact_ids`。

### 当前实现

- `src/runtime/input_selector.py`

## 11. Candidate image pool 需要包含 plan-level artifact pool

### 问题

replan 后，新 task 的 `input_artifact_ids=[]`。

但 task 启动前仍然需要看到：

- 原始输入图片
- preserved task output
- 当前 plan 级可见 artifact

如果候选池只来自 session artifact index 或 dependency final output，可能漏掉 plan-level preserved artifacts。

### 解决方案

`build_candidate_image_pool(...)` 合并以下来源：

- session-level image pool
- current plan `input_artifact_ids`
- dependency tasks 的 `final_artifact_id`
- current task-local image artifacts

并且只保留真实存在、kind 为 image 的 artifact，自动去重。

### 当前实现

- `src/runtime/input_selector.py`
  - `build_candidate_image_pool(...)`
  - `_append_image_candidate(...)`

对应测试：

- `test_build_candidate_image_pool_includes_current_plan_inputs`

## 12. Candidate images 文本要保持最小

### 问题

输入选择 prompt 如果把 artifact 的 payload、uri、source_ids 都塞进去，会很重。

用户希望只保留足够判断的信息。

### 解决方案

候选图片文本使用轻量格式：

```text
[artifact_id] summary | 原始输入图片
[artifact_id] summary | source=generated_result
```

当前只包含：

- artifact id
- artifact summary
- 轻量 source label

### 当前实现

- `src/runtime/input_selector.py`
  - `build_candidate_images_text(...)`
  - `_resolve_source_label(...)`

## 13. 原始输入图片不能伪造 source

### 问题

原始输入图本身没有 source。

如果写成：

```text
source=original_input
```

会混淆含义，因为这个字段更适合表示“由某个输入派生而来”。

### 解决方案

对 `art_img_input_*`：

```text
原始输入图片
```

不写 `source=...`。

对生成图：

```text
source=generated_result
```

或者读取 payload 中已有 source。

### 当前实现

- `src/runtime/input_selector.py`

## 14. LLM structured output 适配问题

### 问题

测试过 LangChain 的 `with_structured_output(...)` 多种模式后，OpenAI-compatible provider 对 structured output 的支持并不稳定。

尤其是：

- `json_schema`
- `function_calling`
- provider 自身兼容性

不同供应商表现不一致。

### 解决方案

当前 LLM wrapper 使用：

- LangChain `ChatOpenAI`：普通文本请求。
- PydanticAI：结构化输出请求。

实现位置：

- `src/llm/client.py`

关键函数：

- `invoke_llm(...)`
- `invoke_structured_llm(...)`

## 15. Ad hoc 脚本导入时遇到 circular import

### 问题

直接在脚本中执行：

```python
from agents import PlanAgent
```

曾触发循环导入：

```text
ImportError: cannot import name 'EvaluatorAgent' from partially initialized module 'agents'
```

原因是：

- `agents.__init__` 同时导入多个 agent。
- `EvaluatorAgent` 依赖 `runtime.scheduler`。
- `runtime.__init__` 又导入 `runtime.graph`。
- `runtime.graph` 再导入 `agents`。

### 临时解决方案

真实 LLM 测试脚本中先导入顶层 agent entry：

```python
import agent as _agent_entry
from agents.plan_agent import PlanAgent
```

这样可以复用应用自身的初始化顺序。

### 后续建议

后续可以考虑拆 `runtime.__init__`，不要在 package init 中直接导入 `build_runtime_graph`，减少副作用导入。

## 16. 真实 LLM 测试结果

### 测试场景

输入：

- `art_img_input_001`：上衣单品图
- `art_img_input_002`：裙子单品图
- `art_img_input_003`：人脸参考图
- `art_img_input_004`：古寺背景图
- `art_keep_001`：已通过 task_001 的人物主体结果

replan 目标：

- 保留已完成的人物主体。
- 将背景准备、主体入景、融合润色拆成更稳定的后续任务。

### LLM 生成的新 plan

```text
plan_002
task_001: passed, compose_subject
task_004: prepare_background, inputs=[]
task_005: place_subject_into_scene, inputs=[]
task_006: blend_and_refine, inputs=[]
```

这符合预期：

- passed task 被保留。
- replan suffix tasks 没有提前绑定 input artifacts。

### task 启动前输入选择结果

当前 runnable task：

```text
task_004 prepare_background
```

候选池：

```text
art_img_input_001: 上衣单品图
art_img_input_002: 裙子单品图
art_img_input_003: 人脸参考图
art_img_input_004: 古寺拍照背景图
art_keep_001: 已完成的人物主体结果
```

LLM 最终选择：

```text
selected_artifact_ids = ["art_img_input_004"]
resolved_input_artifact_ids = ["art_img_input_004"]
```

判断合理：

- `prepare_background` 只需要背景图。
- 上衣、裙子、人脸不是背景准备阶段的核心输入。
- `art_keep_001` 可辅助估计人物体量，但不是必需输入。

## 17. 当前提交记录

本轮已按功能拆成三个 commit：

```text
4fab1d1 Refactor planner and execute runtime schemas
ce17641 Use plan artifact pool for task input selection
8b45b96 Cover replan and observe runtime behavior
```

## 18. 验证方式

单测：

```bash
uv run pytest -q tests/regression
```

当前结果：

```text
51 passed
```

真实 LLM 测试结论：

- replan suffix task 输入为空。
- plan-level artifact pool 能进入 task 输入候选池。
- LLM 能基于 summary 选择正确输入。
