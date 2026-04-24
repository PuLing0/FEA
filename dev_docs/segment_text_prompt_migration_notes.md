# Segment Text Prompt Migration Notes

本文记录本轮将 `segment tool` 从 `image + grounding_ref` 迁移到 `image + prompt` 过程中，遇到的新问题、已经验证过的事实、以及对后续迭代最有价值的新思路。

重点不是复述已有设计，而是沉淀这轮真实试验里新暴露出来的工程判断。

## 1. 关键方向变化

### 原方向

之前的主线是：

- `grounding -> segment -> crop`
- `segment` 强依赖 `grounding_ref`
- `grounding` 提供：
  - bbox
  - positive_points
  - negative_points

### 新方向

用户后来明确要求：

- `segment tool` 一开始只需要：
  - `image`
  - `prompt`
- 不再要求 `grounding_ref`

于是本轮将 `SegmentArgs` 外部接口收敛为：

```python
SegmentArgs(
    image_ref: str,
    prompt: str,
    backend_name: str | None = None,
)
```

并同步修改：

- `src/schema/tools.py`
- `src/tools/segment_tool.py`
- `src/agents/execute_agent.py`
- 相关测试

## 2. 新发现：3.1 的 text-only 对“单一目标” surprisingly strong

这一轮最重要的新发现，不是来自理论分析，而是来自真实对比实验。

我们使用官方 `sam3` 仓库 image 路径，在本地 `sam3.1_multiplex.pt` 上做了多轮真实测试，发现：

### 对以下类型目标，`text-only` 效果很好

- `fig2`：白色衬衣
- `fig3`：白色短裙
- `fig4`：左侧石狮子

这些 case 的共同点是：

- 单一物体或单一服饰
- 目标语义清晰
- 目标边界与背景相对可分

### 对以下类型目标，`text-only` 效果一般

- `fig1`：白衣女生整体

这个 case 的特点是：

- 目标不是单纯“一个衣服”或“一个雕像”
- 是人物主体
- 与头发、皮肤、书本、前景等复杂区域耦合
- 文本语义能找对对象，但最终 mask 容易偏粗

### 工程结论

`text-only` 不应被理解成“总是粗糙”的弱方案。

更准确地说：

- 对简单语义对象，`text-only` 可能就是最优路径
- 对复杂人物主体，`text-only` 更适合作为 proposal，而不是 final mask

## 3. 新发现：`text + positive box` 不一定优于 `text-only`

在 `fig1` 上，我们真实比较了：

- `text_only`
- `text + positive person box`
- `text + positive person box + negative boxes`
- `text + face/upper-body positive boxes + negative boxes`

实验结果显示：

- `text + positive box` 的确会让预测更“聚焦”
- 但不稳定
- 对复杂人物图，常常出现：
  - 吞掉大块主体和前景
  - 变成奇怪的楔形区域
  - 比 `text-only` 更人工、更脆弱

### 工程结论

在当前这套 3.1 image 路径里：

- `text-only` 不是 baseline fallback
- 它反而应当是第一优先路径

也就是说，不应该默认：

- 先加几何 prompt
- 再看结果

而应该改成：

- 先试 `text-only`
- 只有结果明显不够好时，再加几何 refinement

## 4. 新发现：3.1 可以做单图 image segmentation，但运行方式有坑

最初误以为 `sam3.1_multiplex.pt` 不能用于 image segmentation。

后续通过真实实验确认：

- **3.1 可以做单图 image segmentation**
- **也支持 text prompt**

但有两个重要坑：

### 1. 运行时要使用 CUDA + `bfloat16 autocast`

如果直接走 image path，容易遇到：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

后来按官方脚本风格，在 CUDA 下使用：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    ...
```

问题消失，image text-prompt 路径能够跑通。

### 2. `sam3.1_multiplex.pt` 仍然会有少量 `missing_keys`

实际加载时仍然会打印：

- `backbone.vision_backbone.convs.3.*` 等少量 missing keys

这说明：

- 当前 checkpoint 和这条 image path 不是“完全无噪声匹配”
- 但已经能真实跑出结果

### 工程结论

后续若继续依赖 3.1 image path：

- 应把 `bfloat16 autocast` 当作默认运行前提
- 应把少量 `missing_keys` 视为“已知但暂不阻断”的告警，而不是立即判为不可用

## 5. 新问题：官方 backend 的 interactive refinement 路径没有打通

本轮原本尝试把 `segment tool` 做成：

- `text-only proposal`
- 若不够好，再自动进入：
  - `box + points`
  - `mask_input refinement`

但在真实 agent 流程里，第二阶段没有打通。

### 当前失败点

当 `text-only` 结果被判定为“不够好”时，工具会尝试回退到几何 refinement 路径。

这一步在当前官方 backend 上会失败：

```text
Official SAM3 image model does not expose inst_interactive_predictor.
```

### 原因

我们当前能稳定跑通的是：

- 官方 image path
- `build_sam3_image_model(..., enable_inst_interactivity=False)`
- `Sam3Processor`
- `set_image`
- `set_text_prompt`

但不能稳定跑通的是：

- 官方 image interactive predictor 路径
- 尤其是依赖 `inst_interactive_predictor` 的做法

### 工程结论

当前阶段，不应在主流程里默认启用第二阶段几何 refinement。

更稳妥的做法是：

- **先将 `segment` 固定为 `text-only` 模式**
- 待 interactive refinement 路径真正打通，再重新引入 fallback

## 6. 新决策：当前 `segment tool` 应先以 `text-only` 为准

综合本轮实验结果，当前最稳妥的产品化选择是：

### 当前默认行为

- `segment` 接收：
  - `image_ref`
  - `prompt`
- 主路径：
  - 官方 `sam3` 3.1 image text-only segmentation
- 输出：
  - `mask artifact`
  - `mask_score`
  - `source_stage = sam_text_only`

### 暂不启用

- 强依赖 `grounding_ref`
- 默认 `box + points` refinement
- 默认 `mask_input` 第二轮 refinement

### 原因

- 当前最稳定
- 对单物体/服饰类目标效果已经足够好
- agent 流程已经能真实产出结果
- 不会被未打通的 interactive backend 拖垮

## 7. 新问题：本地测试与真实 CUDA 路径的行为差异

在实现 `text-only` proposal 之后，本地单元测试最初失败，不是逻辑坏了，而是：

- 单测没有配置本地 `SAM3_CHECKPOINT_PATH`
- tool 仍然尝试走官方 backend
- 于是触发 Hugging Face 下载
- 离线环境报错

### 解决方式

将 `text-only proposal` 是否启用的判定改成：

- 只有当前环境显式设置了 `SAM3_CHECKPOINT_PATH`
- 且该路径存在
- 才尝试官方 3.1 text-only proposal

### 工程结论

后续凡是接真实视觉 backend 的 tool，都应该明确区分：

- 本地/CI 的 schema-level tests
- 真实 CUDA integration tests

不要让普通单测隐式触发大模型或权重加载。

## 8. 对后续迭代最有价值的新思路

### 思路 1：`segment` 应分层，而不是只有一个模式

长期看，`segment` 最合理的结构应该是：

1. `text-only proposal`
2. `proposal quality check`
3. 若失败，再 `interactive refinement`

但当前只稳定实现了第 1 层。

### 思路 2：不同目标类型应该走不同策略

本轮实验已经说明：

- 衣服 / 独立物体：`text-only` 很强
- 人物整体：`text-only` 粗，后续仍需 refinement

所以未来最好不要把所有 `segment` 请求都强行走一条固定路径。

### 思路 3：`segment tool` 的 prompt 应该可由上游 task 直接驱动

现在 agent 中已改为：

- `ExecuteAgent` 调 `SEGMENT` 时，直接传 task instruction 作为 `prompt`

这点是对的，说明：

- 上游 planning / understanding 产出的语言描述，可以直接变成 segmentation query
- 不必总是先走 `grounding`

## 9. 当前状态总结

本轮之后，当前仓库中：

- `segment tool` 已改成 `image + prompt`
- agent 中 `SEGMENT` 已改成直接传 prompt
- 本地相关回归通过
- 真实 agent 流程在以下目标上验证通过：
  - `fig2` 白衬衣
  - `fig3` 白短裙
  - `fig4` 左石狮子
- `fig1` 复杂人物图仍然偏粗，但路径已经可跑

### 当前最稳妥的产品结论

现阶段可以把 `segment(image + prompt)` 当作：

- **语义单物体抠图入口**

但不要把它误认为：

- 已经解决复杂人物整体精细抠图

那一步还需要后续专门增强。
