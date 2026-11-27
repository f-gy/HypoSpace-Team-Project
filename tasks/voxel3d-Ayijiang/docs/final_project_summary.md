# Final Project 总结（答辩用）

**评测口径：** 老师 HypoSpace 官方平台 `Benchmark3D.evaluate_single_observation_set`  
**任务域：** 3D Voxel Reconstruction（`3d_grid3_h3_b3`）  
**模型：** Qwen2.5-7B-Instruct（本地）

---

## 1. 任务要求

基于论文 / GitHub [HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace)：把 LLM 当作「有限假设集合的采样器」，在观测不足（underdetermined）时评价其提出**多样且合法**假设的能力。

官方三项指标（README「非常优秀」线）：

| 指标 | 含义 | 优秀线 |
|---|---|---|
| **Validity (V)** | 提出的假设有多少与观测一致（合法率） | > 90% |
| **Uniqueness (U)** | 提出的假设有多少彼此不重复 | > 80% |
| **Recovery (R)** | 覆盖了多少枚举出的合法假设空间 | > 80% |

本项目聚焦 **3D 体素重建**：给定顶视图 Observation，在重力约束下生成可能的 3D 结构；用官方 validator / hash / GT recovery 计分，**不改官方代码与 GT**。

---

## 2. 痛点是什么

1. **合法生成极难：** 官方 Layer 自由文本 prompt 下，7B 模型几乎总能解析，但结构常与顶视图不一致或违反重力 → 官方 **Validity ≈ 0.7%**，Recovery ≈ 0。  
2. **正确性指标看不见的模式崩塌：** Baseline 的 Uniqueness 虚高（≈96%），因为大量**彼此不同的错误结构**；看起来「多样」，实际没有科学价值。  
3. **假设空间大、查询预算小：** 每题仅 `n_queries=3`，而本池平均 \|GT\|≈19.6；即便 3 次全是不重复的合法 GT，Recovery 均值理论上限约 **26%**，README 的 R>80% 在此预算下不可达。  
4. **早期轨道扩展（ITOE）无效：** Validity=0 时没有合法种子可扩；Validity 抬起来后，多数 Observation 的平面自同构仍太弱，Recovery 几乎不动。

核心矛盾：**先要能稳定产出合法假设，再谈多样性与覆盖；自由 Layer 生成卡在第一步。**

---

## 3. 创新点（讲清楚「新在哪」）

### 总表述（推荐答辩一句）

**Constraint-Preserving Structured Generation（CPSG）流水线：**  
把 HypoSpace 3D 的「自由生成多层 0/1 网格」改成「只预测 Observation 占据列的柱体高度 → 程序装配为官方 Structure3D」，并叠加轻量程序净化与一次约束重试；在**官方评测入口**上把 Validity 从近 0 提升到 >90%，同时抬升 Uniqueness（合法意义下）与 Recovery。

### 分别说明「算不算新算法」

| 组件 | 性质 | 怎么说才诚实 |
|---|---|---|
| **CPSG（柱高表示 + 程序还原）** | **任务定制的结构化生成 / 表示变换**，不是全新通用 LLM 算法 | 官方数据集生成器本身用柱高枚举 H；我们**首次把该表示从「数据集构造」显式接到「LLM 解码接口」**，并证明与官方合法空间**集合等价**（审计 GO），从而在不缩 H 的前提下大幅降约束违反。 |
| **SupportProj（空列置 0）** | 程序投影，工程组件 | **不是**新学习算法；是把模型在空列上的非法质量投影回 Observation 支撑。答辩须标明「程序侧」。 |
| **Range 一次重试** | 约束反馈 / 自修正，属已有范式在本任务上的落地 | **不是**首创「反馈重试」思想；是在 CPSG 输出空间上对 range 失败做**一次、禁止静默填高**的重试，专门冲官方 V≥90%。 |
| **Anti-dup** | 简单去重再采样 | 工程手段，保护官方 Uniqueness；失败则保留首个合法结果，避免伤 V。 |
| **ITOE / BAE** | 探索过的旁支 | ITOE 官方口径下 **R 几乎无增益**；BAE 是有界程序搜索，**不得计入官方三项**，仅可作附录。 |

### 创新定位（一句话归类）

- **不是：** 提出全新的大规模预训练算法或通用解码理论。  
- **是：** 针对 HypoSpace 3D 的 **constraint-preserving 结构化假设生成方法**（表示对齐 + 程序约束保持 + 轻量重试），并用老师官方平台给出可复现增益。  
- **等价性审计**是方法成立的前提：证明柱高空间 = 官方合法 H，避免「缩小假设空间刷 Validity」的质疑。

---

## 4. Baseline 与本方法数据对比（官方平台）

设定：同一 50 个 Observation；种子 7/17/27；`n_queries=3`；Qwen2.5-7B-Instruct。

| 方法 | Validity | Uniqueness | Recovery | 说明 |
|---|---|---|---|---|
| **Baseline：官方 Layer prompt** | **0.7%** | 95.8% | **0.07%** | U 虚高（错结构多样） |
| 中间版：CPSG + SupportProj | 80.7% | 35.6% | 8.5% | 合法率大幅上升 |
| **本方法：CPSG + SupportProj + RangeRetry + AntiDup** | **93.6%** | **76.9%** | **19.7%** | 主结果 |

相对 Baseline：

| 指标 | 变化 |
|---|---|
| Validity | 0.7% → **93.6%**（过 README 90% 线） |
| Uniqueness | 95.8% → **76.9%**（从「错误多样性」变为「合法近不重复」；接近但未过 80%） |
| Recovery | 0.07% → **19.7%**（约 ×280；仍远低于 80%，受 3-query 上限约束） |

相对中间版 cpsg_supportproj：V +12.9 pp，U +41.3 pp，R +11.2 pp（RangeRetry / AntiDup 贡献）。

数据出处：`results/official_hypospace/`、`results/official_hypospace_v90/`。

---

## 5. 本方法优缺点

### 优点

1. **对准痛点：** 直接解决「生成不合法」这一 HypoSpace 3D 上的第一瓶颈。  
2. **与官方空间对齐：** 柱高表示经审计与官方 GT **等价**，增益不能简单归因于缩 H。  
3. **官方口径可复现：** 分数来自未改动的 `valid_rate / novelty_rate / recovery_rate`。  
4. **V 达优秀线，U/R 同向改善：** 不是牺牲覆盖去刷单一指标。  
5. **可解释、可拆开报：** 模型预测高度 vs 程序投影/重试分工清晰，便于诚信答辩。  
6. **输出更短：** 高度图比多层 Layer 文本短，同 `max_new_tokens` 下实际 completion 更省（重试版因内部再生成 token 会升高，需披露）。

### 缺点 / 局限

1. **新颖性偏「任务方法」而非「新算法」：** 结构化生成、投影、反馈重试均有先例；贡献在组合与 HypoSpace 3D 落地。  
2. **未达 README 全线 90/80/80：** U≈77% 接近；R≈20% 受 `n_queries=3` 理论束缚，无法用当前预算冲到 80%。  
3. **仍依赖程序侧：** SupportProj / retry / anti-dup 不是纯端到端模型能力；须避免把程序收益说成「模型变聪明」。  
4. **额外 token：** RangeRetry / AntiDup 在单 query 槽内增加调用，与「同 n_queries」公平，但与「同 token」不完全公平。  
5. **Recovery 仍是短板：** ITOE 未奏效；更强覆盖需要更大查询预算或另立有界扩展（且不能混进官方 LLM 三项）。  
6. **领域专用：** 柱高表示绑定重力柱状结构，不能直接包装成全领域通用 ITOE。

---

## 6. 答辩时建议怎么说（30 秒版）

老师要求用 HypoSpace 官方三项评「合法、不重复、覆盖」。我们在 3D 任务上发现 7B 自由生成 Validity 几乎为 0。于是提出 CPSG：让模型只预测占据列高度，程序还原 Structure3D（表示与官方 H 等价），再加 SupportProj 与一次禁止填高的 range 重试。官方平台上 Validity 从 0.7% 到 93.6%，Uniqueness 与 Recovery 同步提升；创新点是**面向该基准的约束保持结构化生成流水线**，不是新的通用大模型算法，也主动说明 Recovery 受查询预算上限限制、未宣称达到 80% 优秀线。
