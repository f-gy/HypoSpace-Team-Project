# 官方 HypoSpace 评测报告

**状态：** 已完成  
**入口：** 未改动的 `Benchmark3D.evaluate_single_observation_set`  
**指标（官方字段名）：**

| 老师/README 名称 | 官方代码字段 | 公式 |
|---|---|---|
| Validity | `valid_rate` | n_valid / n_queries |
| Uniqueness | `novelty_rate` | n_unique / n_queries |
| Recovery | `recovery_rate` | n_recovered_GT / n_GT |

**产物：** `results/official_hypospace/`  
**脚本：** `scripts/run_official_hypospace_eval.py`

---

## 设置

- 同一批 50 obs；种子 7/17/27；`n_queries=3`；`max_retries=1`（调用次数=3）
- 模型：本地 Qwen2.5-7B-Instruct，temperature=0.3
- 方法：
  1. **official_prompt**：直接答官方 Layer prompt  
  2. **cpsg_supportproj**：CPSG 高度生成 + SupportProj → 转成 `Structure:` 文本，再交给**官方 parser + validator + 三项指标**
- **不含 BAE**（程序扩展不是官方 LLM query 循环里的输出）

---

## 三种子均值（官方口径，均 ≤100%）

| Method | Validity | Uniqueness | Recovery |
|---|---|---|---|
| official_prompt | **0.7%** | 95.8% | **0.07%** |
| cpsg_supportproj | **80.7%** | 35.6% | **8.5%** |

按种子：

| seed | official V/U/R | CPSG+Proj V/U/R |
|---|---|---|
| 7 | 0.7% / 96.7% / 0.07% | 80.0% / 34.7% / 8.1% |
| 17 | 0.7% / 96.0% / 0.07% | 81.3% / 37.3% / 9.2% |
| 27 | 0.7% / 94.7% / 0.07% | 80.7% / 34.7% / 8.1% |

说明：
- 官方 prompt 下 Qwen **几乎总能解析，但几乎全不合法** → Uniqueness 虚高（大量不同的错结构）。
- CPSG+SupportProj 的 V/U/R 与先前自建脚本同量级，且 **U 不会超过 100%**。
- 相对官方 prompt：Validity 约 **0.7% → 80.7%**；Recovery 约 **0.07% → 8.5%**。

---

## 和之前「184%」的关系

| 口径 | Uniqueness |
|---|---|
| **本次官方** `novelty_rate` | ≈ **36%**（≤100%） |
| 先前 BAE 的 `U_expanded` | 可 >100%（分子含程序扩展） |

**答辩请用本次官方三项。** BAE 的覆盖增益单独作为「有界程序扩展」附录，不要和官方 Uniqueness 混成一个百分数。

---

## 后续：RangeRetry 冲 V≥90%（官方同入口）

见 **`docs/official_hypospace_v90_report.md`** / `results/official_hypospace_v90/`。

| Method | Validity | Uniqueness | Recovery |
|---|---|---|---|
| cpsg_supportproj（上表） | 80.7% | 35.6% | 8.5% |
| **cpsg_supportproj_retry** | **93.6%** | **76.9%** | **19.7%** |

V 已过 README 90% 线；U/R 均未回撤（反而上升）。R 受 `n_queries=3` 理论上限约束，无法到 80%。

---

## 诚信声明

指标由官方 `evaluate_single_observation_set` 计算；未修改 Validator/GT；CPSG 仅替换 LLM 回复文本生成方式。
