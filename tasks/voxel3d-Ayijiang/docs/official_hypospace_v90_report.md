# 官方评测：CPSG+SupportProj+RangeRetry（冲 V≥90%）

**状态：** 已完成  
**入口：** 未改动的 `Benchmark3D.evaluate_single_observation_set`  
**产物：** `results/official_hypospace_v90/`  
**脚本：** `scripts/run_official_hypospace_eval.py --methods cpsg_supportproj_retry`

---

## 方法（相对上一版 cpsg_supportproj）

1. **SupportProj**：空列强制 h=0（程序投影，不改占据列）。  
2. **Range 一次重试**：SupportProj 后若仍有 range / parse 失败，用错误反馈再让模型生成一次；**禁止静默填高**。  
3. **Anti-dup**：若与本 obs 已产出合法 hash 重复，再尝试一次不同高度图；失败则保留第一次合法结果（不牺牲 V）。

官方 `n_queries=3`、`max_retries=1` 不变；额外 LLM 调用发生在**单个 query 槽内部**，并计入 token。

---

## 官方三项（三种子均值）

| Method | Validity | Uniqueness | Recovery |
|---|---|---|---|
| official_prompt（先前） | 0.7% | 95.8% | 0.07% |
| cpsg_supportproj（先前） | 80.7% | 35.6% | 8.5% |
| **cpsg_supportproj_retry（本轮）** | **93.6%** | **76.9%** | **19.7%** |

### 分种子

| seed | V | U | R |
|---|---|---|---|
| 7 | 92.7% | 74.0% | 19.6% |
| 17 | 94.0% | 78.0% | 19.5% |
| 27 | 94.0% | 78.7% | 19.9% |

### 相对 cpsg_supportproj 的 Δ

| 指标 | Δ |
|---|---|
| Validity | **+12.9 pp**（80.7% → 93.6%） |
| Uniqueness | **+41.3 pp**（35.6% → 76.9%） |
| Recovery | **+11.2 pp**（8.5% → 19.7%） |

---

## 门控（相对先前官方 cpsg_supportproj）

| Gate | 结果 |
|---|---|
| Validity ≥ 90% | **通过**（93.6%） |
| Uniqueness 不跌（允许 −2 pp） | **通过**（升至 76.9%） |
| Recovery 不跌（允许 −1 pp） | **通过**（升至 19.7%） |

README「非常优秀」线（90 / 80 / 80）：
- **V：达到**  
- **U：接近**（76.9%，差约 3 pp）  
- **R：未达**（且在 `n_queries=3`、本池 mean \|GT\|≈19.6 时，理论均值上限约 **26%**，不可能到 80%）

---

## Adapter 统计（三种子合计量级）

每种子约 150 个官方 query 槽：

| 事件 | seed7 | seed17 | seed27 |
|---|---|---|---|
| first_ok | 114 | 114 | 116 |
| range_retry_ok | 25 | 27 | 25 |
| range_retry_fail | 11 | 9 | 9 |
| antidup_ok | 4 | 9 | 9 |
| antidup_keep_first | 28 | 23 | 24 |

Range 重试约救回 **25–27 / 150** 次调用，是 V 过 90% 的主杠杆；anti-dup 对 U 有贡献，失败时保留首个合法输出故不伤 V。

---

## 诚信声明

- 指标仅由官方 `valid_rate` / `novelty_rate` / `recovery_rate` 计算。  
- 未修改 Validator / GT。  
- 未静默 clamp / 填高；未把 BAE 程序扩展计入官方三项。  
- Token 高于 cpsg_supportproj（因内部重试），须在报告中披露。
