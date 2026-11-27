# BACE presentation script and Chinese correspondence

The prepared script has 167 spoken words. With an eight-second pause allowance, its estimated duration is 1:59 at 90 words/minute or 2:13 at 80. The pronunciation convention treats BACE, COVE and LoRA as words and LLM as three spoken letters.

讲稿共167个口播词。包含八秒停顿时，90词/分约1分59秒，80词/分约2分13秒。BACE、COVE和LoRA按单词发音，LLM按三个字母发音。

## Complete English speaking script

### Slide 1

My method is BACE, or Budget-Aware Canonical Exclusion. A coverage-trained LoRA model proposes hypotheses from partial observations. Repeated structures waste attempts. With sixteen attempts, BACE targets thirteen distinct structural classes. It samples normally while slack remains, then masks seen classes when another repeat would make the target unreachable. The LLM generates, and every answer counts.

### Slide 2

I compared the same checkpoint with and without BACE on twenty-six Boolean tasks using three sampling seeds. Recovery rose from forty-one point zero two percent to fifty point three four percent. Recovery gained nine point three two points, with a bootstrap interval of about six to thirteen points. Validity fell by three point five three points. The LLM also outperformed uniform sampling. This supports decoding, while the training-loss benefit remains unproven.

### Slide 3

COVE improves final system outputs using a learned executor and a fixed hypothesis catalog. The executor-only control matches its scores, so these results do not demonstrate better LLM generation.

## Sentence aligned script translation

### Slide 1

**1.1 English**

My method is BACE, or Budget-Aware Canonical Exclusion.

**中文**

我的方法是BACE，即预算感知的规范结构排除。

**1.2 English**

A coverage-trained LoRA model proposes hypotheses from partial observations.

**中文**

一个经过覆盖目标训练的LoRA模型根据部分观测提出假设。

**1.3 English**

Repeated structures waste attempts.

**中文**

重复结构会浪费尝试次数。

**1.4 English**

With sixteen attempts, BACE targets thirteen distinct structural classes.

**中文**

在十六次尝试中，BACE的目标是得到十三个不同的结构类别。

**1.5 English**

It samples normally while slack remains, then masks seen classes when another repeat would make the target unreachable.

**中文**

还有余量时正常采样；当再重复一次就会使目标无法达到时，屏蔽已经见过的类别。

**1.6 English**

The LLM generates, and every answer counts.

**中文**

生成仍由LLM完成，每个回答都计入预算。

### Slide 2

**2.1 English**

I compared the same checkpoint with and without BACE on twenty-six Boolean tasks using three sampling seeds.

**中文**

我在二十六道布尔任务上，使用三个采样种子，对比同一个模型检查点使用和不使用BACE的表现。

**2.2 English**

Recovery rose from forty-one point zero two percent to fifty point three four percent.

**中文**

恢复率从百分之四十一点零二上升到百分之五十点三四。

**2.3 English**

Recovery gained nine point three two points, with a bootstrap interval of about six to thirteen points.

**中文**

恢复率提升了九点三二个百分点，bootstrap区间约为六到十三个百分点。

**2.4 English**

Validity fell by three point five three points.

**中文**

合法率下降了三点五三个百分点。

**2.5 English**

The LLM also outperformed uniform sampling.

**中文**

LLM也优于均匀采样对照。

**2.6 English**

This supports decoding, while the training-loss benefit remains unproven.

**中文**

这支持解码模块的收益，而训练损失的额外收益仍未得到证实。

### Slide 3

**3.1 English**

COVE improves final system outputs using a learned executor and a fixed hypothesis catalog.

**中文**

COVE使用学习执行器和固定假设目录改善最终系统输出。

**3.2 English**

The executor-only control matches its scores, so these results do not demonstrate better LLM generation.

**中文**

仅使用执行器的对照也达到相同分数，因此这些结果不能证明LLM的生成能力有所提高。

## Slide 1 text translations

| English | 中文逐项对应 |
|---|---|
| BACE for hypothesis coverage | 用于假设覆盖的BACE |
| Budget-Aware Canonical Exclusion with a coverage-trained LoRA model | 将预算感知的规范类排除与覆盖目标训练的LoRA模型结合 |
| Problem<br>Repeated structures waste a fixed generation budget. | 问题<br>重复结构会浪费固定的生成预算。 |
| Learned proposal<br>Qwen2.5-1.5B with coverage LoRA<br>Loss: SFT minus coverage proxy | 学习得到的提议分布<br>Qwen2.5-1.5B与覆盖LoRA<br>损失为SFT损失减去覆盖代理。 |
| Generation target<br>13 distinct structural classes in 16 attempts | 生成目标<br>十六次尝试中得到十三个不同结构类别 |
| Generation process | 生成过程 |
| Sample normally and track canonical classes | 正常采样，并记录规范结构类别。 |
| Compute slack: attempts left minus new classes needed | 计算余量：剩余尝试次数减去仍需增加的新类别数。 |
| At zero slack, mask previously seen classes | 余量为零时，屏蔽已经见过的类别。 |
| Let the LLM generate and count every attempt | 由LLM生成，并计入每一次尝试。 |
| Finite grammar during intervention. No observation-validity filtering or free retries. | 干预期间使用有限语法；不根据观测合法性筛选候选，也没有免费重试。 |

## Slide 2 text translations

| English | 中文逐项对应 |
|---|---|
| BACE results and decoder contribution | BACE结果与解码模块的贡献 |
| Same trained checkpoint, 26 Boolean tasks, 3 inference seeds, 16 attempts per task | 使用同一训练检查点，在二十六道布尔任务上采用三个推理种子，每题十六次尝试。 |
| Recovery gain: +9.32 percentage points | 恢复率增加九点三二个百分点。 |
| 95% task-bootstrap interval: [5.90, 12.82] points | 按题bootstrap的95%区间为五点九零到十二点八二个百分点。 |
| Validity decreases by 3.53 points in the matched comparison | 在匹配对照中，合法率下降三点五三个百分点。 |
| LLM + BACE exceeds uniform + BACE by 10.31 recovery points | LLM与BACE组合的恢复率比均匀采样与BACE组合高十点三一个百分点。 |
| One training run. Exploratory follow-up on previously used Boolean tasks. Extra coverage-loss benefit over SFT remains unproven. | 只有一次训练运行；这是对之前使用过的布尔题的探索性跟进。覆盖训练损失相对SFT的额外收益仍未得到证实。 |

### Result table translation

| Method 方法 | Validity 合法率 | Uniqueness 不重复率 | Recovery 恢复率 |
|---|---:|---:|---:|
| Same LLM, ordinary sampling<br>同一LLM正常采样 | 39.90% | 64.42% | 41.02% |
| Same LLM + BACE<br>同一LLM加BACE | 36.38% | 81.49% | 50.34% |
| Uniform + BACE, no LLM<br>无LLM的均匀采样加BACE | 28.93% | 83.89% | 40.03% |

## Slide 3 text translations

| English | 中文逐项对应 |
|---|---|
| COVE as an alternative system approach | 作为另一种系统思路的COVE |
| COVE improves final system outputs using a learned executor and a fixed hypothesis catalog. | COVE使用学习执行器和固定假设目录改善最终系统输出。 |
| The executor-only control matches its scores, so these results do not demonstrate better LLM generation. | 仅使用执行器的对照也达到相同分数，因此这些结果不能证明LLM的生成能力有所提高。 |

## Understanding the claims

BACE限制重复结构，但不提前筛出满足观测的测试答案。余量是剩余尝试次数减去仍需增加的新类别数。唯一率保证来自约束，恢复率和合法率仍需要实测。

结果页对比同一检查点的两种解码方式，因此支持解码模块的贡献。它没有证明覆盖训练损失优于普通SFT。三个种子是推理重复，测试观测题已在旧实验中出现。

第三页仅口播两句话。COVE使用学习执行器和固定目录改善最终系统输出，但仅执行器对照也同分，不能据此宣称LLM生成能力提高。其接近99%的数字不与BACE表混用。

## Sources

- [BACE method and results](../bace/README.md)
- [BACE raw result summary](../bace/results/canonical_preference_v5/test/summary.json)
- [COVE raw result summary](../cove/results/all80/test/summary.json)
- https://github.com/Xiao-ada/EEC4300-HypoSpace
- https://proceedings.mlr.press/v306/chen26bz.html
