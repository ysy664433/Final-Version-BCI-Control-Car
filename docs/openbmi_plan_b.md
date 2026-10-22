# OpenBMI 方案 B：数据增强、知识蒸馏与监督式对比学习

## A、B 的边界

方案 B 根据本次给定的三点定义实现：保留 EEGNetLight 与知识蒸馏，增加频带扰动和通道掩膜，在原损失上增加按运动意图监督的对比损失。

| 项目 | 方案 A | 方案 B |
| --- | --- | --- |
| 训练入口 | `python -m src.plan_a.train` | `python -m src.plan_b.train` |
| 方法 | 可学习弱特征门控 + DANN | 随机 EEG 增强 + SupCon |
| DANN 与 GRL | 有 | 无 |
| 可学习前端弱特征门控 | 有 | 无 |
| 对比投影头 | 无 | 有，仅训练使用 |
| 默认模型目录 | `models/openbmi_plan_a/` | `models/openbmi_plan_b/` |
| 默认结果目录 | `outputs/openbmi_plan_a/` | `outputs/openbmi_plan_b/` |

两者共享现有 EEGNetLight 底座定义、教师模型、数据加载、CSP 拟合、划分和通用统计函数，训练循环与模型构造分别实现。B 不调用 A 的训练循环，不把 DANN 参数加入 B，也不把方案 A 的权重作为 B 的初始化。B 的 checkpoint 标记 `framework=plan_b`，B 推理入口会拒绝 A 的 checkpoint。原 A 文件无需修改。

原方案 A 的前端增强是网络学习的时间门控；这里 B 的随机数据增强是训练策略，两者不是同一个模块。为区分方法，当前不把 A 的门控叠加进 B。后续若要研究 A+B，需要额外建立组合实验，不能把组合结果当作纯 B。

## 数据增强

增强仅发生在训练阶段。每个 trial 生成两个独立视图，保持左右手标签不变：

1. 频带扰动：在 4–8、8–13、13–30、30–45 Hz 四个频带上分别采样增益，默认 `[0.8, 1.2]`，对同一个 trial 的各通道使用相同频带增益，以限制空间结构破坏。
2. 相对噪声：按每通道 RMS 加入零均值 Gaussian 噪声，默认标准差为 RMS 的 0.05 倍。
3. 通道掩膜：默认每通道以 0.2 概率置零，并保证至少保留一个通道。

参数为 `--band-strength 0.2 --channel-mask 0.2 --noise-std 0.05`。三个强度均可设为零关闭相应变换，允许范围为 `[0, 0.5]`。这些操作模拟部分幅度变化、噪声和通道退化，不代表完整复现低表现受试者的神经机制，也不能保证任何增强都保留足够的任务信息。强度应仅依据验证集选择。

增强在还原训练均值、尺度后的原始数值空间执行，再使用原有训练统计量标准化。每个增强视图均重新通过已拟合的 CSP 和 CSP 标准化器，保持 EEG 与 CSP 分支输入一致；不会将增强 EEG 与未增强的 CSP 特征拼接。CSP 不在增强批次、验证集或测试集上重新拟合。

频域增强可使用训练窗口内的完整数据，但部署时完全关闭，因此不会给在线滤波加入未来样本依赖。在线底座仍在 cue 后完整 4 秒窗口收齐后推理。

## 监督式对比学习

对 192 维融合特征增加 `192 → 192 → projection_dim` 投影头，默认 `projection_dim=128`。投影输出进行 L2 归一化，再计算 supervised contrastive loss：

- 同一 trial 的两个视图为正样本。
- 同一运动意图的其他 trial 及其视图也为正样本，不区分正常或低表现组。
- 不同运动意图的 trial 为负样本。
- 排除样本自身，使用稳定的 log-sum-exp 计算。批次只有一种意图、没有负样本时，返回可微零损失并记录次数。

监督标签是 `left/right`，不是受试者编号或 BCI 分组。低表现组的标签与正常组一起参与训练，沿用有监督跨会话协议。

```text
L = (1 - alpha) * (L_CE + variance_weight * L_var)
    + alpha * L_KD
    + contrastive_weight * L_SupCon
```

`L_CE`、`L_KD` 和概率方差正则分别在两个视图上计算后取平均。默认 `alpha=0.5`、蒸馏温度 `T=3`、`contrastive_weight=0.1`、对比温度 `0.1`。教师与学生同步训练，教师也接收相同的增强视图；教师输出在蒸馏项中停止梯度。推理只运行任务分类分支，不需要投影、对比配对、增强或教师。

## 对照与训练

| B 中的变体 | 随机 EEG 增强 | 对比学习 | 每 trial 的训练视图数 |
| --- | --- | --- | --- |
| `baseline` | 无 | 无 | 1 |
| `augment` | 有 | 无 | 2 |
| `contrast` | 无 | 有 | 2，同一输入通过随机 dropout 形成特征差异 |
| `plan_b` | 有 | 有 | 2 |

所有变体保留知识蒸馏。`augment` 与 `plan_b` 使用相同视图数与增强流程，适合判断增加对比损失是否有额外作用。A、B 的 `baseline` 使用相同初始化、采样及单视图训练流程；本地合成数据验证了其单轮训练后的预测完全一致。

安装依赖与数据布局同 [方案 A](openbmi_plan_a.md)。可使用 `python -m pip install -r src/plan_b/requirements.txt`，服务器上需另确认 PyTorch 支持 GPU。下面均从项目根目录执行。当前只准备代码和小规模功能验证，尚未运行大批量训练。

数据准备检查不训练神经网络：

```bash
python -m src.plan_b.train --subjects 1 5 --modes offline online --prepare-only --output-root outputs/openbmi_plan_b_prepare
```

沿用现有名单的探索性实验：

```bash
python -m src.plan_b.train --device cuda --amp --workers 4 --modes offline online --variants baseline augment contrast plan_b --seeds 42 43 44 --domain-source supplied --output-root models/openbmi_plan_b_supplied
```

正式实验建议仅用训练会话判定低表现组。为方便 A/B 比较，下面各命令明确采用相同受试者全集、随机种子、划分、批次大小和基础训练参数：

```bash
python -m src.plan_a.train --device cuda --amp --workers 4 --modes offline online --variants baseline plan_a --seeds 42 43 44 --domain-source training --batch-size 8 --cache-root outputs/openbmi_cache --output-root models/openbmi_plan_a_training

python -m src.plan_b.train --device cuda --amp --workers 4 --modes offline online --variants baseline augment contrast plan_b --seeds 42 43 44 --domain-source training --batch-size 8 --cache-root outputs/openbmi_cache --output-root models/openbmi_plan_b_training
```

两个命令独立运行、独立保存 checkpoint；可以顺序运行，不是联合训练。共享缓存只缓存相同预处理的 trial，不包含模型参数。不要同时向同一个缓存根目录首次生成相同条目；并行作业可以分别指定缓存目录。

GPU 显存不足时，两个方案的 `--batch-size` 一并减小。B 的两个视图保留两份计算图，训练计算量和显存开销更大。当前比较以相同 epoch、原始 trial 数和优化器更新次数为准，并非相同总 FLOPs；报告会保留这个限制。可另做计算预算匹配实验。

续训使用原命令追加 `--resume`。配置、核心代码和训练数据检查及轮次恢复规则同 A。产物布局为：

```text
models/openbmi_plan_b_training/online/plan_b/seed_42/
  config.json
  preprocessing.joblib
  best.pt
  last.pt
  history.json
  validation_subjects.json
```

`history.json` 包含 CE、KD、SupCon、目标组验证 balanced accuracy、视图数和缺少负样本的批次数。`best.pt` 仍仅由 session1 验证集目标组指标选择。

## 评估与 A/B 比较

单独分析 B 的增强与对比学习消融：

```bash
python -m src.plan_b.evaluate --model-root models/openbmi_plan_b_training --device cuda --output-root outputs/openbmi_plan_b_training
```

除各变体与 baseline 的比较外，当相应消融变体存在时，报告还直接比较 `augment → plan_b` 和 `contrast → plan_b`，以分析对比损失和增强各自的额外作用；多重检验校正覆盖整份 B 报告。

最终同时比较 A 与 B：

```bash
python -m src.tools.compare_plan_ab --plan-a-root models/openbmi_plan_a_training --plan-b-root models/openbmi_plan_b_training --device cuda --output-root outputs/openbmi_plan_ab_training
```

跨方案比较入口在读取测试数据前核查受试者、训练数据指纹、测试协议、通道、域来源、种子、底座维度和基础超参数。缺少某方案的 baseline、种子不一致或协议不一致会报错，不会静默将不同实验拼在一起。

输出包含各自 baseline → A、baseline → B、A → B，以及 A/B baseline 一致性检查，分别报告全部受试者、原名单低表现组和训练来源低表现组。提升始终表示 `candidate - reference`。同一受试者的种子先取均值，再做受试者 bootstrap CI、双侧 Wilcoxon；Holm 校正覆盖整份 A/B 报告的所有比较。

逐受试者 CSV、试次预测、完整运行配置及统计 JSON 分别保存，A/B 的试次文件位于不同子目录。读取最终 session2 测试成绩后不要据此调整增强强度或对比权重；调参使用 session1 验证集。

原名单使用过两次会话，因此 `--domain-source supplied` 的任何结果仍是探索性结果。只有整套实验从训练开始使用 `--domain-source training`，才能避免该名单造成的测试信息使用。训练来源的分组可能与原名单不同，不能把两套名单混用来挑选提升更大的结果。显著性与效果大小需等待真实训练和独立测试。

## 独立在线接口

```python
from src.plan_b.inference import OpenBMIStreamProcessor

processor = OpenBMIStreamProcessor(
    "models/openbmi_plan_b_training/online/plan_b/seed_42",
    input_channels=channel_names,
    sample_rate=1000,
)
processor.schedule_trial(cue_sample)
results = processor.process_chunk(raw_chunk)
```

通道名、单位、零基 cue 索引、连续滤波状态及新记录重置方式同 A 的独立接口。这里不启用训练增强，不连接原四分类 GUI，也不修改小车控制映射。
