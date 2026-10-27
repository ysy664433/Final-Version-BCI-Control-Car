
# SDAD-Net 实验索引

本文件回答：做过哪些实验、各自解决什么问题、结果在哪里，以及哪些结果可以用于项目展示。

## 1. 统一实验协议

- 数据：OpenBMI，54名受试者，左右手运动想象二分类。
- 训练：session1 的 EEG_MI_train。
- 验证：session1 的 EEG_MI_test。
- 测试：session2 的 EEG_MI_test。
- 固定分组：根据训练数据确定25名低表现者、29名高表现者，后续不重新划分。
- 达标标准：准确率 ≥ 70%。
- 主要指标：低表现组平均准确率、达标人数，以及全体表现。
- 检查点选择：依据验证集指标，测试时使用固定检查点。

低表现组属于实验分组，不代表临床诊断。验证集结果与测试集结果分别记录。

## 2. 正式主结果

以下目录均位于原服务器 `/mne_data/mne_data/bci_runs/` 下。

| 编号 | 实验 | 训练入口 | 测试入口 | 测试结果目录 |
|---|---|---|---|---|
| M0 | 联合 CSP+LDA 传统参考 | `compare_pooled_csp_lda.py` | 同一脚本 | `pooled_csp_lda_test_001/` |
| M1 | 预训练42／后期42 | `repeat_group_domain42.py` | `evaluate_group_domain42.py` | `group_domain_repeat42_001_test/` |
| M2 | 预训练43／后期43 | `late_subject_domain.py` | `evaluate_subject_domain.py` | `late_subject_domain_43_001_test/` |
| M3 | 预训练43／后期44 | `repeat_group_domain44.py` | `evaluate_group_domain44.py` | `group_domain_source43_seed44_001_test/` |

每次后期配对实验包含：

- `control`：继续进行蒸馏训练，不加入域对抗。
- `group`：保持蒸馏，加入后期高低组 DANN。

M2 还包含 `subject` 分支，用于受试者身份对抗消融。

### session2 测试结果

| 运行 | 方法 | 全体均值 | 全体达标 | 低表现组均值 | 低表现组达标 |
|---|---|---:|---:|---:|---:|
| 固定参考 | CSP+LDA | 57.11% | 7/54 | 53.92% | 1/25 |
| 42／42 | 无对抗对照 | 73.56% | 33/54 | 71.60% | 13/25 |
| 42／42 | SDAD-Net | 73.02% | 34/54 | 71.20% | 14/25 |
| 43／43 | 无对抗对照 | 74.17% | 35/54 | 71.76% | 13/25 |
| 43／43 | SDAD-Net | 75.35% | 38/54 | 73.48% | 17/25 |
| 43／44 | 无对抗对照 | 75.44% | 37/54 | 73.72% | 15/25 |
| 43／44 | SDAD-Net | 75.15% | 36/54 | 73.64% | 14/25 |
| 三次平均 | 无对抗对照 | 74.39% | 35.00/54 | 72.36% | 13.67/25 |
| 三次平均 | SDAD-Net | 74.51% | 36.00/54 | 72.77% | 15.00/25 |

解释边界：

- 相对于 CSP+LDA 的提升属于完整方法的收益，不能全部归因于 DANN。
- 相对于配对蒸馏对照，DANN 的收益较小，且不同运行之间存在波动。
- 三次运行只有两个预训练起点；43／44不是独立的seed44预训练。
- 平均人数是运行结果的算术平均，不是某一次实际人数或集成结果。
- session2 已被多次查看，后续结果属于探索性分析。

## 3. 消融与历史探索

| 类别 | 实验入口／结果目录 | 回答的问题 |
|---|---|---|
| 初步后期DANN | `warm_dann_pair.py`、`repeat_warm_dann_seed43.py` | 先学分类、再加DANN是否更合适？ |
| 后期类条件DANN | `late_classwise_dann.py`、`evaluate_late_classwise.py` | 左右手类别内对齐是否更有效？ |
| 受试者身份对抗 | `late_subject_domain.py` 的 `subject` 分支 | 对抗受试者身份是否优于高低组对抗？ |
| 从头训练DANN | `repeat_dann_pair.py`、`dann_domain003_001/` | 从训练开始加入DANN是否影响学习？ |
| 原方案A | `formal_a_001/`、`plan_a_domain003_001/` | 增强与域对抗的组合效果如何？ |
| 原方案B | `formal_b_seed42_001/`、`shared_baseline_b_001/` | 原对比学习方案是否有效？ |
| 去除遮挡增强 | `plan_b_no_mask_001/` | 遮挡增强是否可能损害任务信息？ |
| 类条件对齐 | `train_classwise_original.py`、`train_alignment.py` | 改变对齐方式是否减少负迁移？ |
| 受试者加权 | `train_subject_weighted.py` | 增加困难受试者权重是否有效？ |
| 跨受试者对比 | `train_cross_subject_contrast.py` | 跨个体同类特征约束是否有效？ |
| 批量候选筛选 | `parallel_experiments_v1/` | 比较采样、蒸馏、增强与辅助约束 |
| 小样本流程检查 | `quick_screen.py`、`smoke_a_001/`、`smoke_b_001/` | 流程是否可运行，快速对照是否充分学习？ |

注意：

- `alignment_pair_001/` 曾手动停止，不能记为全部分支完成。
- 小规模筛选中对照未充分学习，不能据此认定候选模块无效。
- 历史实验不能与正式三次配对结果混合计算平均值。
- 本仓库早期 `baseline` 通常已经包含蒸馏，不等同于团队第一代无蒸馏模型。

## 4. 诊断工具

| 脚本 | 分析内容 |
|---|---|
| `diagnose_seed42.py` | 各模型逐受试者表现 |
| `probe_augmentation.py` | 不同增强下的预测敏感性 |
| `probe_gradients.py` | 辅助损失的梯度大小与方向 |
| `inspect_training_steps.py` | 实际更新次数、AMP跳步与数值状态 |
| `audit_representation.py` | 冻结特征的分类信息与泛化差距 |
| `audit_branches.py` | 时序分支和CSP分支的信息贡献 |
| `probe_centering.py` | 中心化处理的影响 |
| `probe_running_center.py` | 运行式中心化的影响 |
| `profile_session1_eeg.py` | 受试者EEG统计差异 |
| `compare_thresholds.py` | 70%阈值跨越及净变化 |

诊断结果用于提出和排查假设，不能单独作为因果证明或算法有效性的最终证据。

## 5. 归档要求

每项保留实验应对应：

1. 代码版本与启动命令。
2. 配置、数据划分和固定分组。
3. 初始化检查点及随机种子。
4. 训练历史与验证集逐人结果。
5. 固定检查点的测试结果。
6. 结论与局限。

资源位置见 [artifacts.md](artifacts.md)，模块说明见 [README.md](README.md)。
