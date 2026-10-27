# SDAD-Net 资源与结果清单

本文件回答：数据、模型和实验结果在哪里，以及复现、继续训练和成果展示分别需要什么。

## 1. 资源分开管理

| 类型 | 建议位置 | 内容 |
|---|---|---|
| 代码 | GitHub仓库 | 训练、评估、推理及诊断脚本 |
| 文档 | `docs/sdad/` | 方法说明、实验索引和资源清单 |
| 轻量结果 | `reports/sdad/` | 汇总JSON、逐人指标、配置、成果图 |
| 大型资源 | 外部持久化存储 | 原始EEG、缓存和模型权重 |

上传代码不代表数据和权重已经备份。

仅有汇总JSON可以支持结果查看，但不能完成模型复现或继续训练。

## 2. 历史资源位置

### 原服务器

| 资源 | 历史路径 |
|---|---|
| 原始数据 | `/mne_data/mne_data/` |
| session1 | `/mne_data/mne_data/session1/` |
| session2 | `/mne_data/mne_data/session2/` |
| 分组CSV | `/mne_data/mne_data/openbmi_bci_illiteracy_result.csv` |
| 实验根目录 | `/mne_data/mne_data/bci_runs/` |
| 预处理缓存 | `/mne_data/mne_data/bci_runs/cache/` |
| 后期准备数据 | `/mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared/` |

### 本地已使用的保存目录

`D:\Final-Version-BCI-Control-Car\outputs\seed42_diagnostics\`

以上是历史位置，不保证当前全部文件仍然存在。迁移时需要逐项确认。

## 3. 正式实验资源

以下路径均相对于原服务器实验根目录。

| 实验 | 初始化资源 | 训练结果 | 测试结果 |
|---|---|---|---|
| CSP+LDA参考 | session1训练数据及对应预处理 | 由参考脚本生成 | `pooled_csp_lda_test_001/` |
| 起点42／后期42 | `formal_a_001/online/baseline/seed_42/last.pt` | `group_domain_repeat42_001/` | `group_domain_repeat42_001_test/` |
| 起点43／后期43 | `dann_pair_43_44_001/online/baseline/seed_43/last.pt` | `late_subject_domain_43_001/` | `late_subject_domain_43_001_test/` |
| 起点43／后期44 | 同一seed43初始化资源 | `group_domain_source43_seed44_001/` | `group_domain_source43_seed44_001_test/` |

42起点来自原训练第118轮；43起点来自第107轮。

初始化资源除权重外，还应保留配套配置、CSP与标准化器。

## 4. 不同用途需要的文件

| 用途 | 必需材料 |
|---|---|
| 查看实验结论 | 汇总JSON、逐受试者指标、实验协议 |
| 绘制成果图 | 同一实验的逐人指标、固定分组、方法名称 |
| 执行模型推理 | 学生权重、模型配置、CSP、标准化器、预处理说明 |
| 复现后期实验 | 初始化学生与教师状态、数据、配置、代码版本、准备数据生成方式 |
| 完整恢复训练 | 还需优化器、学习率调度器、AMP及必要随机状态；以脚本保存能力为准 |

部分早期 `best.pt` 不包含教师状态，不能直接当作完整续训资源。

文件名相同不代表内容相同，迁移时应记录校验值。

## 5. reports/sdad/ 建议目录

以下是待建立的归档结构，不表示这些文件已经全部上传。

    reports/sdad/
    ├── README.md
    ├── reference_csp_lda/
    ├── source42_continue42/
    │   ├── control/
    │   └── group/
    ├── source43_continue43/
    │   ├── control/
    │   ├── group/
    │   └── subject/
    ├── source43_continue44/
    │   ├── control/
    │   └── group/
    ├── summary/
    └── figures/

各目录用途：

- `reference_csp_lda/`：传统参考的配置、汇总及逐人指标。
- `source*_continue*/`：对应运行的配置、验证和测试记录。
- `summary/`：三次配对结果及算术平均。
- `figures/`：成果图和图表数据来源。
- `README.md`：说明结果协议、文件对应关系及局限。

原始结果保留原文件名；如另外生成统一格式，需记录与原文件的对应关系。

## 6. 单次实验归档内容

每次运行至少保留：

- 实验名称。
- 训练和测试脚本。
- Git提交版本。
- 启动命令与配置。
- 预训练起点、后期随机种子。
- 固定分组与数据协议。
- 初始化及被测检查点的校验值。
- 训练历史、验证集逐人结果。
- 测试集逐人结果与汇总。
- 检查点选择规则。
- 完成、失败或中止状态。

材料缺失时明确标记，不把未核验内容写成已确认。


相关说明：

- [模块说明](README.md)
- [实验索引](experiment_index.md)
