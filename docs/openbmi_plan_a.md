# OpenBMI 方案 A：蒸馏、弱特征增强与域对抗

## 实现范围

入口位于 `src/plan_a/`，以 OpenBMI 左右手二分类重新训练。底座保留原项目的 CNN、两层 Transformer、CSP 分支、融合分类头、ShallowConvNet 同步教师蒸馏，以及输出概率的方差正则。原来的 v65/v50pro 为另一数据集的四分类权重，不能直接当作 OpenBMI 二分类权重加载。

PDF 的方案 A 是结构参考。这里采用可复现的具体实现：

- 时序分支前端加入逐通道时间门控，增益范围为 `(0, 2)`，零初始化时增益为 1，避免一开始就把所有信号减半。网络学习哪些位置放大或抑制，并不能预先保证所选位置是 ERD/ERS。
- 192 维融合特征经过梯度反转层送入轻量域分类器。域 0 为正常表现组，域 1 为低表现组；两个域都使用运动想象真实标签。这是有监督的跨组对齐，并非无标签目标域适应。
- 推理只调用任务分支，不需要域标签或教师模型。
- `baseline`、`enhance`、`dann`、`plan_a` 四种变体分别对应蒸馏基线、仅增强、仅域对抗、完整方案 A。相同随机种子下，共有底座参数使用相同初始化。

损失为：

```text
L_hard = CE(student, labels) + 1e-5 * mean(var(softmax(student)))
L_KD = T² * KL(softmax(teacher / T) || softmax(student / T))
L = (1 - alpha) * L_hard + alpha * L_KD + domain_weight * L_domain
```

默认 `alpha=0.5`、`T=3`、`domain_weight=0.3`。GRL 系数随训练进度从 0 平滑增加至 1；它和 `domain_weight` 共同决定特征提取器受到的对抗梯度大小。域损失分别求两个域的平均后再等权相加，单个批次缺少一个域时跳过该批次的域损失，并记录次数。采样按域与任务类别的组合进行逆频率加权，所有变体使用相同采样策略。

## 数据与实验协议

本地数据布局：

```text
mne_data/
  session1/s1/sess01_subj01_EEG_MI.mat
  ...
  session2/s54/sess02_subj54_EEG_MI.mat
openbmi_bci_illiteracy_result.csv
```

受试者集合固定后，所有变体采用同一划分：

| 数据 | 用途 |
| --- | --- |
| session1 / EEG_MI_train | 模型、CSP、标准化器拟合；训练来源分组 |
| session1 / EEG_MI_test | 验证、调参、早停与最佳 checkpoint 选择 |
| session2 / EEG_MI_test | 最终一次性评估 |
| session2 / EEG_MI_train | 当前协议不用 |

这是同一批受试者的跨会话验证，不能当作未见受试者泛化结果。目标组的真实标签也参与训练。独立受试者泛化需要另设留一受试者协议。

数据从连续 `x` 滤波后按 `t` 切片。在本地原始文件中，`x[t:t+4000]` 与 `smt[:, trial, :]` 精确对应，因此这里不对 `t` 减一。标签依据文件的 `class` 字段映射为 `0=left, 1=right`。默认使用 11 个固定运动相关通道：FC1、FC2、FC5、FC6、C3、Cz、C4、CP1、CP2、CP5、CP6。每段为 4 秒、250 Hz、1000 点。

离线版对连续记录使用零相位 SOS 带通与陷波；在线版使用有状态因果 SOS 滤波，再按全局采样相位抽取至 250 Hz。两者均为四阶 0.5–100 Hz Butterworth 带通和 50 Hz 陷波。降采样前先带限；当前 IIR 阻带衰减有限，如后续优化抗混叠滤波，需同步修改训练和在线预处理并重训。

原始输入保留 OpenBMI MAT 的数值单位。均值与尺度仅由训练数据计算；更换采集设备时必须核对单位、通道、采样率。CSP 仅在训练集拟合，特征另行标准化。缓存键包含源文件路径、大小、修改时间、处理版本和模式；更换滤波实现时需增加 `data.py` 的缓存版本。

## 两种分组来源

默认 `--domain-source supplied` 使用现有 CSV 的准确率 `<0.70` 划分组别，满足沿用现有名单的需求。原名单生成脚本混用了两次会话并进行交叉验证，因此它包含最终测试会话的信息。这种设置的结果属于探索性结果，不能作为独立测试证明。

正式验证建议使用 `--domain-source training`。脚本仅在 session1 的训练块内，以 8–30 Hz、CSP-LDA 五折交叉验证重新确定每位受试者的组别。该过程每折独立拟合 CSP，阈值默认 0.70。由于通道、预处理和数据范围与原名单不同，得到的名单可能不同。这是本项目的操作性低表现定义，不是临床诊断。

无论选择哪种训练域，评估报告都会分别输出全部受试者、原名单低表现组、训练来源低表现组的结果。使用原名单训练后，即使只报告训练来源组也不能消除训练阶段已经引入的测试信息；正式结果必须整套实验使用 `--domain-source training`。

## 服务器安装

建议 Python 3.10 或以上。先依据服务器 CUDA 驱动安装匹配的 PyTorch，再安装其余依赖；requirements 不会自动保证 CUDA 可用。

```bash
python -m pip install -r src/plan_a/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

所有命令均从项目根目录执行，使用模块启动方式。数据和 CSV 必须单独同步至服务器，它们目前不在 Git 跟踪范围内。`--data-root`、`--groups-csv`、`--cache-root`、`--output-root` 可指定服务器路径。

先检查 CLI 和数据准备，准备过程会计算训练来源分组和 CSP，不训练神经网络：

```bash
python -m src.plan_a.train --help
python -m src.plan_a.train --subjects 1 5 --modes online --prepare-only --output-root outputs/plan_a_prepare
```

上述两名受试者在原名单中属于不同域。若使用训练来源分组后仅有一个域，需扩大受试者集合，脚本会拒绝只含单域的实验。

沿用现有名单的实验：

```bash
python -m src.plan_a.train --device cuda --amp --workers 4 --modes offline online --variants baseline plan_a --seeds 42 43 44 --domain-source supplied --output-root models/openbmi_supplied
```

正式对照与消融实验：

```bash
python -m src.plan_a.train --device cuda --amp --workers 4 --modes offline online --variants baseline enhance dann plan_a --seeds 42 43 44 --domain-source training --output-root models/openbmi_training
```

默认每批 8 段，最多 120 轮。Transformer 在 1000 个时间点上计算注意力，显存不足时减小 `--batch-size`，所有对照变体需采用相同配置。上述命令顺序运行各实验，不会自动多卡并行。数据准备占用数 GB 内存和较大的缓存空间。

断点恢复使用原命令并追加 `--resume`。每轮结束保存 `last.pt`，恢复模型、教师、优化器、调度器、混合精度 scaler、早停状态和历史；各轮随机种子固定，支持在轮次边界重放采样。中断轮次会重新执行。参数、核心代码、记录的环境版本或训练源数据改变时拒绝恢复。配置中记录代码指纹和 Python、PyTorch、NumPy、CUDA 版本。跨硬件或 GPU 非确定性算子仍可能导致数值差异。已有运行目录不会被普通训练命令覆盖。

每个实验保存：

```text
models/openbmi_training/online/plan_a/seed_42/
  config.json
  preprocessing.joblib
  best.pt
  last.pt
  history.json
  validation_subjects.json
```

`best.pt` 由验证集中目标组的受试者平均 balanced accuracy 选择。不会因单次达到某个准确率就提前终止整个实验，也不会按测试成绩挑选种子。`last.pt` 是训练恢复文件，`best.pt` 是评估与部署文件。仅加载自己可信的 joblib 和 checkpoint 文件。

新增 `.gitignore` 排除了 Python 缓存、默认方案 A 数据缓存、模型产物和评估产物，避免把大文件当作代码提交。需要同步到服务器的数据、名单 CSV 和训练产物应单独安排传输。

## 最终评估

参数选择完成后再运行测试，避免反复查看 session2 成绩调参：

```bash
python -m src.plan_a.evaluate --model-root models/openbmi_training --device cuda --output-root outputs/openbmi_training
```

报告包含逐受试者准确率、balanced accuracy、kappa，以及试次标签与预测概率。比较先对同一受试者的多随机种子取均值，再用受试者作为独立单位计算提升、受试者 bootstrap 95% 置信区间、双侧 Wilcoxon 配对检验，并对本次报告中的多项比较做 Holm 校正。不会把试次或种子冒充额外的受试者。少于两名受试者不做推断检验。

`comparison.json` 同时给出低于 70% 的人数变化。这个人数和显著性不能单独证明“解决 BCI 盲”，应结合效应大小、置信区间、个体退步人数和跨会话稳定性解释。是否显著提升必须等待真实训练和独立测试；代码实现本身不代表实验结论。

## 在线推理接入

新在线推理入口为 `src.plan_a.inference.OpenBMIStreamProcessor`。它加载二分类在线模型，接收带通道名的原始数据块，维护连续滤波状态，并对预先登记的 cue 后 4 秒窗口分类：

```python
from src.plan_a.inference import OpenBMIStreamProcessor

processor = OpenBMIStreamProcessor(
    "models/openbmi_training/online/plan_a/seed_42",
    input_channels=channel_names,
    sample_rate=1000,
    device="cpu",
)
processor.schedule_trial(onset_sample=cue_sample)
results = processor.process_chunk(raw_chunk)
```

`raw_chunk` 为 `[输入通道数, 本次采样点数]`，单位必须与训练 MAT 一致；`onset_sample` 为相对本次连续流起点的零基采样位置，应在包含该 cue 的数据块送入前登记。每次调用返回 0 个或多个结果。需要新建处理器来开始一条新的独立记录，不能每个数据块重置滤波器。处理器不连接串口、不直接控制小车。

时序门控在在线模型中使用左侧填充。原 CNN 和 Transformer 可以查看已收齐的整个 4 秒窗口；模型不在窗口完成之前输出最终分类，因此不会读取观测窗口之后的数据。实时调度、网络传输和硬件延迟仍需服务器训练后实测。

原 GUI/receiver 是 IV-2a 四分类、25 通道协议，新模型为 OpenBMI 左右手二分类、11 个选定通道，不能直接替换原权重。这里提供独立流式入口；GUI 的二分类模式和控制映射需要后续按实际采集设备接入。
