# BCI-Control-Car

基于运动想象 EEG（MI-EEG）的脑控小车项目，包含完整的离线训练、在线推理、发流/收流与可视化界面流程。

## 1. 目录

```text
BCI-Control-Car/
├─ src/
│  ├─ offline/
│  │  ├─ train_v42.py
│  │  ├─ train_v65_distill.py
│  │  └─ evaluate_v6		5_offline.py
│  ├─ online/
│  │  ├─ train_v50pro_online.py
│  │  ├─ evaluate_v50pro_stream.py
│  │  ├─ sender.py
│  │  ├─ receiver.py
│  │  ├─ main_app.py
│  │  ├─ debug_dll_load.py
│  │  └─ torch_bootstrap.py
│  │  └─receive_control_utf8.py
│  └─ tools/
│     ├─ generate_cue_timeline.py
│     └─ compare_logs.py
├─ models/
│  ├─ offline/v42/
│  ├─ offline/v65/
│  └─ online/v50pro/
├─ outputs/
│  ├─ timelines/
│  └─ logs/
│     ├─ online_accuracy/
│     └─ runtime/
├─ Data/                       # 原始数据
├─ assets/
│  ├─ slides/acc_data.pptx
│  └─ pyinstaller/main_app.spec
├─ docs/
├─ archive/
├─ BCI-Control-Car-GUI-win64/
│  ├─ BCI-Control-Car.exe

```

## 2. 环境依赖

建议 Python 3.8+。核心依赖：

- `torch`
- `mne`
- `scikit-learn`
- `scipy`
- `numpy`
- `matplotlib`
- `joblib`
- `pyserial`（串口控制时）
- `pyxdf`（GUI 离线读取 xdf 时）

## 3. 快速运行（按顺序）

### Step A: 离线训练（可选）

```bash
python src/offline/train_v42.py
python src/offline/train_v65_distill.py
python src/online/train_v50pro_online.py
```

训练/导出的模型会保存到：

- `models/offline/v42/`
- `models/offline/v65/`
- `models/online/v50pro/`

### Step B: 生成 Cue 时间线

```bash
python src/tools/generate_cue_timeline.py --subject 5
```

默认输出：`outputs/timelines/cue_timeline0tt.txt`  
`receiver.py` 与 `main_app.py` 默认读取该文件。

### Step C: 在线仿真（双进程）

终端 1（发送端）：

```bash
python src/online/sender.py
```

终端 2（接收+推理）：

```bash
python src/online/receiver.py
```

### Step D: GUI 演示（可选）

```bash
python src/online/main_app.py
```

## 4. 关键默认路径（已改为相对项目根目录）

- 数据 GDF：`Data/BCICIV_2a_gdf`
- 数据 MAT：`Data/A0xE/A0xE`
- 在线模型：`models/online/v50pro`
- 时间线：`outputs/timelines/cue_timeline0tt.txt`

如需改路径，直接在对应脚本顶部常量处修改。

## 5. 常用工具脚本

- 生成时间线：`src/tools/generate_cue_timeline.py`
- 对比在线日志：`src/tools/compare_logs.py`
- 离线评估（v65）：`src/offline/evaluate_v65_offline.py`
- 在线流式评估：`src/online/evaluate_v50pro_stream.py`

## 6. OpenBMI 方案 A（二分类）

新增的蒸馏基线、弱特征增强和 DANN 训练位于 `src/plan_a/`，支持离线与因果在线预处理、消融对照、断点恢复及跨会话评估。服务器运行命令、分组来源限制与在线推理接口见 [OpenBMI 方案 A 文档](docs/openbmi_plan_a.md)。原有 IV-2a 模型与运行流程保持独立。

方案 B 在 `src/plan_b/` 独立实现频带扰动、通道掩膜和监督式对比学习，保留蒸馏底座，使用单独的训练入口、权重与结果目录。训练及 A/B 配对比较命令见 [OpenBMI 方案 B 文档](docs/openbmi_plan_b.md)。


