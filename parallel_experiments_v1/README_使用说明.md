# 八个候选，一次提交，受限并发

这是可检验的候选训练配方集合，不是八种已证实有效或原创的算法。
默认全部54人、种子42、online、最多120轮、patience30、batch size4、CUDA+AMP。
保留原采样、数据划分和基础优化器，不做个人微调，不读取session2。
原基线、原70.40%模型和失败的类别内对齐均不重跑。

## 候选

|任务|改动|起点|
|---|---|---|
|subject_adaptive|训练困难的低表现受试者CE权重最多1.5倍|原蒸馏基线，无门控、无DANN|
|subject_uniform|低表现组统一1.25倍CE权重|同上|
|subject_all|全体受试者中训练困难者最多1.5倍CE权重|同上|
|no_kd_control|去掉学生KD梯度，CE系数仍为0.5|同上；为控制随机流程仍计算教师|
|delayed_dann|域损失0.03，第10轮之后渐进启动，过渡40轮|原门控+DANN+蒸馏|
|gradient_protected|将与分类梯度相反的合成辅助梯度投影，记录触发批次数|原门控+DANN0.03+蒸馏；非原PCGrad完整复现|
|weak_contrast|对比权重0.01，不使用随机EEG增强|原B的contrast分支，仍有两个视图和蒸馏|
|weak_augment|频带强度0.1、噪声0.01、不掩膜、不用对比损失|原B的augment分支，仍有蒸馏|

重加权前三种均先正常训练5轮。自适应权重只由历史训练CE的EMA确定，不根据验证或测试成绩决定谁被加权。
同一批次中权重相同则使用原始CE；否则使用归一化加权CE。权重上限不代表性能保证。

## 上传

解压ZIP，把整个 parallel_experiments_v1 文件夹上传到服务器项目根目录：

    /root/Final-Version-BCI-Control-Car/parallel_experiments_v1/

目录内所有Python文件应处于同一级，不要分别改名。原src文件不修改。
服务器需要之前已经可用的GPU环境，以及原基线结果和挂载数据。

## 一次启动

先停止自己此前仍在训练的任务，或使用 --max-parallel 1。不要未经检查kill其他进程。
数据会先准备一次，保存到本批次的 _prepared 目录，之后各模型通过CPU内存映射读取同一份张量。
因此先等待 preparation.log 中54人准备完成，再看各任务日志。

```bash
cd /root/Final-Version-BCI-Control-Car
nohup python -u parallel_experiments_v1/launch_experiments.py \
  --run-root /mne_data/mne_data/bci_runs/parallel_001 \
  --max-parallel 2 \
  >> /mne_data/mne_data/bci_runs/parallel_001_master.log 2>&1 < /dev/null &
```

只启动一次。存在同名run-root会拒绝覆盖，不会自动续训或重跑失败任务。
这是八个完整实验，总计算量仍可能需要数小时；并发不保证比顺序运行更快。梯度保护任务多一次反向梯度计算，会更慢。
默认每任务估计预留8GB显存，另留2GB余量；24GB显卡最多放行两个任务。
同时检查剩余CPU内存。估计值无法保证不OOM；发生错误会标记failed，不会自动反复重试。
显存不足时保持排队；max-parallel是上限，不保证始终占满。
只支持当前单个可见GPU的场景，不自动扩展到多卡。

## 查看

```bash
tail -f /mne_data/mne_data/bci_runs/parallel_001_master.log

python parallel_experiments_v1/launch_experiments.py \
  --run-root /mne_data/mne_data/bci_runs/parallel_001 --status

tail -f /mne_data/mne_data/bci_runs/parallel_001/subject_adaptive.log
```

状态queued是等待，running是运行，completed是正常完成，failed是报错；先检查对应log。
Ctrl+C退出tail不会停止训练。SSH断开后nohup继续，关机/释放容器会中断。

## 统一停止整批

```bash
python parallel_experiments_v1/launch_experiments.py \
  --run-root /mne_data/mne_data/bci_runs/parallel_001 --stop
```

只向本批次记录且命令路径匹配的进程发送TERM，并阻止后续排队任务启动。不删除任何结果。
状态stop_requested需要等待进程退出；若日志提示仍在关闭，手动检查PID，脚本不会强制kill其他进程。

## 结果

每个任务都在 /mne_data/mne_data/bci_runs/parallel_001/任务名/online/变体名/seed_42 下。
best.pt和validation_subjects.json沿用原低表现组均值选模，可与旧基线的同规则结果进行探索性比较。
best_threshold.pt和validation_subjects_threshold.json采用低表现组达标人数、全体达标人数、低表现组均值依次选模；
这些属于另一选模规则，不能与按均值选出的旧基线直接比较后宣称纯算法收益。
每轮逐人结果在validation_epochs，训练记录在history.json。

自动生成 experiment_summary.csv，只汇总主best.pt对应的结果，包括向上/向下跨70%及净达标人数。
也可随时执行：

```bash
python parallel_experiments_v1/launch_experiments.py \
  --run-root /mne_data/mne_data/bci_runs/parallel_001 --summarize
```

未完成/失败的任务会保留状态，不能与完整训练当成等预算结果。候选很多时验证集会被反复利用，最高结果存在选择偏差。
70%定义为>=70%，固定25人低表现组和54人全体都报告；不把阈值表述为临床诊断。
选出候选后固定方法，再进行独立session2最终评估。现在不自动测试session2。

## 可选：先只提交部分任务

在启动命令增加 --jobs subject_adaptive delayed_dann weak_contrast 可以只跑三项。
默认不加--jobs就是八项全部排队。不推荐直接把max-parallel改成8；显存容量限制仍会生效。

## 已有检查与限制

本地CPU合成数据测试涵盖原基线兼容性、权重上限、三个配方的一轮训练、保存记录、共享张量读取和汇总。
Linux GPU并发调度、信号停止与真实EEG八项全流程尚未实机验证。
本包仅添加独立文件，不安装依赖、不上传数据、不改动原仓库代码，不触发远端训练。
