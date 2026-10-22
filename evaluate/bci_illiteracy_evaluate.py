# ==============================================
# 项目：基于OpenBMI(Lee2019)，CSP-LDA基线评估BCI盲
# 参考论文：Lee, M. H. et al. (2019). EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy. GigaScience
# BCI盲判定规则：MI任务，CSP-LDA平均准确率 <70% → BCI illiteracy（BCI盲），>=70% → BCI正常
# ==============================================


# ==========【绝对第一行！！！在所有import前面！！】==========
import os
# 目标D盘路径
target_path = r"D:\Final-Version-BCI-Control-Car\mne_data"
# 先创建文件夹，如果不存在就自动新建（不用手动去资源管理器建）
os.makedirs(target_path, exist_ok=True)
# 设置环境变量
os.environ['MNE_DATA'] = target_path




import warnings
warnings.filterwarnings("ignore")
import requests
# 修复SSL，无递归版本
_original_get = requests.get
def new_get(url, **kwargs):
    kwargs["verify"] = False
    return _original_get(url,**kwargs)
requests.get = new_get

import pandas as pd
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import moabb
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery

dataset = Lee2019_MI()
dataset.download = False

# 1. 加载OpenBMI MI数据集（Lee2019）
# 首次运行自动从GigaScience下载原始EEG数据

paradigm = LeftRightImagery(fmin=8, fmax=30) # MI经典频段8~30Hz，和原始论文保持一致

# 3. 逐个被试评估，5折交叉验证，计算单被试平均准确率
subject_list = dataset.subject_list
result_table = []

for subj_id in subject_list:
    print(f"==== 正在评估被试 {subj_id} ====")
    try:
        # 获取该被试全部MI数据
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subj_id])
        # 【推荐】每个被试独立新建CSP+LDA管道，消除跨被试参数污染
        pipeline = make_pipeline(
            CSP(n_components=8),
            LDA()
        )
        # 5折分层交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_list = []
        for train_idx, test_idx in cv.split(X,y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc_list.append(accuracy_score(y_test, y_pred))
        mean_acc = np.mean(acc_list)
        # 【核心：BCI盲分类规则，来自原始OpenBMI论文】
        threshold = 0.70
        if mean_acc < threshold:
            bci_label = "BCI_illiterate(BCI盲)"
        else:
            bci_label = "BCI_literate(BCI正常)"
        result_table.append({
            "subject_id": subj_id,
            "mean_accuracy": round(mean_acc, 4),
            "BCI_status": bci_label
        })
    except Exception as e:
        print(f"被试{subj_id}处理失败，跳过，错误信息：{str(e)}")
        continue

# 4. 输出结果表格
df = pd.DataFrame(result_table)
print("\n======= 全部被试BCI评估结果 =======")
print(df)
# 保存csv，直接放进你的竞赛报告
df.to_csv("openbmi_bci_illiteracy_result.csv", index=False, encoding="utf-8-sig")
# 统计汇总
count_illiterate = len(df[df["BCI_status"]=="BCI_illiterate(BCI盲)"])
count_literate = len(df[df["BCI_status"]=="BCI_literate(BCI正常)"])
print(f"\n统计：BCI盲人数={count_illiterate}, BCI正常人数={count_literate}")
print(f"BCI盲比例：{count_illiterate / len(df):.3f}")
