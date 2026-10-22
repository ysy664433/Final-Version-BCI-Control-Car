import moabb
print(moabb.__version__)
# ✅ 新版名字：Lee2019_MI
from moabb.datasets import Lee2019_MI
ds = Lee2019_MI()
print("被试ID列表：", ds.subject_list)
