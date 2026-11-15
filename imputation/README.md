# 联合多重补全（Joint Multiple Imputation）

针对 Colly 采集的企业经营面板中**营收/成本缺失或仅披露区间**，但部分**利润值和地区
营收总额已公布**的情形，本模块在一个统一的概率模型里联合补全，而不是逐行独立插补后
再硬调。

## 方法

### 1. 二元高斯回归（联合学习营收、成本及其相关性）

对企业 *i* 的特征向量 `x_i`（行业、地区、年份、员工数、资产等）建模

```
(R_i, C_i)' ~ N( B' x_i , Σ ),   Σ = [[ s_R² , ρ s_R s_C ],
                                      [ ρ s_R s_C , s_C² ]]
```

`B` 是回归系数矩阵，`Σ` 中的相关系数 `ρ` 与两个尺度一起从数据估计，因此营收与成本
的相关性被显式学习并参与补全。

### 2. 全部观测共同拟合（不分行独立处理）

把堆叠隐变量 `z = [R_0, C_0, R_1, C_1, …]` 上的每一种披露都写成线性观测：

| 披露形式 | 处理 |
| --- | --- |
| 确切营收/成本单元格 | 等式约束 `e_j' z = value` |
| 区间 `[lo, hi]` | 删失（截断）高斯观测，区间概率进入似然 |
| 已公布利润 | 对比值的等式 `R_i − C_i = profit` |
| 地区营收/成本总额 | 跨行求和等式 `Σ_{i∈g} R_i = total_g` |

等式观测拼成一个联合多元高斯似然，区间观测以条件边际高斯概率（复合似然）进入，
MAP 估计一次性拟合全部参数。构建阶段会校验等式系统的自洽性，矛盾的披露直接报错。

### 3. 在硬约束下联合采样（不存在"先独立补再硬调"）

贝叶斯 Gibbs 采样两步交替：

1. **隐变量块**：所有等式约束的可行集是一个仿射空间 `z = x0 + K u`（`K` 是等式矩阵
   的零空间基）。Metropolis 在 `u` 空间游走，任何违反区间或非负边界的提议都被拒绝。
   因此每个被保留的采样都**精确**满足利润恒等式、地区总额和区间边界，无需事后调整。
2. **参数块**：给定补全后的数据，`(B, Σ)` 从 Normal-Inverse-Wishart 共轭后验抽取，
   补全不确定性天然包含参数（含 `ρ`）的不确定性。

每条链从可行域内不同的后验高斯随机点启动，链间差异即补全间不确定性。

### 4. 多重补全与不确定性汇总

- 运行多条链，输出 split-R-hat、接受率等收敛诊断；
- 每条链作为一个独立补全集，按 **Rubin 规则**分解组内（within）与组间（between）
  方差及缺失信息比例（FMI），并给出单元格级与总均值级两层汇总；
- 经验后验分位数（5/25/50/75/95%）刻画每个补全单元格的不确定性。

**被地区总额完全覆盖的总均值，其组间方差在数学上必为零**（总额固定了组和）；
不确定性保留在未被钉死的单元格上，见 `rubin_per_cell.csv`。

## 硬性保证

- 原始确切观测在所有补全中逐位不变；
- 每套补全都满足 `利润 = 营收 − 成本`、地区总额、区间边界与非负性（测试强制校验）；
- 不把任何单一补全当作确定事实：始终输出多套明细与完整不确定性分布。

## 目录

```
joint_impute/
  data.py    单元格解析、设计矩阵、Problem 数据结构
  model.py   观测构建、二元高斯回归、联合 MAP 拟合、零空间
  impute.py  约束 Gibbs/MH 采样、收敛诊断、Rubin 汇总、约束校验
scripts/
  make_demo.py       生成带缺失/区间/利润/总额的演示面板
  run_imputation.py  命令行管线
tests/               单元与端到端测试（含逐 draw 约束校验）
```

## 使用

```bash
# 1. 生成演示数据（真实使用时替换为自己的 panel.csv / totals.csv）
python -I scripts/make_demo.py

# 2. 运行联合多重补全
python -I scripts/run_imputation.py \
  --panel /workspace/data/demo/panel.csv \
  --totals /workspace/data/demo/totals.csv \
  --group-cols region \
  --numeric employees assets \
  --categorical industry region year \
  --out-dir /workspace/output/demo \
  --chains 4 --draws 250
```

### 输入格式

`panel.csv` 必含 `id, revenue, cost` 列，可选 `profit` 与特征列。单元格写法：

- 确切值：`512.30`
- 区间：`[440.00, 580.00]`（方括号或圆括号均可）
- 缺失：留空或 `NA`

`totals.csv` 含分组列与 `total_revenue`（可选 `total_cost`）。

### 输出

| 文件 | 内容 |
| --- | --- |
| `imputed_draws_long.csv` | 每套补全的逐企业营收/成本/利润明细（含链编号） |
| `imputed_summary.csv` | 每单元格后验均值、标准差与 5–95% 分位数 |
| `rubin_per_cell.csv` | 单元格级 Rubin 方差分解与缺失信息比例 |
| `completed_posterior_means.csv` | 后验均值参考表（**非**确定事实，仅为方便） |
| `report.json` | 模型参数、MCMC 诊断、`ρ` 后验、约束校验结果 |

## 测试

```bash
python -I -m pytest tests/ -q
```

端到端测试会对每一个 MCMC draw 验证：确切单元格不变、利润恒等、地区总额、区间包含
与非负性，并检查后验覆盖率、相关性学习与 Rubin 分解的正确性。
