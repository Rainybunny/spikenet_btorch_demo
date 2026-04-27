# 静态 Bump 复现测试报告（修订版）

## 1. 报告定位与结论边界

本报告记录的是：将 SpikeNet 的 Qi & Gong 2022 静态高斯输入实验迁移到 btorch 后，完成一套可运行、可视化、可追溯的缩放版测试流程。

当前阶段结论边界：

1. 已完成缩放版（hw=10）定性复现流程，能够观察到随输入宽度变化的空间活动形态差异。
2. 尚未完成原文同规模、全参数扫描、5 次重复的定量复现实验。
3. 因此“趋势一致”可成立，“严格数值复现”尚不能下结论。

---

## 2. 迁移实现与关键修正

### 2.1 已实现模块

1. 在 btorch 新增多延迟指数突触通道 `SpikeNetExponentialPSC`。
2. 新增双通道组合突触 `SpikeNetCompositePSC`（AMPA + GABA）。
3. 在 `demo/qi_gong_2022_static_bump.py` 中实现静态高斯输入测试脚本。
4. 在 `demo/qi_gong_2022_full_scale.py` 中保留全尺度构图工具。

### 2.2 统一单位口径（重要）

1. SpikeNet C++ 单位体系为 msec + mV + nF + µS + nA。
2. 迁移后在静态脚本中采用电导型到电流的显式换算：

$$
I = g \cdot (E_{rev} - V)
$$

3. `SpikeNetCompositePSC.psc` 在提供 `E_ampa/E_gaba` 的场景下表示总电导（µS），通过 `current(v)` 才得到 nA 电流。
4. 默认 `RecurrentNN` 的 `synapse.psc + x` 路径适合电流型突触；对于电导型突触，本实验脚本采用手工循环并显式调用 `synapse.current(v)`，避免单位混淆。

### 2.3 外部输入标度修正

早期版本把 `N_ext` 额外乘进单事件权重，导致电流放大约 1000 倍，网络饱和发放。修正后：

1. 事件均值由下式给出（每神经元每步）：

$$
\lambda = N_{ext} \cdot rate_{kHz} \cdot dt_{ms} / 1000
$$

2. 单事件保持 `g_ext`，不再乘 `N_ext`。
3. 外部输入通过指数衰减电导状态累积，再转为电流输入。

---

## 3. 从测试数据生成开始的完整测试流程

下面给出本次静态实验的全流程，顺序与代码执行一致。

### 3.1 测试配置准备

1. 读取配置 `Cfg`：网络规模、神经元参数、连接参数、输入参数、随机种子。
2. 固定 `seed`，确保图结构与统计可复现。
3. 设定仿真时间步 `dt=0.1 ms`，总时长 `T_ms=2000 ms`。

### 3.2 生成测试网络数据（结构数据）

这一阶段生成“测试样本本体”，即每次仿真要用的完整网络拓扑与参数。

1. 生成 E 群体二维规则网格坐标（torus 上等间距点）。
2. 生成 I 群体二维随机均匀坐标。
3. 按距离依赖概率生成四类连接：EE、I→E、E→I、I→I。
4. 对 EE 边使用 log-normal 权重采样；其余连接使用对应常数权重。
5. 为每条边采样离散延迟步（1 到 `delay_max_steps`）。
6. 将边按延迟步分桶，构建 `{delay_step: sparse CSR matrix}`。
7. 实例化 `SpikeNetNeuron` 与 `SpikeNetCompositePSC`，得到可仿真的网络对象。

输出产物：

1. 结构化坐标数据（`coords_e`）。
2. 稀疏延迟权重桶（exc/inh）。
3. 已初始化参数的 neuron/synapse 模块。

### 3.3 生成测试输入数据（刺激数据）

对每个测试条件（每个 `sigma`），生成一份独立输入数据。

1. 构造静态高斯速率图：

$$
rate_i = rate_{ext,E} \cdot \left(1 + contrast \cdot e^{-\frac{r_i^2}{2\sigma^2}}\right)
$$

2. I 群体使用常数速率 `rate_ext_I`。
3. 将速率图转为每步事件概率 `p = N_ext * rate * dt / 1000`。
4. 每步采样外部事件并写入外部电导状态。

说明：当前脚本用 Bernoulli 采样近似 Poisson 事件计数（在本参数下均值较小，近似可用），但严格对齐原始 C++ 可进一步替换为 Poisson 采样并做对比。

### 3.4 执行仿真（时间推进）

对每个 `sigma` 条件，执行完整时间循环：

1. 初始化 neuron/synapse 内部状态。
2. 计算外部电导衰减系数 `alpha_ext = exp(-dt/tau_ampa_ext)`。
3. 每个时间步执行：
  - 采样外部事件（E 和 I）。
  - 更新外部电导状态 `ext_gs`。
  - 读取膜电位 `v_now`。
  - 计算复发突触电流 `rec_current = synapse.current(v_now)`。
  - 计算外部电流 `ext_current = ext_gs * (E_ampa - v_now)`。
  - 求和得到 `total_input` 并推进神经元一步。
  - 用当前脉冲更新突触历史缓冲。
  - 记录 E 群体脉冲时间与神经元索引。

输出产物：

1. 脉冲时间数组 `spike_t`。
2. 脉冲神经元索引数组 `spike_n`。

### 3.5 统计与可视化（测试结果生成）

1. 统计每个条件总脉冲数与均值频率：

$$
  ext{mean\_rate\_hz} = \frac{N_{spike}}{N_e \cdot T_{sec}}
$$

2. 取最后 500 ms 脉冲，计算每个 E 神经元发放率。
3. 将发放率重排为二维网格，生成空间热图。
4. 绘制对应输入速率图。
5. 输出最后一个 `sigma` 条件的 raster 图。

输出文件：

1. `demo/qi_gong_2022_static_bump.png`
2. `demo/qi_gong_2022_static_raster.png`
3. （脚本还包含双高斯实验并输出 `demo/qi_gong_2022_double_gaussian.png`）

### 3.6 结果归档与复核

1. 保存图像产物与控制台日志。
2. 记录运行环境（conda 环境、Python 版本、关键依赖）。
3. 核对结论是否与图像和统计一致，不做超出证据范围的推断。

---

## 4. 与原始 SpikeNet 方案的一致性与差异

### 4.1 一致项

1. 核心神经元与突触参数保持与原模型一致（例如 `tau_ref`、`g_mu`、`g_EI`、`g_IE`、`g_II`、`rate_ext_E`、`contrast` 等）。
2. 使用 torus 上的空间高斯输入与距离依赖连接。
3. 使用随机延迟并保留 E/I 两群体结构。

### 4.2 缩放与近似项

1. 网络规模由 `hw=31` 缩放到 `hw=10`。
2. 仿真时长由 10 s 缩放到 2 s。
3. `sigma` 由 10 点全扫描缩减为 3 个代表值。
4. 当前仅单次运行，未执行每条件 5 次重复统计。
5. 外部事件采样采用 Bernoulli 近似（待补 Poisson 对照实验）。

---

## 5. 当前测试证据与保守结论

### 5.1 可确认事实

1. 静态脚本在 `spike` 环境可运行并输出图像结果。
2. 在已跑通条件中，能观察到输入宽度变化引起的空间发放形态变化。
3. 已定位并修正外部输入 1000 倍标度错误。

### 5.2 现阶段结论（保守表述）

1. 可以认为已完成“缩放版、定性趋势复现”的工程实现。
2. 暂不宣称“与原文严格定量一致”或“完整 AI 状态已被严格验证”。
3. 后续需补全多次重复与全扫描后，再给出稳健的统计结论。

---

## 6. 复现实操步骤（建议）

1. 启动环境：`conda activate spike`
2. 运行静态实验：`python demo/qi_gong_2022_static_bump.py`
3. 检查输出文件是否更新：
  - `demo/qi_gong_2022_static_bump.png`
  - `demo/qi_gong_2022_static_raster.png`
4. 记录每个 `sigma` 的 `total_spikes` 与 `mean_rate_hz` 到表格。
5. 对比三类证据是否一致：控制台统计、空间热图、raster。

---

## 7. 待完善事项

1. 完整扫描 `sigma in linspace(0, pi, 10)` 并保存每点统计。
2. 每个 `sigma` 至少运行 5 次，报告均值、标准差与置信区间。
3. 增加 Bernoulli 与 Poisson 外部事件模型的对照测试。
4. 增加 AI 指标（如 ISI CV、Fano factor），替代仅凭 raster 目测判断。
5. 在可接受资源下执行 `hw=31` 全规模对比实验，补充与原始 SpikeNet 的定量差异分析。
