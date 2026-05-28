# Band-Energy-Gated Fixed-Filter Headphone ANC

本项目当前技术路线已经简化为：

```text
频带能量检测 + 固定滤波器库硬选择 + FxNLMS residual
```

也就是说，当前主系统不再使用 logits、sigmoid 软权重、Kalman logits 平滑和 Predictive logits 预测。系统直接根据参考麦克风 `x(n)` 的频带能量，选择一个或少数几个对应频带的固定控制滤波器。

## 核心思想

耳罩 ANC 主链路：

```text
reference x(n)
    ↓
band-energy analysis
    ↓
energy smoothing + top-k hard selection
    ↓
binary alpha, alpha_i ∈ {0, 1}
    ↓
select fixed filters from filter bank
    ↓
W_gfanc
    ↓
frame-to-frame interpolation
    ↓
optional FxNLMS residual W_adapt
    ↓
W_final = W_gfanc + W_adapt
    ↓
anti-noise y(n)
    ↓
secondary path S(z)
    ↓
e(n) = d(n) + S(z) * y(n)
```

当前版本仍是离线仿真，不包含实时声卡闭环。

## 主要文件

主体模块：

```text
config.py              全局参数
path_model.py          P(z), S(z), S_hat(z) 路径模型
filter_bank.py         固定滤波器库加载、保存、组合
mock_gfanc.py          BandEnergyGate 频带能量硬选择器
fxnlms.py              FxNLMS residual 自适应
evaluate.py            降噪指标和画图
main_simulation.py     主仿真入口
```

训练脚本：

```text
train_filter_bank.py   训练固定滤波器库
```

检查诊断：

```text
test_modules.py        模块级测试
sanity_check.py        端到端自检
diagnose_outputs.py    读取 outputs 并生成诊断图
```

## 固定滤波器库

滤波器库包含 8 个固定 FIR 控制滤波器：

```text
W1: 50-80 Hz
W2: 80-110 Hz
W3: 110-150 Hz
W4: 150-200 Hz
W5: 200-250 Hz
W6: 250-300 Hz
W7: 50-150 Hz
W8: 150-300 Hz
```

训练：

```bash
python train_filter_bank.py
```

输出：

```text
data/filters/filter_bank.npy
data/filters/filter_bank_meta.json
data/filters/filter_bank_frequency_responses.png
```

`filter_bank.npy` 的 shape 应为：

```text
(8, 256)
```

## 频带能量门控

核心类：

```python
BandEnergyGate
```

位置：

```text
mock_gfanc.py
```

它做的事情：

```text
1. 对当前 frame 的 x_segment 计算各频带能量
2. 对频带能量做指数平滑
3. 选择能量最强的 top-k 个频带
4. 输出 binary alpha
```

现在的 `alpha` 不再是 0 到 1 之间飘的软权重，而是硬选择：

```text
alpha_i = 0 或 1
```

如果启用多个滤波器，默认会做平均：

```text
W_gfanc = sum(selected W_i) / selected_count
```

这样可以避免多个滤波器叠加导致增益过大。

## 关键参数

在 [config.py](./config.py) 中：

```python
fs = 8000
control_band = (50, 300)
filter_len = 256
secondary_path_len = 128
num_filters = 8
frame_len = 256
block_size = 64

band_gate_top_k = 3
band_gate_threshold_ratio = 0.2
band_energy_smoothing = 0.8
band_gate_min_hold_frames = 2
normalize_selected_filters = True
w_gfanc_norm_limit = 1.0

step_size = 0.03
leakage = 0.999
rho_adapt = 0.2
y_limit = 0.3
```

含义：

```text
band_gate_top_k
最多同时启用几个滤波器。

band_gate_threshold_ratio
只有能量达到最强频带一定比例的频带才可能被选中。

band_energy_smoothing
频带能量平滑系数，越大切换越慢。

band_gate_min_hold_frames
最小保持帧数，避免频带来回跳。

normalize_selected_filters
启用多个滤波器时是否平均。

w_gfanc_norm_limit
W_gfanc 整体范数上限。

y_limit
扬声器控制输出限幅。
```

## 仿真模式

主入口：

```bash
python main_simulation.py --mode all
```

当前主模式：

```bash
python main_simulation.py --mode fxnlms_only
python main_simulation.py --mode band_gated
python main_simulation.py --mode band_gated_fxnlms
```

含义：

```text
fxnlms_only
传统 FxNLMS 基线，W_gfanc = 0，只学习 W_adapt。

band_gated
只使用频带能量硬选择得到的 W_gfanc，不使用 residual。

band_gated_fxnlms
频带能量硬选择 W_gfanc + FxNLMS residual W_adapt。
```

为了兼容旧命令，以下模式仍可运行，但会映射到新路线：

```text
rule_gfanc          -> band_gated
oracle_gfanc        -> band_gated
gfanc_kalman        -> band_gated
gfanc_kalman_fxnlms -> band_gated_fxnlms
full_system         -> band_gated_fxnlms
```

## 输出

常见输出位于：

```text
outputs/
```

包括：

```text
error_before.wav
error_after.wav
anti_noise.wav
alpha.npy
band_energy_smooth.npy
W_gfanc.npy
W_adapt.npy
W_final.npy
psd_before_after.png
error_convergence.png
```

其中：

```text
alpha.npy
每帧二值滤波器选择，shape = (num_frames, num_filters)

band_energy_smooth.npy
每帧平滑后的频带能量

W_gfanc.npy
每个 sample 的基础固定滤波器控制器

W_adapt.npy
每个 sample 的 FxNLMS residual

W_final.npy
W_gfanc + W_adapt
```

## 自检

模块级测试：

```bash
python test_modules.py
```

端到端自检：

```bash
python sanity_check.py
```

正常输出应类似：

```text
SANITY CHECK REPORT
config: PASS
paths: PASS
filter bank: PASS
band gate: PASS
fxnlms: PASS
end-to-end: PASS
```

输出诊断：

```bash
python diagnose_outputs.py
```

诊断脚本会生成：

```text
diagnostic_error_waveform.png
diagnostic_alpha_over_time.png
diagnostic_band_energy.png
diagnostic_filter_responses.png
```

## 当前误差公式

全项目统一使用：

```text
d(n) = P(z) * x(n)
y_s(n) = S(z) * y(n)
e(n) = d(n) + y_s(n)
```

不要混用：

```text
e(n) = d(n) - y_s(n)
```

FxNLMS 的更新方向已经和当前误差公式匹配。

## 安全约束

扬声器输出：

```text
y(n) = clip(y(n), -y_limit, y_limit)
```

残差滤波器：

```text
||W_adapt|| <= rho_adapt * max(||W_gfanc||, eps)
```

基础控制滤波器：

```text
||W_gfanc|| <= w_gfanc_norm_limit
```

音频保存前：

```text
clip to [-1, 1]
```

## 推荐运行顺序

第一次运行：

```bash
python train_filter_bank.py
python test_modules.py
python sanity_check.py
python main_simulation.py --mode all
python diagnose_outputs.py
```

日常检查：

```bash
python sanity_check.py
```

## 如果效果不好，优先调这些参数

如果频带切换太频繁：

```text
增大 band_energy_smoothing
增大 band_gate_min_hold_frames
```

如果输出太大：

```text
降低 y_limit
降低 band_gate_top_k
降低 w_gfanc_norm_limit
降低 step_size
```

如果固定滤波器没效果：

```text
检查 data/filters/filter_bank_frequency_responses.png
重新训练 filter_bank
用真实 S(z) 和真实噪声重训滤波器库
```

如果 residual 太强：

```text
降低 rho_adapt
降低 step_size
```

## 当前路线一句话

```text
根据 x(n) 的频带能量，硬选择少数固定控制滤波器，
通过 W_gfanc 生成基础反噪控制，
再用 FxNLMS 学一个受限的 W_adapt 做残差微调。
```
