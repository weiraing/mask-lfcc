# 校正训练模块（Correction Training）实现报告

> 参考：**Correction Training for Diffusion Models**
> https://github.com/sunovivid/correction-training · arXiv 2610.14136
> 本文件记录该方法在本项目（Refusion / IR-SDE / 像素空间 / ε 预测 / 掩码条件）上的
> 代码审查结论、适配决策、实现清单与实验方案。

---

## 1. 原方法在做什么（源码核对结论）

读了 `src/models/denoiser.py`、`src/models/correction.py`、`configs/train/*.yml` 之后，
原方法的机制可以精确地写成三句话：

1. **校正器 = 去噪器的完整副本**。`self.corr_net = copy.deepcopy(self.net)`，
   随后 `initialize_corrector()` 把去噪器权重原样灌进去
   （`corr_net.load_state_dict(self.net.state_dict(), strict=True)`）。
2. **两个网络被同一个干净目标监督，但梯度在去噪器预测处截断**。
   `losses()` 里 `target` 只算一次，`base_loss = MSE(target, velocity)`、
   `corr_loss = MSE(target, corrected)`，`corrected` 由 `refine()` 产生，
   而 `refine()` 里对 `skip` 和 `value` 分别 `detach()`。
3. **采样时每一步都复用校正器**，`K = corrections_per_step` 次迭代，
   `z` 与 `t` 在 K 次迭代内保持不变（只更新 proposal）。

`combine` 有四种：`direct` / `residual` / `gated` / `base_anchor`。
其中 `residual` 和 `gated` 会把校正网络的输出头**零初始化**
（`nn.init.zeros_(self.corr_net.final_layer.linear.weight)`），所以
**初始时刻校正器是严格恒等映射**——这是它比"再训一个网络"安全得多的根本原因。

`base_anchor` 只对 velocity 参数化有意义（固定去噪器的那一半分量）。
IR-SDE 是 ε/x0 参数化，两个分量互为仿射变换，`base_anchor` 会退化成 `direct`，
因此本实现**不提供** `base_anchor`，并在代码里注明了原因。

---

## 2. 适配到 IR-SDE 时改了什么

| 原方法假设 | 本项目的实际情况 | 处理 |
|---|---|---|
| 潜空间 + VAE，velocity 参数化 | **像素空间**，`IRSDE` 直接作用在图像上，模型预测 **ε** | 校正空间改为 `eps` / `x0`，用 `sde.get_init_state_from_noise` / `get_noise_from_init_state` 互转（两者是**精确互逆**，自检里断言误差 < 1e-6） |
| `base_anchor` 固定 eps 分量 | ε/x0 互为仿射变换，没有独立分量 | 不实现，`CorrectionSpec.validate()` 里显式说明 |
| 类别标签 `labels` | 无类别，改为掩码 `mask` + 观测 `mu` | 校正器与去噪器同签名 `(x_t, mu, t, mask)` |
| `gated` 的 gate 无激活、无界 | 无界增益在上一轮 MTSCE 里是发散来源 | gate 用 `tanh` 有界，且**乘在位移上**而不是在 `skip` 与 `output` 之间插值 |
| `gate_mix: blend / difference` | 两者代数上完全等价（`(1-g)·s + g·o ≡ s + g·(o-s)`） | 只保留一种形式，配置项 `gate_mix` 移除 |

**一处对原实现的修正**：原 `refine()` 的 `gated` 分支是
`output = (1-g)·skip + g·output`，而 `output` 是绝对预测值。
当 `gate_init = 0` 时 `g = 0`，于是 `∂L/∂output = g·∂L/∂combined = 0`，
**输出头拿不到梯度**，必须先学会 gate 才能学 head。本实现改成
`combined = skip + tanh(g)·pred`（`pred` 是位移），`g = 0` 时 head 仍有一阶梯度，
收敛快得多，同时保留"gate 为 0 则完全不动"的恒等性质。

---

## 3. 关键量化依据

### 3.1 为什么校正必须直接学 ε 位移，而不是在 x0 空间学

`A(t) = exp(θ̄_t·dt)` 是 ε 误差被放大到 x0 误差的系数。按本项目的
cosine 调度（T=100, max_sigma=50/255, eps=0.005）实算：

| t | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| `A(t)` | 1.00 | 1.02 | 1.10 | 1.32 | 1.81 | 2.87 | 5.31 | 27.5 | **200** |

训练损失对 ε 的敏感度是 `c(t) = σ²·dt / σ̄(t)`：

| t | 1 | 10 | 30 | 50 | 80 | 100 |
|---|---|---|---|---|---|---|
| `c(t)` | 0.0050 | 0.0080 | 0.0142 | 0.0223 | 0.0364 | 0.0400 |

- 在 **ε 空间**学位移，梯度尺度是 `c(t)`，全程只差 8 倍，**训练稳定**；
- 在 **x0 空间**学位移，梯度尺度是 `c(t)/A(t)`，t=100 时降到 0.0002，
  等于在高噪声段"学不动"。

所以默认 `output: eps`、`combine: residual`。`output: x0` 仍然支持
（`combine` 必须是 `residual`/`gated`），作为消融项。

### 3.2 校正器到底新增了多少参数

| | 参数量 | 说明 |
|---|---|---|
| 去噪器（width=64, enc=[1,1,1,28]） | ≈ 67.80 M | 与 `refusion_mask` 完全一致 |
| 校正器 | 去噪器 + 256 | 只多一个 `proposal_proj`（3→64 1×1 卷积 + bias） |
| 净新增 | **+256** | 约 +0.0004% |

（精确值由 `python verify_correction.py` 最后一节打印。）

注意：校正器**总**参数是 2 倍（≈67.8M × 2），因为它是一份去噪器的副本。
这是原方法的代价，也是它有效的原因——它从"已经很好的函数"出发，
而不是从随机初始化出发。**冻结**去噪器后，额外显存只有校正器自身的
激活和优化器状态；去噪器前向在 `requires_grad=False` 下不建图，不占激活内存。

### 3.3 推理代价

`steps: 1` → 每个反向步 2 次网络评估（**2× NFE**）。
`apply_t_range: [1, 40]` → 额外代价降到 40%（总 1.4× NFE），
且校正器是在 `(x_t, t)` 的边缘分布上训练的，只在子区间使用**不会引入训练/推理不一致**。

---

## 4. 为什么这次有理由认为它会涨点

这是前两个模块失败的地方，所以单独说清楚。

| | MTSCE | SDRC | **本模块** |
|---|---|---|---|
| 作用位置 | 去噪器中间特征 | 采样器的 x0_hat | 采样器的 ε 预测 |
| 学习信号 | 间接（经 loss 反传） | 直接（GT 监督） | 直接（GT 监督，与去噪器同一目标） |
| 初始状态 | 恒等但随机初始化 | 恒等但随机初始化 | **恒等，且背后是一整套预训练权重** |
| 是否可能伤到基网 | 会（联合训练） | 不会 | 不会（基网冻结） |
| 参数量 | 9.7M（+12.7%） | 0.44M | 68M（副本），净新增 256 |
| 第 0 步验证 PSNR | ≠ 基线 | = 基线 | **逐比特 = 基线** |

核心论证：基网冻结、校正器输出头零初始化，所以

```
loss_c(θ=0) ≡ loss_b
```

训练**只能**让 `loss_c` 下降，也就是"在固定去噪器的基础上再做一步下降"。
这是梯度提升（gradient boosting）的形式：第 0 步等于基线，之后每一步都在
降低同一个监督目标。因此它不是"赌新模块有用"，而是"在基线上继续优化"。

`verify_correction.py` 会把这条性质断言成**逐比特相等**：
挂上未训练的校正器后 `reverse_sde` 的 100 步输出与基线 `torch.equal == True`。

**必须承认的风险：曝光偏差（exposure bias）。** 训练用的 `x_t` 来自对 GT 的
前向加噪，采样时的 `x_t` 来自反向轨迹，两者分布不同。这是所有"训练时用真值加噪"
的扩散模型共有的问题，基网自己也是这么训的，所以风险可控，但**不是零**。
缓解手段有三个，都在配置里：

1. `detach_input` / `detach_skip` 截断梯度，校正器只学"修正量"，不会去改基网；
2. 监控 `x0_gain`（见 §6），它一旦长期为负就立刻停；
3. `apply_t_range` 可缩到 `[1, 40]`，只在 x0 估计最有意义的区间施加校正。

---

## 5. 实现清单

### 新增

| 文件 | 作用 |
|---|---|
| `models/modules/correction.py` | 配置校验、`CorrectionSpec`、`CorrectionGate`、`refine()`（训练/采样唯一入口）、从基网初始化校正器 |
| `models/correction_model.py` | `CorrectionModel`：冻结去噪器 + 训练校正器；训练损失、日志、保存、推理挂载 |
| `options/train/refusion_mask_correction.yml` | 训练配置（C 臂） |
| `options/test/refusion_mask_correction.yml` | 评测配置（C 臂） |
| `verify_correction.py` | 自检（CPU，10 秒，无需数据与权重） |
| `CORRECTION_IMPLEMENTATION.md` | 本文件 |

### 修改

| 文件 | 改动 |
|---|---|
| `models/modules/DenoisingNAFNet_arch.py` | 新增 `proposal_ch` 参数与零初始化 `proposal_proj`；`forward(..., proposal=None)`。`proposal_ch=0` 时**不构建任何模块**，基网逐比特不变 |
| `models/modules/__init__.py` | 导出 correction 组件，移除 sdrc |
| `models/__init__.py` | `model: correction` → `CorrectionModel` |
| `models/denoising_model.py` | **移除** SDRC 挂载点，恢复成干净的基线（`refusion_mask.yml` 现在逐比特等于原始实现） |
| `utils/sde_utils.py` | `set_corrector` / `_correction_active` / `_correct_x0`；`reverse_sde` 与 `reverse_posterior` 走校正分支 |
| `test.py` | 报告校正实际生效步数，为 0 时告警 |

### 删除（SDRC 已按要求移除）

`models/modules/sdrc.py`、`models/sdrc_model.py`、`verify_sdrc.py`、
`options/train/sdrc_dark.yml`、`options/test/refusion_mask_sdrc.yml`、
`SDRC_IMPLEMENTATION.md`

### 设计上刻意保证的三条

1. **零初始化 ⇒ 恒等**。`combine: residual/gated` 时输出头清零，`pred = 0`，
   `combined = skip + 0`，浮点上逐比特等于去噪器预测。
2. **窗口外走原代码路径**。`_correction_active(t)` 为假时完全不经过
   `refine()`，不做 eps↔x0 往返（虽然精确，但 float32 下不保证逐比特）。
3. **`steps: 0` 是真正的无操作**，可用于"同一份代码跑基线"的对照。

---

## 6. 怎么跑

### 第 0 步：自检（必做，10 秒）

```bash
cd codes/config/inpainting
python check_env.py -opt=options/train/refusion_mask_correction.yml   # 新机器上先跑这个
python verify_correction.py
```

`check_env.py` 按 `train.py` 完全相同的方式解析路径，检查：

- `project root` / `experiments root` 是否与预期一致；
- `pretrain_model_G` 是否存在，以及它的 state_dict 键是否与 `network_G` 的配置**逐键匹配**
  （宽度/block 数写错、拿错权重，都会在这里立刻报出来）；
- 训练/验证集的 GT 与 LQ 文件数是否相等、**文件名列表是否完全一致**
  （`LQGTMaskDataset` 按排序下标配对，名字不同会让 GT/LQ 静默错位，这是最难查的部署事故）；
- mask 能否按文件名主干与 GT 配对；
- `gpu_ids` 与可见 GPU 数是否匹配。

失败项非零退出。`verify_correction.py` 再断言方法本身的数学性质：

- 默认配置解析正确、非法组合被拒绝；
- `eps → x0 → eps` 误差 < 1e-6；
- `residual` / `gated` 在 steps=1、2 与 t=1/20/60/100 下**逐比特返回原预测**；
- `direct+eps` 起点等于去噪器输出；
- 梯度只进校正器，不进基网；`detach_*` 真的截断计算图；
- `proposal` 特征通路是活的；`proposal_ch=0` 不构建任何东西；
- 挂上未训练校正器后 `reverse_sde` 与基线**逐比特相同**；
  `apply_t_range=[1,40]` 时恰好调用 40 次；
- 参数量与推理代价。

任何一项失败就停下来，不要开始训练。

### 第 1 步：训练

```bash
python train.py -opt=options/train/refusion_mask_correction.yml
```

配置默认：

| 项 | 值 | 理由 |
|---|---|---|
| `pretrain_model_G` | mask 基线 `30000_G.pth`（33.62 dB） | 校正器从 B 出发 |
| `freeze_base` | `true` | 不重开 MTSCE 的失败模式 |
| `combine` / `output` | `residual` / `eps` | §3.1 |
| `train_steps` / `steps` | 1 / 1 | 2× NFE |
| `optimizer` | `AdamW` | 零初始化头在 Adam 下收敛快 |
| `lr_correction` / `lr_backbone_scale` | 1e-4 / 0.2 | 头快、主干轻推 |
| `weight_decay_correction` | 0 | AdamW 的衰减会侵蚀复制过来的预训练权重 |
| `grad_clip` | 1.0 | 上一轮崩掉的直接原因之一 |
| `niter` | 30000 | 与基线同步数，保证对比公平 |
| `batch_size` | 16 | **必须与基线一致**（基线实际是 16，不是配置文件里的 4） |

### 第 2 步：看日志，不要看 PSNR

训练日志每 100 步打这些量：

| key | 含义 | 判据 |
|---|---|---|
| `x0_gain` | `1 - MSE(x0_corr, GT) / MSE(x0_base, GT)`，同一 `x_t`、同一 `t` 的**配对**比较 | **这是唯一的早期判据**。500 步内应稳定为正 |
| `delta_rms` | 校正量在 ε 空间的 RMS | 应从 0 单调上升后趋于平台；突然飙升 = 不稳 |
| `loss` / `loss_base` | 校正后 / 校正前的监督损失 | `loss` 应始终 ≤ `loss_base` |
| `grad_norm` | 裁剪前梯度范数 | 应平稳；持续等于 `grad_clip` 说明在硬顶，需要降 lr |

**早停规则**：跑 2000 步，如果 `x0_gain` 的滑动平均 ≤ 0，**立即停**。
校正器面对的是一个固定的、已经被充分优化的去噪器，如果连单步的配对误差都降不下来，
放到多步采样里更不可能赢——这一条能省掉后面三臂 × 30000 步的算力。

### 第 3 步：A/B 对比（必须配对）

```bash
python test.py -opt=options/test/refusion_mask.yml            --seed 0
python test.py -opt=options/test/refusion_mask_correction.yml --seed 0
```

两份配置指向**同一个** `30000_G.pth`，唯一差别是校正器。
`--seed 0` 会让每张图的反向 SDE 噪声流固定，于是逐图差异只来自模型本身，
±0.05 dB 的差异才可信。不要用两次独立采样的均值去比，那有 ±0.1 dB 的采样抖动。

日志末尾会打印校正实际生效的步数，若为 0 说明窗口配置有问题，PSNR 差异纯属噪声。

---

## 7. 论文里怎么写这三臂

| 臂 | 配置 | 隔离出的贡献 |
|---|---|---|
| A | `options/train/refusion.yml`（LQGT，无 mask） | 改进前的 Refusion |
| B | `options/train/refusion_mask.yml`（LQGTMask） | A→B = 掩码通道（Degradation-Aware Latent） |
| **C** | `options/train/refusion_mask_correction.yml` | **B→C = 校正训练（本文第二个模块）** |

只有 B→C 是本文第二个创新点的证据。C 与 B 使用同一份 `30000_G.pth`、
同一个 batch、同一个步数、同一个 seed，唯一变量是校正器——这一点比"总 PSNR 高于
Refusion"重要得多。

同时报告：参数量（B ≈67.8M / C ≈135.6M，净新增 256）、推理 NFE（1× / 2×，
或 `apply_t_range=[1,40]` 时的 1.4×）。审稿人一定会问这个。

---

## 8. 已知限制

- **本机没有 torch**，只做了 `py_compile` 静态检查。`verify_correction.py` 与训练
  都**没有实际运行过**。第一步务必先跑自检。
- 配置文件里的 `pretrain_model_G` / `pretrain_model_C` 是**相对运行目录**的路径
  （`../../../experiments/...`），与服务器无关：`options.py` 用自身位置算出
  `root = <options.py>/../../../`，而 `../../../` 从运行目录起算正好回到同一个 root。
  换机器只要保持 `<root>/codes/config/inpainting` 与 `<root>/experiments/` 的相对结构即可，
  配置一行都不用改。`options.py` 现在也支持 `$VAR` 与 `~`（`osp.expandvars`），
  布局不一致时可以用环境变量覆盖。
- 训练配置默认用完整 dark val 做监控（约 600 张，一次评估约 1 小时）。
  建议先建一个 60 张的子集（配置注释里给了命令），最终结果仍用完整 val 评一次。
- `freeze_base: false` 的联合训练路径已实现但**默认关闭**。它会重新引入
  "新模块把基网带崩"的风险；只有在冻结版本确认有效之后，才值得试。
- MTSCE 的代码仍然留在 `models/modules/mtsce.py`（默认 `enabled: false`，
  不构建任何模块）。要彻底删掉说一声，需要同时改 `DenoisingNAFNet_arch.py` 的 import。
