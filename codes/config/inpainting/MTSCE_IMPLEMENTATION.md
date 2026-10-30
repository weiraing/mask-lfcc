# MTSCE 实现说明（inpainting 任务）

多尺度纹理与结构协同增强模块（Multi-scale Texture and Structure Collaborative
Enhancement，MTSCE）的代码实现、接入方式、配置项与验证结果。
对应方案文档：`Refusion 多尺度纹理与结构协同增强模块实现方案.md`。

本次改动**只涉及 `codes/config/inpainting`**（inpainting 任务），未改动扩散数学定义、
噪声调度、采样算法与监督损失；其它任务目录（deblurring / denoising-sde / ...）完全未动。

---

## 1. 文件清单

### 新增

| 文件 | 作用 |
| --- | --- |
| `codes/config/inpainting/models/modules/mtsce.py` | MTSCE 独立模块：结构分支、纹理分支、掩码感知融合、残差输出、消融开关、配置校验 |
| `codes/config/inpainting/options/train/refusion_mask_mtsce_stage1.yml` | 阶段一（只训练新模块，冻结基网） |
| `codes/config/inpainting/options/train/refusion_mask_mtsce_stage2.yml` | 阶段二（联合微调） |
| `codes/config/inpainting/options/train/refusion_mask_mtsce_scratch.yml` | 从头训练版（`pretrain_model_G: ~`，单阶段，不加载任何预训练权重） |
| `codes/config/inpainting/options/test/refusion_mask_mtsce.yml` | 实验 C：掩码通道 ON + MTSCE ON |
| `codes/config/inpainting/options/test/smoke_mtsce.yml` | 端到端 smoke（用已有权重 + 新建 MTSCE） |
| `codes/config/inpainting/options/test/smoke_mtsce_struct_only.yml` | 消融 D2（仅结构分支） |
| `codes/config/inpainting/options/test/smoke_mtsce_texture_only.yml` | 消融 D3（仅纹理分支） |
| `codes/config/inpainting/options/test/smoke_mtsce_fusion_mean.yml` | 消融 D4（简单平均融合） |
| `codes/config/inpainting/verify_mtsce.py` | 模块级自检（形状 / 掩码对齐 / 梯度 / 权重兼容 / 消融 / 参数量与耗时） |
| `codes/config/inpainting/verify_mtsce_infer.py` | 真实权重下的**确定性**推理对比（基线 vs MTSCE） |
| `codes/config/inpainting/verify_mtsce_train.py` | 训练路径 smoke（真实数据集 + 冻结 + 参数组 + 保存/加载） |

### 修改

| 文件 | 改动 |
| --- | --- |
| `models/modules/DenoisingNAFNet_arch.py` | `ConditionalNAFNet` 新增 `mtsce` 配置参数；按 `insertion_points` 构建 `ModuleDict`；forward 中按层级调用并把掩码缩放到对应分辨率；模块关闭时不构建任何子模块，前向与原网络完全一致 |
| `models/modules/__init__.py` | 导出 `MTSCE` |
| `models/base_model.py` | 新增 `load_state_dict_compatible()`；`load_network()` 支持 `allow_missing_prefixes` / `allow_unexpected_prefixes` 显式兼容加载；`torch.load` 加 `map_location="cpu"` |
| `models/denoising_model.py` | 检测 `network_G.setting.mtsce.enabled`；兼容加载（仅允许 `mtsce.*` 缺失）；打印 MTSCE 参数量；支持 `train.finetune.freeze_base` 与 `train.finetune.lr_mtsce` 参数组 |
| `test.py` | 新增 `--seed`（逐图重设随机种子，保证 A/B 可复现）；修正 Windows 下 `cv2` 与 `torchvision` 的 DLL 导入顺序 |
| `train.py` | 修复 `use_tb_logger: false` 时末尾 `tb_logger.close()` 抛 `UnboundLocalError` 的问题（改为 `tb_logger = None` + 判空） |
| `.gitignore` | 忽略 `test.py` / `train.py` 运行时生成的 `log`、`result` 软链接 |

接口变化只有一处：`ConditionalNAFNet(..., mtsce=None)`。不传该参数时行为与改动前完全一致。

---

## 2. 模块实现

### 2.1 计算流程

```
F0  = Proj(F)                        # 1x1, C -> mid
Fs  = S(F0)                          # 多尺度结构分支（并行空洞卷积）
Ft  = T(F0)                          # 局部纹理分支（深度可分离卷积）
Ff  = A(Fs, Ft, E(Ms))               # 掩码感知自适应融合
Fout= F + alpha * Proj_out(Ff)       # 残差输出
mid = max(min_hidden, round(C * hidden_ratio))
```

* **结构分支**：1×1 降维 → 3 路并行 3×3 空洞卷积（默认膨胀率 1/2/3）→ concat → 1×1 融合。
* **纹理分支**：DW3×3 → GroupNorm → PW1×1 → GELU → DW3×3 → GroupNorm → PW1×1。
  第一版没有引入傅里叶变换或注意力，避免多个机制同时进入而无法归因。
* **融合**：`[A_s, A_t] = Softmax(Conv1x1([Fs, Ft, E(Ms)]))`，逐像素满足
  `A_s + A_t = 1`，`Ff = A_s ⊙ Fs + A_t ⊙ Ft`；`E(·)` 是两层 3×3 卷积的掩码嵌入。
* **残差**：`Proj_out` 权重与偏置**零初始化**，`alpha` 初始化为 `residual_scale_init`（默认 0.1）。

### 2.2 三个关键实现选择

1. **零初始化 ⇒ 未训练的 MTSCE 是严格恒等映射。**
   因此把已有权重加载进带 MTSCE 的网络后，输出与基线**逐比特相同**（见 §4.2），
   微调起点不会因为新模块而抖动。代价是第一步只有 `Proj_out` / `alpha` 有梯度，
   之后所有上游子模块才有梯度（自检脚本已确认第 4 步后全部子模块梯度非零）。
2. **掩码一律最近邻缩放。** 二值掩码用双线性插值会产生无意义的中间值并模糊缺损边界；
   `resize_mask()` 统一处理 `[B,1,H,W]` / `[B,H,W]`、多通道掩码折叠、`mask_invert` 取反。
3. **掩码约定与数据集一致。** `1 = 缺损区`、`0 = 完整区`（`LQGTMask_dataset.py` 已如此二值化）。
   若将来换成相反约定的数据，只需在配置里设 `mask_invert: true`，不必改代码。

### 2.3 消融开关（对应方案 §13.2）

| 实验 | 配置写法 | 说明 |
| --- | --- | --- |
| D1 | `mtsce.enabled: false` | 完全等价基线，`state_dict` 键集合与基线一致 |
| D2 | `use_texture: false` | 只保留结构分支（纹理分支输出置零，参数布局不变） |
| D3 | `use_structure: false` | 只保留纹理分支 |
| D4 | `fusion: mean`（或 `add`） | 简单融合，无可学习门控 |
| D5 | `fusion: adaptive` + `use_mask_condition: true` | 完整方法 |
| 附加 | `use_mask_condition: false` | 去掉掩码条件的融合门控 |

`structure_dilations: [1]` 可退化为单尺度结构分支；`hidden_ratio` 控制参数量。

---

## 3. 网络接入

### 3.1 插入点命名

`mtsce.insertion_points` 支持：

| 名称 | 位置 | 通道（width=64, enc=[1,1,1,28], dec=[1,1,1,1]） | 分辨率 |
| --- | --- | --- | --- |
| `encoder_1` | 第 1 个 encoder 之后 | 64 | H |
| `encoder_2` | 第 2 个 encoder 之后 | 128 | H/2 |
| `encoder_3` / `encoder_4` | … | 256 / 512 | H/4 / H/8 |
| `bottleneck` | `middle_blks` 之后（方案的位置 A） | 1024 | H/16 |
| `decoder_1` | 最深一层 decoder 之后 | 512 | H/8 |
| `decoder_4` | 最高分辨率 decoder 之后（方案的位置 B） | 64 | H |

非法名称、越界索引会在构建期直接报错，并列出全部合法名称。

### 3.2 掩码传递

掩码仍由原有数据流传入（`feed_data(..., mask)` → `sde.set_mask()` →
`sde.noise_fn/score_fn` → `ConditionalNAFNet.forward(..., mask=...)`），
不通过任何通道索引去“猜”掩码位置。forward 中：

1. 输入拼接前把掩码对齐到图像尺寸（并折叠多通道掩码）；
2. 图像 padding 到 `padder_size` 的整数倍时，掩码按同样的 pad 量补齐；
3. 每个 MTSCE 内部再按自身特征图尺寸做最近邻缩放。

自检脚本用 hook 验证了 5 个插入点实际收到的掩码分辨率为 `[64, 32, 16, 32, 64]`，
且缩放后仍只含 `{0, 1}`。

### 3.3 关闭时的行为

`enabled: false` 时不会构建 `ModuleDict`，forward 中所有 `_apply_mtsce()` 直接返回原张量：
参数量、`state_dict` 键集合、输出都与改动前的网络逐比特一致（自检第 8 项）。

---

## 4. 验证结果

环境：Python 3.13 / PyTorch 2.8.0+cu128 / RTX（CUDA 可用）；`codes/config/inpainting` 下运行。

### 4.1 模块自检 `verify_mtsce.py`

```
python verify_mtsce.py --verbose --real-config      # CPU
python verify_mtsce.py --device cuda --real-config  # GPU
```

**332 条断言全部通过**（CPU 与 GPU 各跑一次），覆盖：

* 6 种非法配置被拒绝（未知字段、非法 fusion、偶数卷积核、`hidden_ratio<=0`、空插入点、两分支同时关闭）；
* 输入 32×32 / 60×60（非 2 的幂、需 padding）输出形状一致，`mask=None` 与低分辨率掩码均可运行；
* 零初始化恒等性：`max |baseline - mtsce| = 0.000e+00`（逐比特相同）；
* 掩码缩放：64→{64,32,16,8,7×9} 均为最近邻、保持二值、`mask_invert` 正确取反；
* 梯度：31 个 MTSCE 张量均有有限梯度；4 步之后**每个**子模块梯度非零（无“死分支”）；
* 权重兼容：旧权重 668 个张量全部恢复、31 个 MTSCE 张量保持新初始化；
  缺少基网张量 / 形状不匹配 / 多余未知张量**都会报错**（不是宽松加载）；
* 完整 `state_dict` 保存-加载后输出逐比特一致；
* 5 个插入点均可单独使用，4 个非法名称被拒绝；
* 7 种消融组合均可构建并前向。

真实配置（width=64）参数量：

| 配置 | MTSCE 参数 | 总参数 | 增幅 |
| --- | --- | --- | --- |
| 基线 | 0 | 76,608,963 | — |
| `bottleneck`, `hidden_ratio=0.5` | 9,718,443 | 86,327,406 | +12.69% |
| `bottleneck`, `hidden_ratio=0.25` | 2,697,387 | 79,306,350 | +3.52% |
| `bottleneck` + `decoder_4`, `hr=0.5` | 9,758,166 | 86,367,129 | +12.74% |

去噪网络单次前向耗时（FP32，batch=1）：

| 分辨率 | 基线 | +MTSCE（bottleneck） | 增幅 |
| --- | --- | --- | --- |
| 256×256 | 65.5 ms | 68.7 ms | +4.9% |
| 512×512 | 1008.8 ms | 1102.7 ms | +9.3% |

MTSCE 自身在 bottleneck（1024 通道）上的耗时：8×8 与 32×32 约 3.2 ms，64×64 约 8.2 ms。

### 4.2 真实权重 + 确定性推理对比 `verify_mtsce_infer.py`

用 `lastest_EMA.pth`（掩码通道模型，无 MTSCE 参数）同时构建基线网络与 MTSCE 网络，
对 `val_smoke` 的 3 张图各跑 100 步反向 SDE，并在每次采样前重设同一个随机种子：

```
compatible load: 668 base tensors restored, 31 MTSCE tensors freshly initialised
params: base 76,608,963 + MTSCE 9,718,443 = 86,327,406

0001.png (hole 19.8%) | bit-identical: True (max |diff| 0.0e+00) | PSNR 35.1614 / 35.1614 dB | 6.30s / 6.90s
0002.png (hole 15.5%) | bit-identical: True (max |diff| 0.0e+00) | PSNR 37.4479 / 37.4479 dB | 6.18s / 6.59s
0003.png (hole 10.3%) | bit-identical: True (max |diff| 0.0e+00) | PSNR 39.5123 / 39.5123 dB | 6.78s / 6.39s

RESULT: PASS
```

结论：**已有权重可以原样加载；开启 MTSCE 后推理路径确实执行了新模块，但由于零初始化，
输出与基线逐比特一致。** 这同时证明训练与推理使用的是同一套结构（不存在“训练开、推理关”）。

### 4.3 训练路径 smoke `verify_mtsce_train.py`

```
INFO: MTSCE enabled | insertion_points=['bottleneck'] | mtsce params=9,718,443 | base params=76,608,963 | total=86,327,406 (+12.69%)
INFO: MTSCE stage-1 warm-up: 668 base parameter tensor(s) frozen; only MTSCE parameters are trainable.
INFO: Parameter groups: base lr=4e-05, MTSCE lr=0.0001
--- 4 training iterations ---
  iter 1 | loss 3.344569e-04 | lr 1.000e-04
  iter 2 | loss 2.035652e-04 | lr 1.000e-04
  iter 3 | loss 2.127574e-04 | lr 1.000e-04
  iter 4 | loss 1.259186e-04 | lr 1.000e-04
MTSCE tensors updated: 31/31
frozen base tensors unchanged: OK
checkpoint reload: strict OK, 31 MTSCE tensors
TRAINING SMOKE TEST PASSED
```

说明：真实 `LQGTMask` 数据集 + 真实 SDE 训练目标 + Lion + EMA 全部跑通；
冻结的基网参数在 4 步后逐比特未变，MTSCE 参数已更新，含 MTSCE 的检查点可被
`strict=True` 独立加载。

另外用**真实训练入口**跑了一遍完整流程（临时小配置：niter=3、val_freq=2、save=2、
batch=2、crop=64）：

```
python train.py -opt=options/train/<临时小配置>.yml
...
<epoch:  0, iter:       2, psnr: 36.599461
EXIT=0
models/2_G.pth  models/lastest_EMA.pth  models/latest_G.pth   # 各约 345MB
```

即：训练 → 验证 → 存盘 → 正常退出都通过。该过程还暴露并修复了 `train.py` 的一个旧 bug：
`use_tb_logger: false` 时末尾 `tb_logger.close()` 会抛 `UnboundLocalError`。

### 4.4 `test.py` 端到端 A/B（固定种子）

```
python test.py -opt=options/test/smoke_base.yml  --seed 0
python test.py -opt=options/test/smoke_mtsce.yml --seed 0
```

| 指标 | 基线（掩码通道，MTSCE OFF） | MTSCE ON（零初始化） |
| --- | --- | --- |
| PSNR | 38.462759 dB | 38.462759 dB |
| SSIM | 0.961221 | 0.961221 |
| LPIPS | 0.023238 | 0.023238 |
| 单图耗时 | 12.01 s | 12.17 s |

三张图逐图指标完全相同（PSNR 36.858521 / 37.358741 / 41.171014 dB），推理开销约 +1.4%。

> ⚠️ **关于可复现性**：反向 SDE 每步都会注入高斯噪声，`test.py` 原本没有设种子。
> 实测同一配置两次运行的 PSNR 可以相差 **0.3–0.7 dB**（例如同一基线在未固定种子时
> 得到 37.80 / 38.46 dB）。因此做 B/C 对比时**必须**使用 `--seed`，否则观测到的差异
> 很可能只是采样噪声，而不是 MTSCE 的作用。`--seed` 现在按图像重新设种，
> 与网络构建消耗的随机数无关，A/B 结果可直接对齐。

---

## 5. 使用方式

```bash
cd codes/config/inpainting

# 0) 自检（建议每次改配置后跑一遍）
python verify_mtsce.py --verbose --real-config
python verify_mtsce_infer.py
python verify_mtsce_train.py --iters 4

# 1) 阶段一：冻结基网，只预热 MTSCE（配置里已指向已有 30000_G.pth）
python train.py -opt=options/train/refusion_mask_mtsce_stage1.yml

# 2) 阶段二：联合微调（pretrain_model_G 已预填为阶段一的 lastest_EMA.pth）
python train.py -opt=options/train/refusion_mask_mtsce_stage2.yml

# 3) 测试（B/C 必须用相同 --seed）
python test.py -opt=options/test/refusion_mask.yml       --seed 0   # B
python test.py -opt=options/test/refusion_mask_mtsce.yml --seed 0   # C

# 4) 快速回归（3 张图，约 1 分钟）
python test.py -opt=options/test/smoke_mtsce.yml --seed 0
```

**从头训练（不加载任何预训练权重）**：

```bash
python train.py -opt=options/train/refusion_mask_mtsce_scratch.yml
```

三个训练配置的差别只有“从哪儿开始、谁在训练”：

| 配置 | `pretrain_model_G` | `freeze_base` | 适用场景 |
| --- | --- | --- | --- |
| `..._scratch.yml` | `~`（随机初始化） | `false` | 完全自己训，不使用已有权重 |
| `..._stage1.yml` | 已有 `30000_G.pth` | `true` | 在已收敛基网上预热新模块 |
| `..._stage2.yml` | 阶段一的 `lastest_EMA.pth` | `false` | 联合微调出最终模型 |

> `pretrain_model_G` 只加载**权重**（优化器、学习率、迭代数都是新的）；
> 中断后继续训练要用 `resume_state: experiments/inpainting/<name>/training_state/<iter>.state`，
> 此时 `options.check_resume` 会自动把 `pretrain_model_G` 改写成 `models/<iter>_G.pth`，
> 不需要也不应该再手填。
>
> 另外：从头训练时 `freeze_base` 必须是 `false`——冻结一个随机初始化的基网、只训 MTSCE，
> 等于让新模块去拟合噪声，没有意义。

消融实验：复制 `options/train/refusion_mask_mtsce_stage2.yml`，只改 `mtsce` 下的一个开关
（`use_structure` / `use_texture` / `fusion` / `use_mask_condition`），改 `name`，
然后按同样的 stage1 → stage2 流程训练，并用对应测试配置评测。
`options/test/smoke_mtsce_struct_only.yml` 等三个文件给出了 D2/D3/D4 的测试侧写法。
若 B/C 都要从头训（更严格的对比），先复制 `refusion_mask.yml` 把 `pretrain_model_G` 设为 `~`
训出 B，再把 stage1 的 `pretrain_model_G` 指向它。

---

## 6. 已知限制与后续建议

1. **本次只验证“接入正确”，没有验证“效果更好”。** 零初始化意味着未微调的 MTSCE
   与基线完全等价；是否提升 PSNR/SSIM/LPIPS 必须由阶段一/二训练 + 固定种子的 A/B 实验回答，
   在拿到实验结果前不应声称任何指标提升（方案 §15.3 的要求）。
2. **默认只插在 bottleneck（位置 A）。** 方案建议先单点验证，再试位置 B
   （`insertion_points: [bottleneck, decoder_4]`，额外参数几乎不变，因为 decoder_4 只有 64 通道）。
3. **参数增幅 12.7%（`hidden_ratio=0.5`）。** 若显存或训练成本敏感，可先用
   `hidden_ratio: 0.25`（+3.5%）；显存开销主要来自 bottleneck 的 512 通道并行空洞卷积。
4. **MTSCE 只接入了 `ConditionalNAFNet`**（inpainting 实际使用的去噪网络），
   `ConditionalUNet` 未改动；若以后换网络，需要在新网络里提供同样的 `_apply_mtsce` 接入点。
5. **指标应按缺损区域单独统计。** `test.py` 目前只报告全图 PSNR/SSIM/LPIPS，
   全图指标会被大面积完整区域主导；建议后续补一版仅统计 `mask == 1` 区域（及边界带）的指标。
6. **数据集划分要按文物/原图划分**，避免同一件纺织品的相邻裁剪块同时进入训练和测试（方案 §13.3）。
7. **环境相关**：本机 `torchvision` 与 `opencv` 的 DLL 加载顺序冲突，`test.py` 里已把
   `import lpips` 放到 `import utils`（内部导入 cv2）之后；如果重装环境后出现
   `ImportError: DLL load failed while importing cv2`，按同样顺序调整即可。

---

## 7. 配置参数逐项说明

### 7.1 两个训练配置的分工

两者**网络结构完全相同**（同一 `insertion_points`、同一 `hidden_ratio`、同一开关），
差别只在“谁在训练、用什么学习率、跑多久”：

| 项目 | `refusion_mask_mtsce_stage1.yml` | `refusion_mask_mtsce_stage2.yml` |
| --- | --- | --- |
| `name` | `refusion_mask_mtsce_s1` | `refusion_mask_mtsce` |
| 目的 | **预热新模块**：验证 MTSCE 能稳定接入、能学出有意义的残差 | **联合微调**：让基网与新模块协同，追求最终修复质量 |
| 初始化来源 | 已训练好的掩码通道权重 `30000_G.pth`（基网全有、MTSCE 全缺，缺失部分保持零初始化） | 阶段一产出的检查点（基网 + 已预热的 MTSCE，`strict` 全量加载）；`pretrain_model_G` 已预填为 `.../refusion_mask_mtsce_s1/models/lastest_EMA.pth` |
| `train.finetune.freeze_base` | `true`（668 个基网张量冻结，只有 31 个 MTSCE 张量可训练） | `false`（699 个张量全部可训练） |
| `train.finetune.lr_mtsce` | `1e-4`（比基网高一个量级，让新模块快速起步） | `4e-5`（与 `lr_G` 相同，稳定微调） |
| `train.niter` | 20000（短程，够看损失曲线是否正常；5k~20k 通常已足够，可随时中断取最近检查点） | 30000（与 B 基线 `30000_G.pth` 对齐，便于公平对比；想训更久直接改大） |
| 训练中验证集 | `val_smoke`（3 张，约 40 秒/次） | `val_smoke`（3 张，约 40 秒/次）；全量 val 604 张、T=100，一次约 2 小时，不适合训练中跑，最终指标用 `test.py` 离线评测 |
| 保存频率 | 每 5000 iter（每次约 690MB，共约 2.8GB） | 每 5000 iter（每次约 690MB，共约 4.1GB） |
| 产出用途 | 只是中间产物；用来判断“接入是否正常、梯度是否健康” | 论文用的 C 实验模型；与 B（`refusion_mask`）对比 |
| 训练命令 | `python train.py -opt=options/train/refusion_mask_mtsce_stage1.yml` | `python train.py -opt=options/train/refusion_mask_mtsce_stage2.yml`（路径已预填，阶段一跑完即可直接执行） |

为什么必须分两步：`Proj_out` 是零初始化，**第一步只有 `Proj_out` 的权重/偏置有梯度**，
`alpha` 与上游子模块要从第二步才开始更新（自检脚本实测：1 步后 2/31 个张量变化，
4 步后 31/31 全部有梯度）。
若一上来就联合训练，新模块的随机特征会以较大梯度冲击已经收敛的基网；
先用高学习率把 MTSCE 单独喂到“输出合理”，再用小学习率联合微调，训练更稳。

> 阶段二必须沿用阶段一的 `mtsce` 配置：结构不一致（比如改了 `insertion_points` 或
> `hidden_ratio`）时，形状不匹配会被兼容加载直接报错，而不会被静默跳过。

### 7.2 `network_G.setting.mtsce` 参数

| 参数 | 默认值 | 含义 | 调参提示 |
| --- | --- | --- | --- |
| `enabled` | `false` | 总开关。关闭时不构建任何子模块，`state_dict` 与输出和基线逐比特一致（D1） | 做 B 组实验/回归时设 `false` |
| `insertion_points` | `[bottleneck]` | 插入位置列表，可选 `encoder_1..4`、`bottleneck`、`decoder_1..4` | 位置 A = `[bottleneck]`；位置 B = `[bottleneck, decoder_4]`；名字写错会在构建期报错并列出合法值 |
| `hidden_ratio` | `0.5` | 两个分支的宽度 = `round(C × ratio)`，`C` 是该层通道数 | bottleneck 下 `0.5` → 9.72M(+12.69%)，`0.25` → 2.70M(+3.52%) |
| `min_hidden` | `8` | 分支宽度下限，避免插在浅层（64 通道）时过窄 | 一般不用改 |
| `structure_dilations` | `[1, 2, 3]` | 结构分支的并行空洞率，决定感受野数量 | 织物纹理很细时别加大（≥4 易出现栅格效应）；`[1]` 退化为单尺度 |
| `texture_kernel_size` | `3` | 纹理分支深度可分离卷积的核大小（必须正奇数） | 想更关注细节可试 `5`，代价是参数与耗时上升 |
| `use_structure` | `true` | 结构分支开关 | 消融 D3 时设 `false`（只剩纹理分支） |
| `use_texture` | `true` | 纹理分支开关 | 消融 D2 时设 `false`（只剩结构分支） |
| `texture_norm` | `true` | 纹理分支是否使用 GroupNorm | 若发现颜色/亮度被压平可设 `false` |
| `fusion` | `adaptive` | 融合方式：`adaptive`（掩码感知门控，D5）/ `mean`（简单平均，D4）/ `add`（直接相加） | `mean`、`add` 没有可学习门控参数，用于验证“自适应融合”本身的价值 |
| `use_mask_condition` | `true` | 融合门控是否接收掩码特征 `E(Ms)` | 设 `false` 可做“无掩码先验的融合”对照 |
| `mask_feat_ch` | `8` | 掩码嵌入 `E(·)` 的通道数 | 8 已足够（掩码是二值的），一般不用改 |
| `mask_invert` | `false` | 反转掩码极性。数据集约定 `1 = 缺损`；若换成 `0 = 缺损` 的数据设 `true` | 无需改数据加载代码 |
| `residual_scale_init` | `0.1` | 残差缩放系数 `alpha` 的初值（可学习） | 想让新模块初期更“保守”可降到 `0.05`；`Proj_out` 本身零初始化，所以 `0` 也不会完全锁死 |

### 7.3 `train.finetune` 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `freeze_base` | `false` | `true` 时冻结除 `mtsce.*` 以外的全部参数（阶段一用） |
| `lr_mtsce` | 未设置（回落到 `lr_G`） | 单独给 MTSCE 参数组的学习率；与 `lr_G` 不同时会自动建两个参数组 |

`path.strict_load` 保持 `true`：兼容加载只放行 `mtsce.*` 的缺失/多余，其它任何不匹配都会报错。

