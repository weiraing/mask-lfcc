# Refusion 外部监督式修复校正模块：AI 编程实施方案

> **项目目标**：在现有 Refusion 图像修复项目上，保留已有的掩码通道改进和已训练权重，不改动基础去噪网络结构，新增一个可独立训练、可通过配置启用/关闭的外部校正模块，以改善成对数据集上的图像修复质量。
>
> **任务类型**：Image Inpainting（图像缺损修复）
>
> **方案名称**：监督式扩散修复校正模块（Supervised Diffusion Restoration Corrector，SDRC）
>
> **重要说明**：基础 Refusion 权重不重新训练；但 SDRC 校正网络需要单独训练。该模块是否能改善结果必须通过实验验证，不应预先声称必然提升。

---

## 0. 给 AI 编程工具的执行指令

你正在维护一个已有的 Refusion 图像修复代码库。请先阅读并理解项目，再进行最小侵入式修改。**不要假设代码结构、模型预测参数化、潜变量缩放方式或采样器公式；必须从仓库源码确认。**

### 必须遵守

1. 先审查代码并输出实现计划，随后再修改代码。
2. 保留现有基础模型、已有权重、原始采样路径和现有掩码通道改进。
3. 不要重写或替换基础去噪网络，不要擅自改变原有模型的输入通道数。
4. 新增模块必须是可独立创建、训练、保存、加载、启用和关闭的外部组件。
5. 默认只在**潜在空间**操作，避免每个扩散步骤都进行 VAE 解码和重新编码；但必须先确认 Refusion 的实际数据流。
6. 不允许直接将校正后的干净潜变量预测当成扩散状态 `z_t`。校正结果必须通过项目原有的预测参数化和采样公式继续更新。
7. 不允许在采样时使用真实干净目标图像（ground truth）。
8. 不允许让校正模块改变未缺损区域，除非现有算法明确要求；优先通过掩码约束将修正限制在缺损区域，并为边界过渡保留可配置策略。
9. 不能只凭 PSNR 单一指标判断效果。必须保留原始基线，报告缺损区域与全图指标、运行耗时。
10. 任何无法从源码确认的地方，先记录为待确认项，不要编造实现细节。
11. 不要删除或覆盖已有模型权重、数据、日志和配置。
12. 新功能默认关闭，确保不启用 SDRC 时，原始推理路径结果和行为尽可能保持一致。

---

## 1. 开始编码前：代码库审查

请先检查仓库，定位并阅读以下组件：

- Refusion 主模型及其 `forward` 方法。
- 扩散训练损失计算代码。
- 采样器/反向扩散循环。
- 模型预测参数化：模型输出的是 `epsilon`、`x0`、`v`，还是其他形式。
- 潜变量生成、VAE 编码/解码及缩放系数。
- 图像修复条件输入和掩码通道的拼接位置。
- 训练数据读取方式，以及损坏图像、干净目标、掩码的张量约定。
- checkpoint 保存与加载格式。
- 配置文件及命令行参数系统。
- 现有 PSNR、SSIM、LPIPS 等评估代码。

### 请先产出审查报告

创建 `docs/sdrc_code_audit.md`，至少记录：

1. 关键文件路径及职责。
2. 训练时一个 batch 的实际数据流。
3. 采样时每一步的实际数据流。
4. 当前模型输出的语义和预测参数化。
5. 潜变量缩放和归一化规则。
6. 掩码约定：`1` 表示缺损还是有效区域，掩码形状及广播规则。
7. 现有采样器如何从模型输出更新到下一步状态。
8. 校正模块最合适的插入点。
9. 为兼容现有 checkpoint，需要避免哪些改动。

**完成审查报告后再实现。** 如果发现该项目并非在潜在空间扩散，或者预测变量并非 `x0`，请根据实际实现调整本方案，不要强行套用下面的示例公式。

---

## 2. 模块设计

### 2.1 整体流程

原有推理流程：

```text
受损图像 + 掩码
      │
      ▼
Refusion 原有条件输入处理
      │
      ▼
基础扩散模型（已有权重，冻结）
      │
      ▼
原有采样器更新状态
      │
      ▼
最终修复结果
```

启用 SDRC 后：

```text
受损图像 + 掩码
      │
      ▼
Refusion 原有条件输入处理
      │
      ▼
基础扩散模型（已有权重，冻结）
      │
      ▼
按原始模型参数化计算干净样本/潜变量预测
      │
      ▼
SDRC 外部校正模块 ◄── 条件信息：受损图像潜变量、掩码、时间步
      │
      ▼
将校正后的预测送回原有采样器更新公式
      │
      ▼
进入下一扩散步骤
      │
      └────────循环直到采样结束
      ▼
VAE 解码（如项目原本使用 VAE）
      │
      ▼
最终修复结果
```

### 2.2 设计原则

SDRC 不替换基础扩散模型，而是对每一步由基础模型推导出的干净样本预测进行小幅修正。它学习的目标是减少中间预测误差，并在修复任务中利用条件图像和掩码辅助校正。

建议采用**残差校正**：

\[
\tilde{z}_0^{(t)}=\hat{z}_0^{(t)}+\Delta z_0^{(t)}
\]

其中：

- \(z_t\)：当前扩散状态。
- \(\hat{z}_0^{(t)}\)：根据 Refusion 当前预测和项目原有参数化公式推导出的干净潜变量预测。
- \(\Delta z_0^{(t)}\)：SDRC 预测的残差。
- \(\tilde{z}_0^{(t)}\)：校正后的干净潜变量预测。

如果代码中的模型预测不是噪声参数化，仍应使用项目现有函数推导 \(\hat z_0^{(t)}\)，不可自行套用特定公式。

### 2.3 SDRC 输入与输出

建议接口：

```python
corrector(
    x0_pred,          # 当前模型推导出的干净潜变量预测
    condition_latent, # 受损图像/条件图像的潜变量表示
    mask_latent,      # 与潜变量空间对齐的掩码
    timestep,         # 当前扩散时间步或经归一化的时间步嵌入
) -> residual
```

要求：

- `residual` 与 `x0_pred` 的形状一致。
- 输入张量的设备、数据类型和 batch 维度一致。
- 时间步嵌入方式与项目设备和采样时间步定义兼容。
- 模块应支持 batch 推理。
- 不允许将 GT 输入推理阶段的 `corrector`。
- 对掩码的缩放/下采样必须与项目现有掩码语义一致；不能直接用双线性插值处理离散掩码后又假定其仍为二值掩码。

---

## 3. 网络结构建议

先采用轻量级、易于调试的残差网络，不要第一版就引入复杂注意力结构。

建议结构：

1. 将 `x0_pred`、`condition_latent`、`mask_latent` 按通道拼接。
2. 将时间步编码为 embedding，并通过加法或小型投影注入特征层。
3. 使用若干卷积残差块提取局部纹理和结构信息。
4. 输出与潜变量通道数一致的残差。
5. 输出层初始化为零或接近零，使模块初始行为接近原始 Refusion。
6. 在输出端使用掩码约束与可配置残差幅度限制。

结构示意：

```text
x0_pred ───────────────┐
condition_latent ──────┼─> Channel Concat -> Conv Stem
mask_latent ───────────┘                       │
timestep -> Time Embedding -> Feature Modulation
                                              │
                                    Residual Conv Blocks
                                              │
                                         Output Conv
                                              │
                                      Predicted Residual
                                              │
                               Mask / Scale Constraint
                                              │
                                   Corrected x0 Prediction
```

### 3.1 初版实现要求

- 先使用普通卷积残差块。
- 通道宽度、残差块数量、时间步 embedding 维度均配置化。
- 默认小模型，例如 `hidden_channels=64`、`num_res_blocks=4`；需依据项目实际显存和潜变量形状调整。
- 最后一层卷积权重与偏置初始化为零。
- 不要把 SDRC 层添加进基础模型 checkpoint 的参数命名空间。
- 单独保存 SDRC checkpoint，避免影响原有模型加载。

---

## 4. 训练方案

### 4.1 训练目标

基础 Refusion 模型保持冻结，只训练 SDRC。使用项目已有的成对数据：

- `damaged_image`：受损输入图像。
- `gt_image`：干净目标图像。
- `mask`：缺损区域掩码。

具体数据字段名称以项目为准。

训练时，先使用冻结的 Refusion 模型和项目原有扩散过程，生成或提取中间状态与干净潜变量预测，再由 SDRC 学习预测残差。**训练输入分布应尽可能贴近真实采样时的中间预测分布**，避免只用 GT 加随机噪声构造理想输入而导致训练和推理不一致。

### 4.2 重要实现步骤

1. 正常加载现有 Refusion checkpoint。
2. 将基础模型设为 `eval()` 并冻结全部参数。
3. 保持项目原有 VAE、潜变量缩放、条件编码和掩码处理逻辑。
4. 通过现有采样/预测代码收集训练用的中间状态与 \(\hat z_0^{(t)}\)。
5. 用干净目标图像编码得到目标潜变量 \(z_0^{gt}\)。
6. 将 \(\hat z_0^{(t)}\)、条件潜变量、掩码、时间步送入 SDRC。
7. 预测残差并计算校正损失。
8. 仅更新 SDRC 参数。
9. 验证集上评估，保存最优 SDRC checkpoint。

**实现注意**：如果现有采样器不易直接提供中间预测，请优先抽取/复用现有模型预测函数，不要复制一份与原实现逐渐偏离的扩散公式。可以先通过 `torch.no_grad()` 离线缓存中间预测，减少校正模块训练时反复运行基础模型的开销；缓存内容需记录模型 checkpoint、采样器、时间步、数据版本等信息。

### 4.3 监督损失

基础残差目标：

\[
\Delta z_{\mathrm{target}}=z_0^{gt}-\hat z_0^{(t)}
\]

\[
\mathcal{L}_{\mathrm{res}}=
\left\|\Delta z_{\mathrm{pred}}-\Delta z_{\mathrm{target}}\right\|_2^2
\]

其中：

\[
\Delta z_{\mathrm{pred}}=
C_\phi(\hat z_0^{(t)},z_{\mathrm{cond}},M,t)
\]

这只是一个**可实施的初始损失**，需确认目标潜变量与预测潜变量采用完全相同的编码、缩放和归一化约定。若预测样本来自不同噪声时间步或不同采样器设置，应在训练数据构造中明确记录并对齐。

可选增加校正后潜变量重建损失：

\[
\mathcal{L}_{\mathrm{corr}}=
\left\|\tilde z_0^{(t)}-z_0^{gt}\right\|_2^2
\]

组合损失：

\[
\mathcal{L}_{\mathrm{total}}=
\lambda_{\mathrm{res}}\mathcal{L}_{\mathrm{res}}
+\lambda_{\mathrm{corr}}\mathcal{L}_{\mathrm{corr}}
\]

初版可先只使用残差损失，确认训练稳定后再评估组合损失。权重必须配置化。

### 4.4 掩码约束

约定必须先从源码确认：以下示例假设 `M=1` 表示缺损区域。

将掩码映射到潜变量空间，得到 \(M_z\)。如果 `M=1` 表示缺损，建议先测试：

\[
\Delta z_{\mathrm{masked}}=M_z\odot\Delta z_{\mathrm{pred}}
\]

\[
\tilde z_0^{(t)}=\hat z_0^{(t)}+
\alpha(t)\Delta z_{\mathrm{masked}}
\]

其中 \(\alpha(t)\) 是可配置的残差缩放系数。第一版可用常数 `corrector_scale`；确认效果后再尝试随时间步变化的缩放函数。

注意：潜变量中的一个位置通常对应图像中的一片区域，潜变量掩码可能存在边界映射误差。请比较最近邻掩码、面积池化/最大池化掩码等合理映射方式，并在文档中记录选择依据。不要未经验证就认为潜变量掩码能精确对应像素级缺损区域。

### 4.5 数据构造策略

建议分阶段实现：

**阶段 A：最小可运行版本**
- 从现有模型实际推理流程中抽取若干时间步的中间预测。
- 以相同图像的 GT 潜变量作为监督目标。
- 先在小子集上验证前向、损失下降、checkpoint 保存/加载。

**阶段 B：贴近推理分布**
- 使用与真实推理一致的采样器和时间步分布收集预测。
- 训练集覆盖多个时间步，而不是只训练最后一步。
- 记录中间预测对应的时间步和采样配置。
- 如果显存允许，在线生成预测；否则离线缓存。

**阶段 C：验证泛化**
- 在独立验证集评估。
- 关注不同缺损比例、掩码形状和纹样复杂度下的效果。
- 不允许训练集与验证集图像重复。

---

## 5. 采样阶段集成

### 5.1 正确插入位置

建议把 SDRC 放在**基础模型已输出当前预测、采样器即将使用干净样本预测更新状态**的位置。

伪代码：

```python
for timestep in sampler.timesteps:
    model_output = base_model(
        current_state,
        timestep,
        original_condition_inputs,
    )

    x0_pred = original_parameterization_to_x0(
        current_state,
        model_output,
        timestep,
        original_schedule,
    )

    if corrector_enabled and should_apply_corrector(timestep):
        residual = corrector(
            x0_pred=x0_pred,
            condition_latent=condition_latent,
            mask_latent=mask_latent,
            timestep=timestep,
        )
        x0_pred = apply_residual_constraints(
            x0_pred,
            residual,
            mask_latent,
            timestep,
        )

    # 关键：使用项目原有采样器公式继续计算下一状态。
    current_state = original_sampler_update(
        current_state=current_state,
        model_output=model_output,
        x0_pred=x0_pred,
        timestep=timestep,
        # 具体参数按源码接口传递
    )
```

上面是流程示意，不是可直接复制的代码。必须根据仓库的实际采样器 API 实现。若原采样器只接收 `model_output`，需要谨慎修改其最小接口或封装一个兼容适配层，确保使用校正后的预测时数学关系正确。

**禁止做法**：校正 `x0_pred` 后却仍使用未校正的 `model_output` 更新状态，导致校正模块事实上没有影响采样；也禁止不经推导就把 `x0_pred` 直接赋值给 `current_state`。

### 5.2 调用频率

通过配置支持：

- `every_step`：每一步都校正。
- `late_steps`：只在后半段/靠近结束阶段校正。
- `selected_steps`：只在指定时间步校正。
- `disabled`：关闭校正，原始基线。

第一版优先实现 `every_step` 和 `disabled`，并确保时间步判断基于项目实际 timestep，而非假定循环索引与扩散时间步相同。完成主实验后，再增加部分时间步消融。

### 5.3 残差幅度和数值稳定性

配置建议：

```yaml
sdrc:
  enabled: false
  checkpoint: null
  apply_mode: "every_step"
  corrector_scale: 0.1
  max_residual_norm: null
```

- `corrector_scale` 是起始实验参数，不是最佳值承诺。
- 可以提供可选的残差范数裁剪，但默认关闭或采用保守设置，避免掩盖训练问题。
- 若启用掩码约束，需先统一掩码语义。
- 对非有限值（NaN/Inf）应输出明确错误或告警，不应静默替换成任意值。

---

## 6. 建议的代码组织

请尽可能按项目现有风格组织文件，不必为了符合下列路径而大规模重构：

```text
project/
├── models/
│   ├── ...                         # 原有模型，不重写
│   └── sdrc.py                     # 新增 SDRC 网络
├── trainers/
│   └── train_sdrc.py               # 新增独立训练入口
├── samplers/
│   └── ...                         # 原有采样器，最小侵入式集成
├── configs/
│   └── sdrc.yaml                   # 新增配置
├── scripts/
│   ├── train_sdrc.sh               # 可选
│   └── eval_sdrc.sh                # 可选
├── tests/
│   ├── test_sdrc_shapes.py
│   ├── test_sdrc_checkpoint.py
│   └── test_sdrc_sampling.py
└── docs/
    └── sdrc_code_audit.md
```

若仓库已有统一的 `models`、`engine`、`configs` 或 `scripts` 结构，遵循已有布局，不要平行造一套框架。

---

## 7. 配置项

至少支持以下配置，并为每项提供默认值与注释：

```yaml
sdrc:
  enabled: false
  checkpoint: null

  # 网络结构
  hidden_channels: 64
  num_res_blocks: 4
  time_embedding_dim: 128

  # 训练
  learning_rate: 0.0001
  weight_decay: 0.00001
  batch_size: 由现有训练配置决定
  num_epochs: 由实验预算决定
  lambda_res: 1.0
  lambda_corr: 0.0

  # 推理
  apply_mode: "every_step"
  corrector_scale: 0.1
  max_residual_norm: null
  apply_start_timestep: null
  apply_end_timestep: null

  # 掩码与缓存
  mask_interpolation: "nearest"
  cache_intermediate_predictions: false
  cache_dir: null
```

以上数值只是合理的初始配置示例，应根据项目的配置风格、潜变量维度、训练稳定性和显存进行调整。不要把 YAML 中的示例值硬编码进程序。

---

## 8. 命令行接口

根据项目已有入口风格实现，建议至少提供以下能力：

```bash
# 训练独立校正模块
python trainers/train_sdrc.py --config configs/sdrc.yaml

# 原始模型推理：关闭 SDRC
python <existing_inference_entry>.py --sdrc-enabled false

# 加载校正器推理
python <existing_inference_entry>.py \
  --sdrc-enabled true \
  --sdrc-checkpoint <path-to-sdrc-checkpoint>

# 评估原始基线
python <existing_eval_entry>.py --sdrc-enabled false

# 评估校正方法
python <existing_eval_entry>.py \
  --sdrc-enabled true \
  --sdrc-checkpoint <path-to-sdrc-checkpoint>
```

这些命令中的 `<existing_...>` 是占位符，必须替换为仓库真实入口。不要新增与原项目功能重复的入口，优先扩展现有命令行参数。

---

## 9. Checkpoint 与可复现性

SDRC 应独立保存，例如：

```python
{
    "model": corrector.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_metric": best_val_metric,
    "config": config,
    "base_model_checkpoint": base_checkpoint_identifier,
    "metadata": {
        "latent_scale": "...",
        "prediction_parameterization": "...",
        "sampler": "...",
        "mask_convention": "...",
        "dataset_version": "..."
    }
}
```

实际字段应按项目规范实现。不要将示意字符串直接作为有效元数据。

加载校正器时：
- 检查权重文件存在。
- 检查网络结构配置是否匹配。
- 检查基础模型 checkpoint 标识是否与训练时一致；不匹配时至少告警。
- 不得覆盖基础模型权重。
- 兼容缺少 SDRC checkpoint 的情形：若 `enabled=false`，不应尝试加载 SDRC。

---

## 10. 测试要求

### 10.1 单元测试

1. **形状测试**：输入不同 batch size 时，残差形状与 `x0_pred` 一致。
2. **设备测试**：CPU 至少可运行；CUDA 可用时验证 GPU。
3. **初始化测试**：零初始化输出层时，初始残差应为零或数值接近零。
4. **checkpoint 测试**：保存后重新加载，输出一致。
5. **掩码测试**：验证掩码通道、下采样方式、广播维度与约定一致。
6. **时间步测试**：单个时间步、batch 时间步和不同采样时间步格式均按预期工作。
7. **冻结测试**：训练 SDRC 时基础模型参数不产生梯度，优化器只持有 SDRC 参数。
8. **原路径回归测试**：关闭 SDRC 时，推理路径与原代码保持一致。确定性采样下应尽量做到输出一致；若原项目存在非确定性，应明确记录容差与原因。
9. **数据泄漏测试**：确认 GT 只用于训练/评估，不传入推理校正器。

### 10.2 集成测试

使用少量样本完成以下流程：

1. 加载基础模型 checkpoint。
2. 运行原始采样并保存结果。
3. 初始化 SDRC 并运行采样。
4. 验证输出图像尺寸、范围、数据类型正常。
5. 验证校正模块确实被调用，且输出会影响采样结果。
6. 关闭 SDRC 后重复推理，确认恢复原始路径。
7. 训练 SDRC 一个小 epoch，确认 loss 可计算、梯度更新、checkpoint 可保存。
8. 检查无 NaN/Inf，检查显存和耗时。

---

## 11. 实验设计

### 11.1 必做对比

| 实验编号 | 方法 | SDRC 训练 | 采样时校正 | 目的 |
|---|---|---:|---:|---|
| A | 原始 Refusion + 已有掩码通道改进 | 否 | 否 | 基线 |
| B | A + SDRC | 是 | 每一步 | 验证模块有效性 |
| C | A + SDRC | 是 | 仅部分时间步 | 研究调用频率 |
| D | A + SDRC，但 `corrector_scale=0` | 是 | 是但无有效残差 | 排查实现是否引入其他变量 |
| E | A + SDRC，不施加缺损掩码约束（可选） | 是 | 每一步 | 评估掩码约束作用 |

至少完成 A、B；有时间再做 C、D、E。比较时使用相同测试集、相同基础权重、相同采样器设置、相同随机种子（若采样器支持）。

### 11.2 评价指标

建议报告：

- PSNR。
- SSIM。
- LPIPS。
- 缺损区域 PSNR/SSIM（若评估实现可靠）。
- 完整推理时间与单张平均耗时。
- SDRC 参数量与额外显存。
- 古代纺织品局部纹样、经纬纹理、边缘连续性和重复图案一致性的可视化比较。

必须明确 PSNR、SSIM、LPIPS 是全图指标还是仅缺损区域指标；不同方法的统计范围要保持一致。

### 11.3 结果记录表

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 缺损区域指标 | 单张耗时 | 备注 |
|---|---:|---:|---:|---|---:|---|
| 基线 A | 待测 | 待测 | 待测 | 待测 | 待测 | 原始推理 |
| SDRC 每步 B | 待测 | 待测 | 待测 | 待测 | 待测 | 主实验 |
| SDRC 部分步 C | 待测 | 待测 | 待测 | 待测 | 待测 | 消融 |

**不要在没有真实运行结果时填写虚构数值，也不要提前断言 SDRC 一定优于基线。**

---

## 12. 失败回退与调试顺序

如果结果变差，按以下顺序排查：

1. 确认 `x0_pred` 的推导公式与原采样器完全一致。
2. 确认目标潜变量和预测潜变量采用相同 VAE、缩放系数与归一化。
3. 确认掩码语义、尺寸和下采样方法。
4. 确认训练时的中间预测分布与推理时一致。
5. 将 `corrector_scale` 降低，检查是否为过度校正。
6. 比较每一步校正与只在后期校正。
7. 检查校正后的预测是否正确进入原采样器更新公式。
8. 检查未缺损区域是否被不必要地改变。
9. 检查训练/验证数据划分和缓存是否混用。
10. 如果模块未带来稳定收益，保留功能开关和基线，不要为了“看起来有创新”而保留无效模块。

---

## 13. 实施阶段与交付物

### 阶段 1：审查
交付：
- `docs/sdrc_code_audit.md`
- 实际插入点说明
- 可能的兼容性风险列表

### 阶段 2：实现网络和训练
交付：
- SDRC 网络文件
- 独立训练入口
- 配置文件
- 单元测试
- 简单训练日志与 checkpoint

### 阶段 3：集成采样
交付：
- 最小侵入式采样器改动
- SDRC 开关和权重加载
- 原始推理路径回归测试
- 校正器调用次数/时间步日志

### 阶段 4：实验
交付：
- 基线与 SDRC 的统一评估脚本或命令
- CSV/JSON 指标记录
- 可视化对比图
- 额外耗时和参数量记录
- 实验结论，明确报告提升、无提升或退化的情况

### 最终验收条件

- 原有模型权重可以正常加载。
- 原始掩码通道改进未被破坏。
- SDRC 可以独立训练和加载。
- SDRC 关闭时原始推理路径正常。
- SDRC 开启时能确认其输出参与采样更新。
- 推理过程不使用 GT。
- 所有关键张量形状、掩码语义、参数化和潜变量缩放均有源码依据。
- 至少在一小批样本上完成端到端运行。
- 评估结果来自真实运行，不填写虚构实验结论。

---

## 14. 给 AI 编程工具的最终任务描述

请按以下顺序执行：

1. 阅读仓库源码，完成 `docs/sdrc_code_audit.md`，说明真实的数据流、模型参数化、采样器更新公式和建议插入点。
2. 根据实际代码结构确定 SDRC 接口；先不要修改基础去噪网络。
3. 实现独立的轻量级残差校正网络，并保证零初始化时近似恒等映射。
4. 实现独立训练流程：冻结 Refusion，仅训练 SDRC，使用成对数据集监督中间预测残差。
5. 将 SDRC 以最小侵入方式接入原采样流程：校正干净潜变量预测后，继续调用原有采样器的正确更新公式。
6. 增加配置开关，关闭时保持原始路径；不覆盖已有权重。
7. 增加单元测试、集成测试、基线回归测试。
8. 在小数据子集上完成一次训练和推理验证。
9. 输出修改文件清单、运行命令、测试结果、未解决风险。
10. **先不要擅自大规模重构，不要删除原代码，不要编造实验提升；如果关键实现细节无法确认，先报告问题并提出最小修改方案。**

### 研究定位说明

本方案属于“在已有监督式扩散修复模型外增加一个需要单独训练的校正网络”。基础模型保持冻结，不等于整个方法无需训练。该方法的有效性、计算开销和泛化能力都需要通过实验验证。
