# Degradation-Aware Latent（退化感知潜空间）改进说明与消融开关

## 1. 动机

Refusion（IR-SDE + ConditionalNAFNet）处理局部非均匀退化（阴影/遮挡）时，
退化位置信息只隐含在条件图像 `mu`（masked_image）里，网络需要自己"猜"哪里需要修复。
本改进把二值退化掩膜 `mask`（1 = 退化/遮挡区，0 = 干净区）作为**显式退化先验**
注入到潜空间条件中，使 score 网络 `eps_theta(x_t, mu, t)` 变为
`eps_theta(x_t, mu, mask, t)`，即：

```
vanilla:  网络输入 = concat([x_t - mu, mu])            # 6 通道
improved: 网络输入 = concat([x_t - mu, mu, mask])      # 7 通道 (Degradation-Aware Latent)
```

## 2. 涉及的文件

| 文件 | 改动 |
| --- | --- |
| `codes/data/LQGTMask_dataset.py` | 新增。读取 `images / masked_images / masks` 三元组，裁剪/翻转/旋转与 LQ、GT 严格同步，mask 二值化为 1 通道 {0,1}；mask 文件名允许 `0001.png` 或 `0001_mask.png` 两种命名，按文件名 stem 配对 |
| `codes/data/__init__.py` | 注册新模式 `LQGTMask` |
| `codes/config/inpainting/models/modules/DenoisingNAFNet_arch.py` | `ConditionalNAFNet` 增加 `use_mask` 参数：为 True 时 intro 卷积输入通道 6→7，forward 拼接 mask；为 False 时与原版完全一致 |
| `codes/utils/sde_utils.py` | `IRSDE` 增加 `set_mask()`；`noise_fn/score_fn` 仅在 mask 非空时才给模型多传一个参数（对其他任务零影响） |
| `codes/config/inpainting/models/denoising_model.py` | `feed_data(..., mask=None)`；`optimize_parameters/test` 中 `sde.set_mask(...)`，且仅当网络 `use_mask: true` 时才启用 |
| `codes/config/inpainting/train.py` / `test.py` | 从 batch 中取 `mask` 传给 `feed_data`（普通 LQGT 数据集无该键，自动为 None） |
| `codes/config/inpainting/options.py` | 支持 `dataroot_mask` 路径展开 |

## 3. 消融开关（启用 / 关闭）

- **改进版（消融 ON）**：`options/train/refusion_mask.yml`
  - `mode: LQGTMask`、`dataroot_mask: .../masks`、`setting.use_mask: true`
- **基线（消融 OFF，即原版 Refusion）**：`options/train/refusion.yml`
  - `mode: LQGT`、无 `dataroot_mask`、无 `use_mask`（未改动，行为与原代码完全一致）

两个配置除 name / 数据集模式 / use_mask 外**超参完全相同**（Lion、lr 4e-5、niter 700k、
batch 4、crop 128、同 SDE 设置、同随机种子），保证消融公平。

```bash
# 训练改进版
python train.py -opt=options/train/refusion_mask.yml
# 训练基线
python train.py -opt=options/train/refusion.yml

# 测试（先在 options/test/refusion_mask.yml 的 path.pretrain_model_G 填入权重路径）
python test.py -opt=options/test/refusion_mask.yml
python test.py -opt=options/test/refusion.yml
```

结果分别保存在 `experiments/inpainting/refusion_mask` 与 `experiments/inpainting/refusion`，
`results/inpainting/<name>` 下测试输出可直接对比 PSNR/SSIM/LPIPS。

## 4. 注意事项

- mask 约定：**白色（1）= 被遮罩/退化的区域**，与 `masked_images` 中的白色空洞对应。
- 若已有旧版（6 通道）权重，不能用 `use_mask: true` 加载（intro 卷积形状不同），
  反之亦然；消融两组各自从零训练即可。
- `datasets/dark` 的 masks 为 3 通道二值图，dataset 内部已取 max 后按 0.5 阈值二值化，
  灰度图同样兼容。
- 推理（demo）时若不提供 mask，`use_mask: true` 的网络会自动以全零 mask 兜底，
  但建议测试/训练时始终提供真实 mask。
