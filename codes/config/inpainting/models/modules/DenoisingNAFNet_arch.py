import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from .module_util import SinusoidalPosEmb, LayerNorm, exists
from .mtsce import MTSCE, normalize_mtsce_config, resize_mask

logger = logging.getLogger("base")


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time


class ConditionalNAFNet(nn.Module):
    """Refusion conditional NAFNet denoiser with optional mask and MTSCE.

    Extra (compared with the vanilla Refusion implementation):

    * ``use_mask``: concat the binary degradation mask to ``[x_t - mu, mu]``
      (Degradation-Aware Latent, unchanged behaviour).
    * ``mtsce``: optional config dict for the Multi-scale Texture and Structure
      Collaborative Enhancement module, see ``modules/mtsce.py``. When
      ``mtsce.enabled`` is false nothing is built and the forward pass is
      bit-identical to the original network.

    Insertion points (``mtsce.insertion_points``):
        ``encoder_<i>``  after the i-th encoder stage (1 = highest resolution)
        ``bottleneck``   after ``middle_blks`` (position A of the design doc)
        ``decoder_<i>``  after the i-th decoder stage (1 = lowest resolution,
                         so ``decoder_<last>`` is the highest-resolution
                         decoder, i.e. position B of the design doc)
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1, use_mask=False,
                 mtsce=None, proposal_ch=0):
        super().__init__()
        self.upscale = upscale
        self.use_mask = use_mask  # Degradation-Aware Latent: extra 1-ch mask input

        # Correction training (see modules/correction.py): an extra
        # zero-initialised 1x1 projection that injects the *denoiser's*
        # prediction into the corrector's first feature map.  With
        # ``proposal_ch = 0`` nothing is built and the forward pass is
        # bit-identical to the original network, so the baseline denoiser is
        # unaffected.
        self.proposal_ch = int(proposal_ch or 0)

        # ---- MTSCE configuration -------------------------------------------
        self.mtsce_cfg = normalize_mtsce_config(mtsce)
        self.mtsce_enabled = bool(self.mtsce_cfg["enabled"])
        if self.mtsce_enabled and self.mtsce_cfg["use_mask_condition"] and not use_mask:
            logger.warning(
                "MTSCE uses the degradation mask for its fusion gate but "
                "network_G.setting.use_mask is false; the mask is then only "
                "visible to MTSCE (not concatenated to the network input)."
            )

        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel*2 + (1 if use_mask else 0), out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.proposal_proj = None
        if self.proposal_ch > 0:
            self.proposal_proj = nn.Conv2d(
                in_channels=self.proposal_ch, out_channels=width, kernel_size=1,
                padding=0, stride=1, groups=1, bias=True,
            )
            nn.init.zeros_(self.proposal_proj.weight)
            nn.init.zeros_(self.proposal_proj.bias)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        enc_channels = []
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            enc_channels.append(chan)
            chan = chan * 2

        middle_channels = chan
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        dec_channels = []
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            dec_channels.append(chan)
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        # ---- MTSCE modules --------------------------------------------------
        self.mtsce = nn.ModuleDict()
        self.mtsce_points = []
        if self.mtsce_enabled:
            for name in self.mtsce_cfg["insertion_points"]:
                ch = self._resolve_point_channels(
                    name, enc_channels, middle_channels, dec_channels
                )
                self.mtsce[name] = MTSCE(
                    in_ch=ch,
                    hidden_ratio=self.mtsce_cfg["hidden_ratio"],
                    min_hidden=self.mtsce_cfg["min_hidden"],
                    structure_dilations=self.mtsce_cfg["structure_dilations"],
                    texture_kernel_size=self.mtsce_cfg["texture_kernel_size"],
                    use_structure=self.mtsce_cfg["use_structure"],
                    use_texture=self.mtsce_cfg["use_texture"],
                    texture_norm=self.mtsce_cfg["texture_norm"],
                    fusion=self.mtsce_cfg["fusion"],
                    use_mask_condition=self.mtsce_cfg["use_mask_condition"],
                    mask_feat_ch=self.mtsce_cfg["mask_feat_ch"],
                    mask_invert=self.mtsce_cfg["mask_invert"],
                    residual_scale_init=self.mtsce_cfg["residual_scale_init"],
                )
                self.mtsce_points.append(name)
            logger.info(
                "MTSCE built at %s (hidden_ratio=%s, fusion=%s, dilations=%s)",
                self.mtsce_points,
                self.mtsce_cfg["hidden_ratio"],
                self.mtsce_cfg["fusion"],
                self.mtsce_cfg["structure_dilations"],
            )

    @staticmethod
    def _resolve_point_channels(name, enc_channels, middle_channels, dec_channels):
        if not isinstance(name, str):
            raise TypeError("mtsce insertion point must be a string, got {!r}".format(name))
        if name == "bottleneck":
            return middle_channels
        for prefix, chans in (("encoder_", enc_channels), ("decoder_", dec_channels)):
            if name.startswith(prefix):
                try:
                    idx = int(name[len(prefix):])
                except ValueError:
                    break
                if 1 <= idx <= len(chans):
                    return chans[idx - 1]
                raise ValueError(
                    "mtsce insertion point {!r} is out of range; the network has "
                    "{} {} stages".format(name, len(chans), prefix[:-1])
                )
        valid = ["bottleneck"]
        valid += ["encoder_{}".format(i + 1) for i in range(len(enc_channels))]
        valid += ["decoder_{}".format(i + 1) for i in range(len(dec_channels))]
        raise ValueError(
            "unknown mtsce insertion point {!r}. Valid names: {}".format(name, valid)
        )

    def _apply_mtsce(self, name, x, mask):
        if not self.mtsce_enabled:
            return x
        if name not in self.mtsce:
            return x
        return self.mtsce[name](x, mask)

    def forward(self, inp, cond, time, mask=None, proposal=None):
        inp_res = inp.clone()

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        feats = [x, cond]
        if self.use_mask:
            if mask is None:
                # all-clean fallback so the network keeps working without a mask
                mask = torch.zeros_like(inp[:, :1])
            else:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                if mask.shape[1] != 1:
                    # a colour mask carries no extra information: collapse it
                    mask = mask.amax(dim=1, keepdim=True)
                if mask.shape[-2:] != inp.shape[-2:]:
                    # keep the mask aligned with the image, whatever its size
                    mask = resize_mask(
                        mask, inp.shape[-2:], dtype=inp.dtype, device=inp.device
                    )
            feats.append(mask)
        x = torch.cat(feats, dim=1)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        # the mask must follow the same padding so that every MTSCE stage sees
        # a mask that is spatially aligned with its own feature map
        mask_in = mask
        if self.mtsce_enabled and mask_in is not None:
            mask_in = mask_in.to(device=x.device, dtype=x.dtype)
            pad_h, pad_w = x.shape[-2] - H, x.shape[-1] - W
            if pad_h or pad_w:
                mask_in = F.pad(mask_in, (0, pad_w, 0, pad_h))

        x = self.intro(x)

        if self.proposal_proj is not None:
            if proposal is None:
                raise ValueError(
                    "this network was built with proposal_ch={} but the forward "
                    "pass received proposal=None".format(self.proposal_ch)
                )
            p = proposal.to(dtype=x.dtype, device=x.device)
            if p.shape[-2:] != x.shape[-2:]:
                p = F.pad(p, (0, x.shape[-1] - p.shape[-1], 0, x.shape[-2] - p.shape[-2]))
            x = x + self.proposal_proj(p)

        encs = []

        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x, _ = encoder([x, t])
            x = self._apply_mtsce("encoder_{}".format(i + 1), x, mask_in)
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])
        x = self._apply_mtsce("bottleneck", x, mask_in)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])
            x = self._apply_mtsce("decoder_{}".format(i + 1), x, mask_in)

        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    # -- reporting ---------------------------------------------------------
    def mtsce_parameter_count(self):
        """Number of parameters that belong to MTSCE (0 when disabled)."""
        if not self.mtsce_enabled:
            return 0
        return sum(p.numel() for n, p in self.named_parameters() if "mtsce" in n.split("."))

    def parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        extra = self.mtsce_parameter_count()
        return {"total": total, "mtsce": extra, "base": total - extra}
