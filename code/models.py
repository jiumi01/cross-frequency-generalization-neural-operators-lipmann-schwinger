# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Neural operators and baselines for the cross-frequency LS benchmark.

All models consume the same input tensor
    x = [ v , Re psi_inc , Im psi_inc , f/F_MAX ]      (4, N, N)
and predict the scattered field u = psi - psi_inc as two real channels
    y = [ Re u , Im u ]                                (2, N, N)
so that psi_hat = psi_inc + u_hat.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Fourier Neural Operator
# ----------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.in_ch, self.out_ch, self.m1, self.m2 = in_ch, out_ch, m1, m2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, m1, m2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, m1, m2, 2))

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1 = min(self.m1, H // 2)
        m2 = min(self.m2, W // 2 + 1)
        w1 = torch.view_as_complex(self.w1[:, :, :m1, :m2].contiguous())
        w2 = torch.view_as_complex(self.w2[:, :, :m1, :m2].contiguous())
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w1)
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w2)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, width=32, modes=16, n_layers=4,
                 proj=128, norm="none"):
        super().__init__()
        self.modes = modes
        self.norm = norm
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes, modes)
                                 for _ in range(n_layers)])
        self.wn = nn.ModuleList([nn.Conv2d(width, width, 1)
                                 for _ in range(n_layers)])
        if norm == "layer":
            self.nrm = nn.ModuleList([nn.GroupNorm(1, width)
                                      for _ in range(n_layers)])
        else:
            self.nrm = None
        self.fc1 = nn.Conv2d(width, proj, 1)
        self.fc2 = nn.Conv2d(proj, out_ch, 1)

    def forward(self, x):
        x = self.fc0(x)
        for i, (s, w) in enumerate(zip(self.sp, self.wn)):
            y = F.gelu(s(x) + w(x))
            x = self.nrm[i](y) if self.nrm is not None else y
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# ----------------------------------------------------------------------------
# DeepONet (branch = field encoder, trunk = coordinate MLP)
# ----------------------------------------------------------------------------
class DeepONet2d(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, width=32, p=128, n_grid=64, depth=4):
        super().__init__()
        self.p = p
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.branch = nn.Sequential(
            nn.Linear(2 * width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, out_ch * p),
        )
        x = torch.linspace(0, 1, n_grid)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("coords", torch.stack([X.reshape(-1), Y.reshape(-1)], 1))
        trunk = [nn.Linear(2, 4 * width), nn.GELU()]
        for _ in range(depth - 1):
            trunk += [nn.Linear(4 * width, 4 * width), nn.GELU()]
        trunk += [nn.Linear(4 * width, p)]
        self.trunk = nn.Sequential(*trunk)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        B, _, H, W = x.shape
        b = self.branch(self.enc(x)).view(B, 2, self.p)          # (B,2,p)
        t = self.trunk(self.coords)                              # (N^2,p)
        out = torch.einsum("bcp,np->bcn", b, t) + self.bias[None, :, None]
        return out.reshape(B, 2, H, W)


# ----------------------------------------------------------------------------
# Matched end-to-end CNN baseline (U-Net)
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet2d(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, base=24, depth=3):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.down = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(ConvBlock(chs[i] * 2, chs[i]))
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < self.depth:
                skips.append(x)
                x = self.down(x)
        for up, dec, sk in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, sk], dim=1))
        return self.out(x)


class FNO2dFiLM(nn.Module):
    """FNO with per-layer FiLM modulation by the scalar frequency.

    The baseline encodes f only as an input channel; this variant also injects
    f as a per-layer affine modulation (feature-wise linear modulation),
    testing whether a *conditional* architecture that can route the frequency
    through the whole depth moves the band-edge boundary.  The frequency is
    read from the fourth input channel (f/F_MAX, spatially constant), so the
    input tensor is identical to the baseline's.
    """

    def __init__(self, in_ch=4, out_ch=2, width=32, modes=16, n_layers=4,
                 proj=128, norm="none"):
        super().__init__()
        self.modes = modes
        self.norm = norm
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes, modes)
                                 for _ in range(n_layers)])
        self.wn = nn.ModuleList([nn.Conv2d(width, width, 1)
                                 for _ in range(n_layers)])
        if norm == "layer":
            self.nrm = nn.ModuleList([nn.GroupNorm(1, width)
                                      for _ in range(n_layers)])
        else:
            self.nrm = None
        self.film = nn.ModuleList([
            nn.Sequential(nn.Linear(1, width), nn.GELU(),
                          nn.Linear(width, 2 * width))
            for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, proj, 1)
        self.fc2 = nn.Conv2d(proj, out_ch, 1)

    def forward(self, x):
        f = x[:, 3:4, 0, 0].to(x.dtype)            # f/F_MAX scalar per sample
        x = self.fc0(x)
        for i, (s, w) in enumerate(zip(self.sp, self.wn)):
            y = F.gelu(s(x) + w(x))
            gb = self.film[i](f)                    # (B, 2*width)
            g, b = gb.chunk(2, dim=1)               # (B, width) each
            y = y * g[:, :, None, None] + b[:, :, None, None]
            x = self.nrm[i](y) if self.nrm is not None else y
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# ----------------------------------------------------------------------------
def build_model(name, **kw):
    name = name.lower()
    if name == "fno":
        return FNO2d(**kw)
    if name == "fno-film":
        return FNO2dFiLM(**kw)
    if name == "deeponet":
        return DeepONet2d(**kw)
    if name in ("unet", "cnn"):
        return UNet2d(**kw)
    raise ValueError(name)
