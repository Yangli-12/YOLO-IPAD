# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SDFA_PP(nn.Module):
#     def __init__(self, c1, c2, ratio=16, *args, **kwargs):
#         super().__init__()
#         self.c1 = c1
#         self.c2 = c2
#
#         # 1. 结构模拟分支
#         self.struct_branch = nn.Sequential(
#             nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
#             nn.Conv2d(c1, c1, 1, bias=False),
#             nn.BatchNorm2d(c1),
#             nn.GELU()
#         )
#
#         # 2. 差分特征生成
#         self.diff_conv = nn.Sequential(
#             nn.Conv2d(c1, c1, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
#         )
#
#         # 3. 交叉注意力线性投影
#         self.q = nn.Conv2d(c1, c1, 1, bias=False)
#         self.k = nn.Conv2d(c1, c1, 1, bias=False)
#         self.v = nn.Conv2d(c1, c1, 1, bias=False)
#         self.scale = c1 ** -0.5
#
#         # 4. 精炼层
#         reduced = max(1, c1 // ratio)
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c1, reduced, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(reduced, c1, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 5. 通道对齐 (关键修复点 🛠️)
#         # 如果输入 c1=4, 输出 c2=32，我们需要把 identity 从 4 变成 32
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(c1, c2, 1, bias=False),
#             nn.BatchNorm2d(c2)
#         ) if c1 != c2 else nn.Identity()
#
#         self.out_proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()
#
#     def _cross_attn(self, x, diff):
#         B, C, H, W = x.shape
#         # 针对 5090 的极速优化：大图用线性注意，小图用全局矩阵
#         if H * W > 4096:
#             q = torch.sigmoid(self.q(x))
#             k = torch.sigmoid(self.k(diff))
#             return x * (q * k)
#
#         q = self.q(x).view(B, C, -1)
#         k = self.k(diff).view(B, C, -1)
#         v = self.v(diff).view(B, C, -1)
#
#         attn = torch.bmm(q.transpose(1, 2), k) * self.scale
#         attn = attn.softmax(dim=-1)
#         out = torch.bmm(attn, v.transpose(1, 2))
#         return out.transpose(1, 2).view(B, C, H, W)
#
#     def forward(self, x):
#         # 1. 提取差分
#         struct = self.struct_branch(x)
#         diff = self.diff_conv(x - struct)
#
#         # 2. 交叉注意力 (x 是 Q, diff 是 K, V)
#         x_attn = self._cross_attn(x, diff)
#
#         # 3. 增强
#         x_attn = x_attn * self.ca(x_attn)
#
#         # 4. 融合与对齐 (解决 32 对不上 4 的问题)
#         # identity 经过 shortcut 卷积后变成 32 通道
#         identity = self.shortcut(x)
#         out = self.bn(self.out_proj(x_attn + x))
#
#         return out + identity


import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePrior(nn.Module):
    """
    边缘先验增强
    强化港口场景小目标边缘
    """

    def __init__(self, channels):
        super().__init__()

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],
            dtype=torch.float32
        )

        self.register_buffer(
            'weight_x',
            sobel_x.unsqueeze(0).repeat(channels, 1, 1, 1)
        )

        self.register_buffer(
            'weight_y',
            sobel_y.unsqueeze(0).repeat(channels, 1, 1, 1)
        )

        self.channels = channels

    def forward(self, x):

        edge_x = F.conv2d(
            x,
            self.weight_x,
            padding=1,
            groups=self.channels
        )

        edge_y = F.conv2d(
            x,
            self.weight_y,
            padding=1,
            groups=self.channels
        )

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edge


class FrequencyBranch(nn.Module):
    """
    高频增强分支
    强化远距离/低对比度小目标
    """

    def __init__(self, channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):

        # 低频估计
        low_freq = F.avg_pool2d(x, 5, stride=1, padding=2)

        # 高频残差
        high_freq = x - low_freq

        return self.proj(high_freq)


class LinearDiffAttention(nn.Module):
    """
    轻量级差分注意力
    针对港口小目标优化
    """

    def __init__(self, channels):
        super().__init__()

        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)

        self.dw = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False
        )

    def forward(self, x, diff):

        q = torch.sigmoid(self.q(x))
        k = torch.sigmoid(self.k(diff))
        v = self.v(diff)

        attn = q * k

        out = v * attn

        out = self.dw(out)

        return out


class SDFA_PP(nn.Module):

    def __init__(self, c1, c2, ratio=16, *args, **kwargs):
        super().__init__()

        self.c1 = c1
        self.c2 = c2

        # ---------------------------------
        # 1. 结构建模分支
        # ---------------------------------
        self.structure = nn.Sequential(

            nn.Conv2d(
                c1,
                c1,
                3,
                padding=1,
                groups=c1,
                bias=False
            ),

            nn.BatchNorm2d(c1),

            nn.GELU(),

            nn.Conv2d(c1, c1, 1, bias=False)
        )

        # ---------------------------------
        # 2. 边缘先验
        # ---------------------------------
        self.edge_prior = EdgePrior(c1)

        # ---------------------------------
        # 3. 高频增强
        # ---------------------------------
        self.freq_branch = FrequencyBranch(c1)

        # ---------------------------------
        # 4. 差分增强
        # ---------------------------------
        self.diff_enhance = nn.Sequential(

            nn.Conv2d(
                c1,
                c1,
                3,
                padding=1,
                groups=c1,
                bias=False
            ),

            nn.GELU(),

            nn.Conv2d(c1, c1, 1, bias=False)
        )

        # ---------------------------------
        # 5. 差分注意力
        # ---------------------------------
        self.attn = LinearDiffAttention(c1)

        # ---------------------------------
        # 6. 通道注意力
        # ---------------------------------
        hidden = max(8, c1 // ratio)

        self.ca = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, hidden, 1, bias=False),

            nn.GELU(),

            nn.Conv2d(hidden, c1, 1, bias=False),

            nn.Sigmoid()
        )

        # ---------------------------------
        # 7. 输出投影
        # ---------------------------------
        self.out_proj = nn.Sequential(

            nn.Conv2d(c1, c2, 1, bias=False),

            nn.BatchNorm2d(c2),

            nn.SiLU()
        )

        # ---------------------------------
        # 8. Shortcut
        # ---------------------------------
        self.shortcut = nn.Sequential(

            nn.Conv2d(c1, c2, 1, bias=False),

            nn.BatchNorm2d(c2)

        ) if c1 != c2 else nn.Identity()

    def forward(self, x):

        # ---------------------------------
        # 1. 结构建模
        # ---------------------------------
        structure = self.structure(x)

        # ---------------------------------
        # 2. 空间差分
        # ---------------------------------
        spatial_diff = x - structure

        # ---------------------------------
        # 3. 边缘先验
        # ---------------------------------
        edge = self.edge_prior(x)

        # ---------------------------------
        # 4. 高频增强
        # ---------------------------------
        freq = self.freq_branch(x)

        # ---------------------------------
        # 5. 差分融合
        # ---------------------------------
        diff = spatial_diff + edge + freq

        diff = self.diff_enhance(diff)

        # ---------------------------------
        # 6. 差分注意力
        # ---------------------------------
        out = self.attn(x, diff)

        # ---------------------------------
        # 7. 通道增强
        # ---------------------------------
        out = out * self.ca(out)

        # ---------------------------------
        # 8. 残差融合
        # ---------------------------------
        out = self.out_proj(out + x)

        return out + self.shortcut(x)


__all__ = ["SDFA_PP"]