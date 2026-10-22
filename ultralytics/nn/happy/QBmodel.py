import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Blur Estimation Branch
# 模糊估计分支
# =========================================================
class BlurEstimator(nn.Module):
    """
    估计局部模糊程度
    用于动态融合
    """

    def __init__(self, channels):
        super().__init__()

        self.local = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            )
        )

        self.blur_proj = nn.Sequential(

            nn.Conv2d(channels, channels, 1, bias=False),

            nn.BatchNorm2d(channels),

            nn.Sigmoid()
        )

    def forward(self, x):

        # 高频残差
        high_freq = x - F.avg_pool2d(x, 3, 1, 1)

        feat = self.local(high_freq)

        blur_map = self.blur_proj(feat)

        return blur_map


# =========================================================
# Deblur Branch
# 去模糊分支
# =========================================================
class DeblurBranch(nn.Module):
    """
    高频恢复 + 动态感受野
    """

    def __init__(self, channels):
        super().__init__()

        # 小卷积核
        self.conv3 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False
        )

        # 大卷积核
        self.conv5 = nn.Conv2d(
            channels,
            channels,
            5,
            padding=2,
            groups=channels,
            bias=False
        )

        # 高频增强
        self.high_enhance = nn.Sequential(

            nn.Conv2d(channels, channels, 1, bias=False),

            nn.BatchNorm2d(channels),

            nn.GELU()
        )

        self.fuse = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

    def forward(self, x, blur_map):

        feat3 = self.conv3(x)

        feat5 = self.conv5(x)

        # 模糊区域动态使用大感受野
        feat = feat3 * (1 - blur_map) + feat5 * blur_map

        # 高频恢复
        high = x - F.avg_pool2d(x, 5, 1, 2)

        high = self.high_enhance(high)

        feat = feat + high

        return self.fuse(feat)


# =========================================================
# Structure Recovery Branch
# 结构恢复分支
# =========================================================
class StructureRecovery(nn.Module):
    """
    结构边缘恢复
    """

    def __init__(self, channels):
        super().__init__()

        self.edge = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.GELU()
        )

        self.recover = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.GELU()
        )

    def forward(self, x):

        edge = x - F.avg_pool2d(x, 3, 1, 1)

        edge = self.edge(edge)

        out = self.recover(edge)

        return out


# =========================================================
# Context Branch
# 上下文辅助分支
# =========================================================
class ContextBranch(nn.Module):
    """
    大感受野上下文建模
    港口小目标非常重要
    """

    def __init__(self, channels):
        super().__init__()

        self.context = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                7,
                padding=3,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            )
        )

    def forward(self, x):

        return self.context(x)


# =========================================================
# Confidence Adaptive Fusion
# 动态融合模块（核心创新）
# =========================================================
class CAF(nn.Module):
    """
    动态融合：
    去模糊 / 结构恢复 / 上下文
    """

    def __init__(self, channels):
        super().__init__()

        self.weight_generator = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(channels * 3, channels, 1, bias=False),

            nn.GELU(),

            nn.Conv2d(channels, 3, 1, bias=False),

            nn.Softmax(dim=1)
        )

    def forward(self, deblur, structure, context):

        fusion = torch.cat(
            [deblur, structure, context],
            dim=1
        )

        weights = self.weight_generator(fusion)

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]

        out = (
            w1 * deblur +
            w2 * structure +
            w3 * context
        )

        return out


# =========================================================
# BRAM++
# Blur-aware Residual Attention Module
# =========================================================
class BRAM(nn.Module):

    def __init__(self, c1, c2, ratio=16, *args, **kwargs):

        super().__init__()

        self.c1 = c1
        self.c2 = c2

        # -------------------------------------------------
        # 1. 模糊估计
        # -------------------------------------------------
        self.blur_estimator = BlurEstimator(c1)

        # -------------------------------------------------
        # 2. 去模糊分支
        # -------------------------------------------------
        self.deblur_branch = DeblurBranch(c1)

        # -------------------------------------------------
        # 3. 结构恢复分支
        # -------------------------------------------------
        self.structure_branch = StructureRecovery(c1)

        # -------------------------------------------------
        # 4. 上下文分支
        # -------------------------------------------------
        self.context_branch = ContextBranch(c1)

        # -------------------------------------------------
        # 5. 动态融合
        # -------------------------------------------------
        self.caf = CAF(c1)

        # -------------------------------------------------
        # 6. 细化增强
        # -------------------------------------------------
        self.refine = nn.Sequential(

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

            nn.Conv2d(
                c1,
                c1,
                1,
                bias=False
            )
        )

        # -------------------------------------------------
        # 7. Channel Attention
        # -------------------------------------------------
        hidden = max(8, c1 // ratio)

        self.ca = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, hidden, 1, bias=False),

            nn.GELU(),

            nn.Conv2d(hidden, c1, 1, bias=False),

            nn.Sigmoid()
        )

        # -------------------------------------------------
        # 8. Output Projection
        # -------------------------------------------------
        self.out_proj = nn.Sequential(

            nn.Conv2d(c1, c2, 1, bias=False),

            nn.BatchNorm2d(c2),

            nn.SiLU()
        )

        # -------------------------------------------------
        # 9. Shortcut
        # -------------------------------------------------
        self.shortcut = nn.Sequential(

            nn.Conv2d(c1, c2, 1, bias=False),

            nn.BatchNorm2d(c2)

        ) if c1 != c2 else nn.Identity()

    def forward(self, x):

        # -------------------------------------------------
        # 1. Blur Estimation
        # -------------------------------------------------
        blur_map = self.blur_estimator(x)

        # -------------------------------------------------
        # 2. Deblur Branch
        # -------------------------------------------------
        deblur_feat = self.deblur_branch(x, blur_map)

        # -------------------------------------------------
        # 3. Structure Recovery
        # -------------------------------------------------
        structure_feat = self.structure_branch(x)

        # -------------------------------------------------
        # 4. Context Branch
        # -------------------------------------------------
        context_feat = self.context_branch(x)

        # -------------------------------------------------
        # 5. Dynamic Fusion
        # -------------------------------------------------
        feat = self.caf(
            deblur_feat,
            structure_feat,
            context_feat
        )

        # -------------------------------------------------
        # 6. Refinement
        # -------------------------------------------------
        feat = self.refine(feat)

        # -------------------------------------------------
        # 7. Channel Attention
        # -------------------------------------------------
        feat = feat * self.ca(feat)

        # -------------------------------------------------
        # 8. Residual Output
        # -------------------------------------------------
        out = self.out_proj(feat + x)

        return out + self.shortcut(x)


__all__ = ["BRAM"]