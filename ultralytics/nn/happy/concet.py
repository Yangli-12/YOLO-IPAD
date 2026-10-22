from torch import nn


class CAFPN(nn.Module):
    """Context / Confidence Aware Feature Pyramid Fusion.

    用于替代：
        Concat -> C3k2

    核心作用：
    1. 动态融合浅层细节与深层语义
    2. 退化感知融合
    3. 保留小目标高频纹理
    4. 避免深层语义覆盖浅层细节

    特别适合：
    - 港口场景
    - 模糊目标
    - 小目标检测
    """

    def __init__(self, c1, c2=None, ratio=16, *args, **kwargs):
        super().__init__()

        if c2 is None:
            c2 = c1

        self.c1 = c1
        self.c2 = c2

        # =====================================================
        # 1. Local Detail Branch
        # 高频细节增强
        # =====================================================
        self.local_branch = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 1, bias=False),
        )

        # =====================================================
        # 2. Context Branch
        # 上下文语义增强
        # =====================================================
        self.context_branch = nn.Sequential(
            nn.Conv2d(c1, c1, 5, padding=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 1, bias=False),
        )

        # =====================================================
        # 3. Degradation-aware Weight Generator
        # 动态权重生成
        # =====================================================
        hidden = max(8, c1 // ratio)

        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # =====================================================
        # 4. Refinement
        # 融合细化
        # =====================================================
        self.refine = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # =====================================================
        # 5. Shortcut
        # =====================================================
        self.shortcut = (
            nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2)) if c1 != c2 else nn.Identity()
        )

    def forward(self, x):

        # =====================================================
        # Local Detail Feature
        # =====================================================
        local_feat = self.local_branch(x)

        # =====================================================
        # Context Semantic Feature
        # =====================================================
        context_feat = self.context_branch(x)

        # =====================================================
        # Dynamic Fusion Weights
        # =====================================================
        weights = self.weight_generator(x)

        w_local = weights[:, 0:1]
        w_context = weights[:, 1:2]

        # =====================================================
        # Adaptive Fusion
        # =====================================================
        fusion = w_local * local_feat + w_context * context_feat

        # =====================================================
        # Refinement
        # =====================================================
        out = self.refine(fusion)

        # =====================================================
        # Residual Connection
        # =====================================================
        return out + self.shortcut(x)


__all__ = ["CAFPN"]
