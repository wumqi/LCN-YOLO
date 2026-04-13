import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- 基础卷积模块 -------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.relu: x = self.relu(x)
        return x


# ------------------------- CoordConv (位置信息嵌入) -------------------------
class CoordConv(nn.Module):
    """ Adds coordinate channels to input features """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def add_coords(self, x):
        b, _, h, w = x.size()
        y_coords = torch.linspace(-1, 1, h, device=x.device).unsqueeze(1).repeat(1, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).unsqueeze(0).repeat(h, 1)
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
        return torch.cat([x, x_coords, y_coords], dim=1)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# ------------------------- 自适应上下文注意力 CAA -------------------------
class CAA(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CAA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn


# ------------------------- FEMv2-CAP 模块主体 -------------------------
class FEMCAA(nn.Module):
    """
    Feature Enhancement Module v2 with Context and Position awareness.
    多尺度感受野 + 上下文注意 + 坐标卷积 + 自适应融合门控
    """
    def __init__(self, in_channels, out_channels, scale=0.05, reduction=8):
        super(FEMCAA, self).__init__()
        self.scale = scale
        inter_channels = in_channels // reduction

        # --- 多尺度并行分支 ---
        self.branch3x3 = BasicConv(in_channels, inter_channels, 3, 1, 1)
        self.branch5x5 = BasicConv(in_channels, inter_channels, 5, 1, 2)
        self.branch_dil3 = BasicConv(in_channels, inter_channels, 3, 1, 3, dilation=3)
        self.branch_dil5 = BasicConv(in_channels, inter_channels, 3, 1, 5, dilation=5)

        # --- 坐标卷积分支 ---
        self.coord_branch = CoordConv(in_channels, inter_channels)

        # --- 自适应注意力融合 ---
        self.attn = CAA(inter_channels * 5)
        self.gate = nn.Sequential(
            nn.Conv2d(inter_channels * 5, inter_channels * 5, 1),
            nn.BatchNorm2d(inter_channels * 5),
            nn.Sigmoid()
        )

        # --- 输出变换 ---
        self.out_conv = nn.Conv2d(inter_channels * 5, out_channels, 1, bias=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度提取
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        d3 = self.branch_dil3(x)
        d5 = self.branch_dil5(x)
        c1 = self.coord_branch(x)

        # 拼接融合
        out = torch.cat([b3, b5, d3, d5, c1], dim=1)

        # 上下文注意 + 门控融合
        attn = self.attn(out)
        gate = self.gate(out)
        out = attn * gate + out

        # 残差增强
        out = self.out_conv(out)
        out = out * self.scale + self.shortcut(x)
        out = self.relu(out)
        return out


# ------------------------- 测试 -------------------------
if __name__ == "__main__":
    x = torch.randn(1, 256, 64, 64)
    model = FEMCAA(256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
