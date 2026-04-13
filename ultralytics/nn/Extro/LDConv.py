import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # 使用列表推导式而不是生成器，避免重复计算
        grad_input = [grad_input[i] * 0.1 for i in range(len(grad_input))]
        grad_output = [grad_output[i] * 0.1 for i in range(len(grad_output))]

    def forward(self, x):
        # 预计算常用值
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # 获取偏移量
        offset = self.p_conv(x)
        N = self.num_param

        # 计算采样坐标
        p = self._get_p_optimized(offset, h, w, N, device, dtype)

        # 使用grid_sample进行高效的双线性插值
        x_offset = self._bilinear_sample_optimized(x, p, N)

        # 重塑并应用卷积
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N):
        """预计算采样点的基础坐标"""
        base_int = round(math.sqrt(N))
        row_number = N // base_int
        mod_number = N % base_int

        # 使用更高效的坐标生成
        p_n_x_list = []
        p_n_y_list = []

        # 生成主要网格点
        for i in range(row_number):
            for j in range(base_int):
                p_n_x_list.append(i)
                p_n_y_list.append(j)

        # 生成剩余点
        if mod_number > 0:
            for j in range(mod_number):
                p_n_x_list.append(row_number)
                p_n_y_list.append(j)

        p_n_x = torch.tensor(p_n_x_list, dtype=torch.float32)
        p_n_y = torch.tensor(p_n_y_list, dtype=torch.float32)
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)

        return p_n

    def _get_p_0(self, h, w, N, dtype, device):
        """生成基础坐标网格"""
        # 生成基础网格 - 使用更高效的方法
        y_coords = torch.arange(0, h * self.stride, self.stride, device=device, dtype=dtype)
        x_coords = torch.arange(0, w * self.stride, self.stride, device=device, dtype=dtype)

        # 使用meshgrid生成完整网格
        p_0_x, p_0_y = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 展平并重复N次
        p_0_x = p_0_x.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1)

        return p_0

    def _get_p_optimized(self, offset, h, w, N, device, dtype):
        """优化的坐标计算"""
        # 生成基础网格
        p_0 = self._get_p_0(h, w, N, dtype, device)

        # 确保p_n在正确的设备上
        p_n = self.p_n.to(device)

        # 添加偏移
        p = p_0 + p_n + offset

        return p

    def _bilinear_sample_optimized(self, x, p, N):
        """使用grid_sample的优化双线性插值"""
        b, c, h, w = x.shape
        _, _, out_h, out_w = p.shape[:4]

        # 将坐标重塑为grid_sample期望的格式
        # p的形状: [b, 2*N, h, w]
        p = p.view(b, 2, N, out_h, out_w)

        # 分离x和y坐标
        p_y = p[:, 0, :, :, :]  # [b, N, h, w]
        p_x = p[:, 1, :, :, :]  # [b, N, h, w]

        # 归一化坐标到[-1, 1]范围
        p_y_norm = 2.0 * p_y / max(h - 1, 1) - 1.0
        p_x_norm = 2.0 * p_x / max(w - 1, 1) - 1.0

        # 对每个采样点执行grid_sample
        sampled_features = []
        for i in range(N):
            # grid_sample期望的格式: [b, h, w, 2] 其中最后一维是[x, y]
            grid = torch.stack([p_x_norm[:, i, :, :], p_y_norm[:, i, :, :]], dim=-1)
            sampled = F.grid_sample(x, grid, mode='bilinear',
                                    padding_mode='border', align_corners=True)
            sampled_features.append(sampled)

        # 堆叠结果: [b, c, h, w, N]
        x_offset = torch.stack(sampled_features, dim=-1)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        """高效的特征重塑"""
        # 使用einops进行重塑，从 [b, c, h, w, n] 到 [b, c, h*n, w]
        return rearrange(x_offset, 'b c h w n -> b c (h n) w')