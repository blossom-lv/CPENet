from mobilevit import mobile_vit_xx_small, mobile_vit_small
import torch
import torch.nn as nn
import torch.nn.functional as F
from SwinTransformers import SwinTransformer
import math

def dilate(input, ksize=3):
    src_size = input.size()
    out = F.max_pool2d(input, kernel_size=ksize, stride=1, padding=0)
    out = F.interpolate(out, size=src_size[2:], mode="bilinear")
    return out

def resize_match(source, new):
    height, width = source.shape[2], source.shape[3]
    new = F.interpolate(new, size=(height, width), mode='bilinear', align_corners=False)
    
    return new

class DWConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super(DWConv, self).__init__()
        # Depthwise convolution: each input channel has its own convolutional filter
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
            )
    
    def forward(self, x):
        return self.dwconv(x)

class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(PWConv, self).__init__()
        # Pointwise convolution: 1x1 convolution to combine the features
        self.pwconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
            )
    
    def forward(self, x):
        return self.pwconv(x)

class BConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, **kwargs):
        super(BConv, self).__init__()
        
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise ValueError("Currently only kernel_size 1 or 3 is supported for maintaining output size.")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv1(x_out)
        return self.sigmoid(out)

class CrossSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossSelfAttention, self).__init__()
        self.query_conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.query_conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))  # 学习的加权系数

    def forward(self, f_global, f_modal):
        # f_global 和 f_modal 的形状为 (B, C, H, W)
        batch_size, channels, height, width = f_global.size()

        # 对第一个图像特征 (f_global) 和第二个图像特征 (f_modal) 进行卷积
        Q1 = self.query_conv_1(f_global).view(batch_size, channels, -1)  # (B, C, H*W)
        K2 = self.key_conv_2(f_modal).view(batch_size, channels, -1)  # (B, C, H*W)
        V2 = self.value_conv_2(f_modal).view(batch_size, channels, -1)  # (B, C, H*W)

        Q2 = self.query_conv_2(f_modal).view(batch_size, channels, -1)  # (B, C, H*W)
        K1 = self.key_conv_1(f_global).view(batch_size, channels, -1)  # (B, C, H*W)
        V1 = self.value_conv_1(f_global).view(batch_size, channels, -1)  # (B, C, H*W)

        # 计算注意力权重
        A12 = torch.matmul(Q1.transpose(-2, -1), K2) / (channels ** 0.5)  # (B, H*W, H*W)
        A21 = torch.matmul(Q2.transpose(-2, -1), K1) / (channels ** 0.5)  # (B, H*W, H*W)

        # 归一化注意力权重
        attention_weights_12 = F.softmax(A12, dim=-1)  # (B, H*W, H*W)
        attention_weights_21 = F.softmax(A21, dim=-1)  # (B, H*W, H*W)

        # 加权求和
        O1 = torch.matmul(attention_weights_12, V2.transpose(-2, -1))  # (B, H*W, C)
        O2 = torch.matmul(attention_weights_21, V1.transpose(-2, -1))  # (B, H*W, C)

        # 恢复维度
        O1 = O1.view(batch_size, channels, height, width)
        O2 = O2.view(batch_size, channels, height, width)
        #print('O1O2size:', O1.size(), O2.size())

        # 最终输出结果
        output = self.gamma * (O1 + O2)# + f_global  # 残差连接
        #print('output size:', output.size())

        return output

class fusion_scale_ca(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BConv(in_channels * 2, in_channels)
        self.ca = ChannelAttention(in_channels)
        #self.sa = SpatialAttention()
        self.conv2 = BConv(in_channels, out_channels, 1)

    def forward(self, high, low):
        #x = torch.cat((self.upsample2(low), high), 1)
        #print('high size:', high.size(), 'low size:', low.size())
        x = torch.cat((low, high), 1)
        x = self.conv1(x)
        # B, C, H, W = x.size()
        # x1, x2 = torch.split(x, C // 2, dim=1)
        x = x + self.ca(x) * x
        # x = x + self.sa(x) * x
        y = self.conv2(x)

        return y

class fusion_scale_sa(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BConv(in_channels * 2, in_channels)
        #self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.conv2 = BConv(in_channels, out_channels, 1)

    def forward(self, high, low):
        #x = torch.cat((self.upsample2(low), high), 1)
        #print('high size:', high.size(), 'low size:', low.size())
        x = torch.cat((low, high), 1)
        x = self.conv1(x)
        x = x + self.sa(x) * x
        y = self.conv2(x)

        return y

class fusion_modal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sa = SpatialAttention()
        self.conv_r = BConv(in_channels, in_channels, 1)
        self.conv_d = BConv(in_channels, in_channels, 1)

        self.ca = ChannelAttention(in_channels * 2)
        self.conv1 = BConv(in_channels * 2, in_channels)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cross_attention = CrossSelfAttention(in_channels)
        #self.cross_attention2 = CrossSelfAttention(in_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BConv(in_channels, in_channels, stride=2)
        self.conv3 = BConv(in_channels, out_channels)
        
        

    def forward(self, rgb, depth, edge, body):
        B, C, H, W = rgb.size()
        r_guidance = self.sa(rgb + depth)
        depth = depth + depth * r_guidance
        #print('rgb size:', rgb.size(), 'depth size:', depth.size())
        f_r = self.conv_r(rgb)
        f_d = self.conv_d(depth)
        #print('\nf_r size:', f_r.size(), 'f_d size:', f_d.size())

        f_global = torch.cat((f_r, f_d), 1)
        f_global = f_global + self.ca(f_global) * f_global
        f_global = self.conv1(f_global)

        edge = resize_match(f_r, edge)
        body = resize_match(f_d, body)

        f_r = f_r + f_r * edge
        f_d = f_d + f_d * body
        #print('\nf_global size:', f_global.size(), 'f_r size:', f_r.size(), 'f_d size:', f_d.size())

        # f_r = self.maxpool(self.cross_attention1(f_global, f_r))
        # f_d = self.avgpool(self.cross_attention2(f_global, f_d))

        f_rd_fuse = self.cross_attention(self.maxpool(f_r), self.avgpool(f_d))
        #print('\nconv_f_global size:', self.conv2(f_global).size(), 'f_r size:', f_r.size(), 'f_d size:', f_d.size())
        f_fuse = f_rd_fuse + self.conv2(f_global)
        
        y = self.conv3(self.upsample2(f_fuse))

        return y

class auxiliary_edge_generate(nn.Module):
    def __init__(self, in_channels, out_channels, decoder):
        super(auxiliary_edge_generate, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = decoder
        self.th = 0.2
    def forward(self, r_low, r_high, d_low, d_high):
        x = torch.cat((r_high, r_low, d_high, d_low), 1)
        x = self.upsample2(x)
        x = self.conv(x)
        x = self.maxpool(x)
        y = self.decoder(x)
        keep = torch.ones_like(y)
        igore = torch.zeros_like(y)
        th = self.th
        guidance = torch.where(torch.sigmoid(y) >= th, keep, igore)
        body = dilate(guidance)
        edge = body - guidance
        edge = dilate(edge)

        return edge

class auxiliary_body_generate(nn.Module):
    def __init__(self, in_channels, out_channels, decoder):
        super(auxiliary_body_generate, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BConv(in_channels, out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.decoder = decoder
        self.th = 0.2
    def forward(self, r_low, r_high, d_low, d_high):
        x = torch.cat((r_high, r_low, d_high, d_low), 1)
        x = self.upsample2(x)
        x = self.conv(x)
        x = self.avgpool(x)
        y = self.decoder(x)
        keep = torch.ones_like(y)
        igore = torch.zeros_like(y)
        th = self.th
        guidance = torch.where(torch.sigmoid(y) >= th, keep, igore)
        body = dilate(guidance)

        return body

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder = nn.Sequential(
            self.upsample2,
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # self.conv1 = BConv(32, 32, kernel_size=3)
        # self.conv2 = nn.Conv2d(32, 1, kernel_size=1)
        # self.dwconv1 = DWConv(32, 3, padding=1)
        # self.pwconv1 = PWConv(32, 32)
        # self.dwconv2 = DWConv(32, 7, padding=3)
        # self.pwconv2 = PWConv(32, 1)

    def forward(self, x):
        # x1 = self.conv1(self.upsample2(x))
        # x1 = self.conv2(self.upsample2(x1))

        # x2 = self.dwconv1(self.upsample2(x))
        # x2 = self.pwconv1(x2)
        # x2 = self.dwconv2(self.upsample2(x2))
        # x2 = self.pwconv2(x2)

        # return x1 + x2
    
        return self.decoder(x)

class vitxxs_attention(nn.Module):
    def __init__(self, input_isze, norm_layer=nn.LayerNorm):
        super(vitxxs_attention, self).__init__()

        self.input_size = input_isze

        # encoder
        self.rgb_vit_iterate = mobile_vit_xx_small()
        self.depth_vit_iterate = mobile_vit_xx_small()

        #decoder
        self.decoder = decoder()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.fusion_41 = fusion_scale_ca(144 // 2, 80)
        self.fusion_42 = fusion_scale_sa(144 // 2, 80)
        self.fusion_4 = fusion_modal(80, 64)

        self.fusion_31 = fusion_scale_ca(64, 64)
        self.fusion_32 = fusion_scale_sa(64, 64)
        self.fusion_3 = fusion_modal(64, 48)

        self.fusion_21 = fusion_scale_ca(48, 48)
        self.fusion_22 = fusion_scale_sa(48, 48)
        self.fusion_2 = fusion_modal(48, 24)

        self.fusion_11 = fusion_scale_ca(24, 24)
        self.fusion_12 = fusion_scale_sa(24, 24)
        self.fusion_1 = fusion_modal(24, 16)

        self.abg = auxiliary_body_generate(288, 16, self.decoder)
        self.aeg = auxiliary_edge_generate(144, 16, self.decoder)

        self.conv1 = BConv(24, 16)
        self.conv2 = BConv(48, 16)
        self.conv3 = BConv(64, 16)

    def forward(self, rgb, depth):
        rgb_list = self.rgb_vit_iterate(rgb)
        depth_list = self.depth_vit_iterate(depth)

        r4 = rgb_list[4]  # small:(8,64,96,96)  xxsmall:(8,24,96,96)--1
        r3 = rgb_list[3]  # small:(8,96,48,48)  xxsmall:(8,48,48,48)--2
        r2 = rgb_list[2]  # small:(8,128,24,24) xxsmall:(8,64,24,24)--3
        r1 = rgb_list[1]  # small:(8,160,12,12) xxsmall:(8,80,12,12)--4

        d4 = depth_list[4]  # small:(8,64,96,96)  xxsmall:(8,24,96,96)--1
        d3 = depth_list[3]  # small:(8,96,48,48)  xxsmall:(8,48,48,48)--2
        d2 = depth_list[2]  # small:(8,128,24,24) xxsmall:(8,64,24,24)--3
        d1 = depth_list[1]  # small:(8,160,12,12) xxsmall:(8,80,12,12)--4

        # print('r4 size:', r4.size(), 'r3 size:', r3.size(), 'r2 size:', r2.size(), 'r1 size:', r1.size())
        # print('d4 size:', d4.size(), 'd3 size:', d3.size(), 'd2 size:', d2.size(), 'd1 size:', d1.size())

        body = self.abg(r3, self.upsample2(r4), d3, self.upsample2(d4))
        edge = self.aeg(r1, self.upsample2(r2), d1, self.upsample2(d2))
        #print('edge_size:', edge.size(), 'body_size:', body.size())

        
        f_41 = self.fusion_41(self.upsample2(r4), r3)#first_type_fusion
        #print('f_41 size:', f_41.size())
        f_42 = self.fusion_42(self.upsample2(d4), d3)
        #print('f_42 size:', f_42.size())
        f_4 = self.fusion_4(f_41, f_42, edge, body)#second_type_fusion
        #print('f_4 size:', f_4.size())

        f_31 = self.fusion_31(r3, f_4)
        #print('f_31 size:', f_31.size())
        f_32 = self.fusion_32(d3, f_4)
        #print('f_32 size:', f_32.size())
        f_3 = self.fusion_3(f_31, f_32, edge, body)
        #print('f_3 size:', f_3.size())

        f_21 = self.fusion_21(r2, self.upsample2(f_3))
        #print('f_21 size:', f_21.size())
        f_22 = self.fusion_22(d2, self.upsample2(f_3))
        #print('f_22 size:', f_22.size())
        f_2 = self.fusion_2(f_21, f_22, edge, body)

        f_11 = self.fusion_11(r1, self.upsample2(f_2))
        #print('f_11 size:', f_11.size())
        f_12 = self.fusion_12(d1, self.upsample2(f_2))
        #print('f_12 size:', f_12.size())
        f_1 = self.fusion_1(f_11, f_12, edge, body)
        #print('f_1 size:', f_1.size())

        # print('\nf1 size:', f_1.size(), 'f2 size:', f_2.size(), 'f3 size:', f_3.size(), 'f4 size:', f_4.size())
        sailence_map_1, sailence_map_2, sailence_map_3, sailence_map_4 = self.decoder(f_1), self.decoder(self.conv1(f_2)), self.decoder(self.conv2(f_3)), self.decoder(self.conv3(f_4))
        #s1 needs the information of the fellow, s2 and s3 also need, use the cat to fuse them.

        sailence_map_2 = resize_match(sailence_map_1, sailence_map_2)
        sailence_map_3 = resize_match(sailence_map_1, sailence_map_3)
        sailence_map_4 = resize_match(sailence_map_1, sailence_map_4)

        return sailence_map_1, sailence_map_2, sailence_map_3, sailence_map_4
    
    def load_pre(self, pre_model):
        self.rgb_vit_iterate.load_state_dict(torch.load(pre_model),strict=False)
        print(f"RGB Mobile_ViT loading pre_model ${pre_model}")
        self.depth_vit_iterate.load_state_dict(torch.load(pre_model), strict=False)
        print(f"Depth Mobile_ViT loading pre_model ${pre_model}")