import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SELayer(nn.Module):
    def __init__(self, channel, reduction = 8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            pass
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv6 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        # self.cspn2_guide = GMLayer(in_channels)
        # self.cspn2 = Affinity_Propagate_Channel()
        self.se1 = SELayer(in_channels)
        self.se2 = SELayer(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # guidance2 = self.cspn2_guide(x3)
        # x3_2 = self.cspn2(guidance2, x3)
        x3_2 = self.se1(x)
        x4 = self.conv4(torch.cat((x3, x3_2), 1))
        x5 = self.conv5(torch.cat((x2, x4), 1))
        x6 = self.conv6(torch.cat((x1, x5), 1))+self.se2(x3_2)
        return x6


class MFormer(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=200, n_DRBs=8):
        super(MFormer, self).__init__()
        # 2D Nets
        num_feature = channels
        self.reshape0 = nn.Sequential(
            Conv3x3(inplanes, 31, 3, 1),
            nn.PReLU(),
            Conv3x3(31, 31, 3, 1)
        )
        self.mask = nn.Sequential(
            random_masking2(0.3),
            Conv3x3(31, 31, 3, 1),
            nn.PReLU(),
            Conv3x3(31, 31, 3, 1)
        )

        self.reshape1 = nn.Sequential(
            Conv3x3(31, channels, 3, 1),
            nn.PReLU(),
            Conv3x3(channels, channels, 3, 1)
        )

        self.TE0 = Transformer_E(channels,2,3,channels//3,channels)
        self.SE0 = SELayer(channels)

        self.down_sample_1 = nn.Conv2d(channels,channels*2,4,2,1,bias=False)
        self.TE1 = Transformer_E(channels*2,2,3,channels*2//3,channels*2)
        self.up_sample_1 = nn.ConvTranspose2d(channels*2,channels,4,2,1)

        self.down_sample_2 = nn.Conv2d(channels*2,channels*4,4,2,1,bias=False)
        self.TE2 = Transformer_E(channels*4,2,3,channels*4//3,channels*4)
        self.up_sample_2 = nn.ConvTranspose2d(channels*4,channels*2,4,2,1)
        self.reshape2 = nn.Sequential(
            Conv3x3(channels*3,channels,3,1),
            nn.PReLU(),
            Conv3x3(channels,channels,3,1)
        ) 

        self.TD1 = Transformer_D(channels,2,3,channels//3,channels)
        self.TD2 = Transformer_D(channels,2,3,channels//3,channels)


        self.refine1 = nn.Sequential(
            Conv3x3(channels,channels,3,1),
            nn.PReLU(),
            Conv3x3(channels, channels, 3, 1)
        )
        self.reshape3 = nn.Sequential(
            Conv3x3(channels, channels, 3, 1),
            nn.PReLU(),
            Conv3x3(channels, planes, 3, 1)
        )


    def forward(self, x):

        out = self.DRN2D(x)
        return out

    def DRN2D(self, x):

        out = self.reshape0(x)
        before = out.clone()
        out = self.mask(out)
        out = before + 0.1*out 

        after = out.clone() # 用于画特征图2
        out = self.reshape1(out)

        y_0 = out  
        y_1 = self.down_sample_1(y_0)  
        y_2 = self.down_sample_2(y_1)  

        out_2 = self.TE2(y_2)
        out_2 = self.up_sample_2(out_2)
        con1 = self.up_sample_1(out_2)

        y_1 = y_1 + out_2
        out_1 = self.TE1(y_1)
        out_1 = self.up_sample_1(out_1)
        con0 = out_1
        
        y_0 = self.SE0(y_0) + out_1
        out_0 = self.TE0(y_0)
        out = out + out_0

        out = torch.cat((con1, con0, out), 1)
        out = self.reshape2(out)
        
        out = self.TD1(out)
        out = self.TD2(out)

        out = out + y_0

        out = self.refine1(out)
        out = self.reshape3(out)

        return out,before,after


class Conv2D(nn.Module):
    def __init__(self, in_channel=256, out_channel=8):
        super(Conv2D, self).__init__()
        self.guide_conv2D = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        spatial_guidance = self.guide_conv2D(x)
        return spatial_guidance



# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # b, c, h, w = x.shape
        # x = x.reshape(b,c,h*w)
        x = self.net(x)
        # x = x.reshape(b,c,h,w)
        return x

# Light transformer
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        # self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1)) # 这是为了后面sigmoid之后与v相乘的尺度匹配
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.dim = dim
        self.convv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape

        x = x_in.reshape(b,h*w,c)
        
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        v_inp1 = self.convv(v_inp.reshape(b,h,w,c).permute(0,3,1,2))
        v_inp1 = v_inp.reshape(b,h*w,c)
        v_inp += v_inp1

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        
        q = q.transpose(-3, -2) 
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        v_h = F.normalize(v, dim=-1, p=2)

        attn_head = torch.matmul(v_h,v_h.transpose(-2,-1)) 
        attn_head = attn_head.softmax(dim=-1)
        
        q = torch.matmul(attn_head,q)
        k = torch.matmul(attn_head,k)

        attn = torch.matmul(k.transpose(-2, -1),q) 
        attn = attn.softmax(dim=-1)
        x = torch.matmul(v, attn) 
        x = x.permute(0, 3, 1, 2)    
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        
        out_c = self.proj(x).view(b, h, w, c)
        out = out_c 

        return out


class Transformer_E(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Transformer_D(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        x = x.permute(0, 2, 3, 1)
        for attn1,attn2, ff in self.layers:
            
            x = attn1(x)
            x = attn2(x)
            x = ff(x)
        x = x.permute(0, 3, 1, 2)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class random_masking2(nn.Module):

    def __init__(self,mask_ratio): 
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, input1):
        b,c,h,w = input1.shape
        x = input1.reshape(b,c,h*w)
        N, L, D = x.shape  

        x_out = self.pic_channel(x,L,self.mask_ratio)
        x_masked = x_out.reshape(b,c,h,w)           
        return x_masked

    def pic_channel(self,x,channel,mask_ratio):
        num = int(channel*mask_ratio)

        mask_band = []
        for j in range(num):
            mask_band.append(random.randint(0,30))
            
        for i in range(channel):
            if i in mask_band:
                x[:,i,:] += torch.abs(torch.randn(x.shape[0],x.shape[2])).cuda()

        return x

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_tensor = torch.rand(1, 3, 64, 64)
    
    model = MFormer(3, 31, 60, 8)
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(torch.__version__)