import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)


def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
  for y in ys:
      x = x - proj(x, y)
  return x


def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


class LinearBlock(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        norm='none', 
        act='relu', 
        use_sn=False
    ):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(
        self, 
        nf_in, 
        nf_out, 
        nf_mlp, 
        num_blocks, 
        norm, 
        act, 
        use_sn =False
    ):
        super(MLP,self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm = norm, act = act, use_sn = use_sn))
        for _ in range((num_blocks - 2)):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act ='none', use_sn = use_sn))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
    
    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
    
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
    
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                          kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
    
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d,which_bn= nn.BatchNorm2d, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv,self.which_bn =which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    
    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlock2(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, activation=None,
                upsample=None, skip_connection = True):
        super(GBlock2, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv = which_conv
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                          kernel_size=1, padding=0)
        # upsample layers
        self.upsample = upsample
        self.skip_connection = skip_connection

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        
        h = self.activation(h)
        h = self.conv2(h)
        
        if self.learnable_sc:
            x = self.conv_sc(x)
        if self.skip_connection:
            out = h + x
        else:
            out = h
        return out


def style_encoder_textedit_addskip_arch(ch =64,out_channel_multiplier = 1, input_nc = 3):
    arch = {}
    n=2
    arch[96] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [48,24,12,6,3]}

    arch[128] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [64,32,16,8,4]}
    
    arch[256] = {'in_channels':[input_nc]+[ch*item for item in [1,2,4,8,8]],
                                'out_channels':[item*ch for item in [1,2,4,8,8,16]],
                                'resolution': [128,64,32,16,8,4]}
    return arch


class StyleEncoder(ModelMixin, ConfigMixin):
    """
    This class is to encode the style image to image embedding.
    Downsample scale is 32.
    For example:
        Input: Shape[Batch, 3, 128, 128]
        Output: Shape[Batch, 255, 4, 4]
    """
    @register_to_config
    def __init__(
        self, 
        G_ch=64, 
        G_wide=True, 
        resolution=128,
        G_kernel_size=3, 
        G_attn='64_32_16_8', 
        n_classes=1000,
        num_G_SVs=1, 
        num_G_SV_itrs=1, 
        G_activation=nn.ReLU(inplace=False),
        SN_eps=1e-12, 
        output_dim=1,  
        G_fp16=False,
        G_init='N02',  
        G_param='SN', 
        nf_mlp = 512, 
        nEmbedding = 256, 
        input_nc = 3,
        output_nc = 3
    ):
        super(StyleEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16

        if self.resolution == 96:
            self.save_featrues = [0,1,2,3,4]
        if self.resolution == 128:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 256:
            self.save_featrues = [0,1,2,3,4,5]
        
        self.out_channel_nultipiler = 1
        self.arch = style_encoder_textedit_addskip_arch(
          self.ch, 
          self.out_channel_nultipiler,
          input_nc
        )[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(
              SNConv2d,
              kernel_size=3, padding=1,
              num_svs=num_G_SVs, 
              num_itrs=num_G_SV_itrs,
              eps=self.SN_eps
            )
            self.which_linear = functools.partial(
              SNLinear,
              num_svs=num_G_SVs, 
              num_itrs=num_G_SV_itrs,
              eps=self.SN_eps
            )
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):

            self.blocks += [[DBlock(
              in_channels=self.arch['in_channels'][index],
              out_channels=self.arch['out_channels'][index],
              which_conv=self.which_conv,
              wide=self.G_wide,
              activation=self.activation,
              preactivation=(index > 0),
              downsample=nn.AvgPool2d(2)
            )]]

        self.blocks = nn.ModuleList([
          nn.ModuleList(block) for block in self.blocks
        ])
        last_layer = nn.Sequential(
          nn.InstanceNorm2d(self.arch['out_channels'][-1]),
          self.activation,
          nn.Conv2d(
            self.arch['out_channels'][-1],
            self.arch['out_channels'][-1],
            kernel_size=1,
            stride=1
          )
        )
        self.blocks.append(last_layer)
        self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self,x):        
        h = x
        residual_features = []
        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)            
            if index in self.save_featrues[:-1]:
                residual_features.append(h)        
        h = self.blocks[-1](h)
        style_emd = h        
        h = F.adaptive_avg_pool2d(h,(1,1))
        h = h.view(h.size(0),-1)
        
        return style_emd,h,residual_features
