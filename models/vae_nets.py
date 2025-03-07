from torch.autograd import Variable
import functools
import math
import torch
import torch.nn as nn

# /VAE-Cycle-GAN/ from https://github.com/xr-Yang/CycleGAN-VAE-for-reid/
class _Sampler(nn.Module):
    def __init__(self,gpu_id=0):
        super(_Sampler, self).__init__()
        self.gpu_id=gpu_id
        self.kl=0

    def forward(self, input):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        #if opt.cuda:
        eps = torch.FloatTensor(std.size()).normal_().to(device=self.gpu_id)  # random normalized noise
        #else:
            #eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise, normal_(mean=0, std=1, *, generator=None)
        eps = Variable(eps)
        self.kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return eps.mul(std).add_(mu)  # z = mu + std*epsilon ~ N(mu, std)


class _Encoder(nn.Module):
    def __init__(self, imageSizeLog, ngf, nz, nc, norm_layer):
        """VAE Encoder constructor, build the encoder network as a convolutional neural network, 
        with batch-normalization and re-lu for all the intermediate layers, only relu for the first
        convolutional layer while last layer is composed of 2 convolutional layers, one for mu and one for variance.
        

        Args:
            imageSizeLog (int): log2 of image size in pixels
            ngf (int): number of filters of the last conv layer
            nz (int): latent space dimension
            nc (int): number of channels of the input image
        """
        super(_Encoder, self).__init__()
        self.ngf = ngf
        self.nc = nc
        self.nz = nz
        
        # Last 2 convolutions, from a number of channels of ngf * 2 ** (imageSizeLog - 3) to nz (latent dimension),
        # with a 4*4 filter (16 pixels) applied on a 4*4 image after the encoder's convolutions.
        self.conv1 = nn.Conv2d(ngf * 2 ** (imageSizeLog - 3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2 ** (imageSizeLog - 3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x w **2
        self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(imageSizeLog - 3):  # i= 0, 1, ..., imageSizeLog-2
            # state size 
            self.encoder.add_module('conv-{}'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)), nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('batchnorm-{}'.format(ngf * 2 ** (i + 1)), norm_layer(ngf * 2 ** (i + 1)))
            self.encoder.add_module('relu-{}'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*2**(imageSizeLog-2)) x 4 x 4

    def forward(self, input):
        output = self.encoder(input)
        return [self.conv1(output), self.conv2(output)]

class _Decoder(nn.Module):
    def __init__(self, imageSizeLog, ngf, nz, nc, norm_layer=nn.BatchNorm2d):
        """VAE Decoder constructor, build the decoder network as a convolutional neural network, 
        with batch-normalization and re-lu for all the layers except the last one, which is a
        convolutional layer with tanh (Hyperbolic Tangent) activation function.
        

        Args:
            imageSizeLog (int): log2 of image size in pixels
            ngf (int): number of filters of the last conv layer
            nz (int): latent space dimension
            nc (int): number of channels of the input image
        """
        super(_Decoder, self).__init__()
        self.ngf = ngf
        self.nc = nc
        self.nz = nz
        
        self.decoder = nn.Sequential()  # the G network of DCGAN, input: noise vector Z, output: N x 3 x 64 x 64
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (imageSizeLog - 3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', norm_layer(ngf * 2 ** (imageSizeLog - 3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(imageSizeLog - 3, 0, -1):  # i = n-3, n-2, ..., 1
            self.decoder.add_module('conv-{}'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i,
                                                       ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
            self.decoder.add_module('batchnorm-{}'.format(ngf * 2 ** (i - 1)),
                                    norm_layer(ngf * 2 ** (i - 1)))
            self.decoder.add_module('relu-{}'.format(ngf * 2 ** (i - 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())
        
    def forward(self, input):
        return self.decoder(input)

class VAEGenerator(nn.Module):
    def __init__(self, nc, ngf, imageSize, gpu_ids, nz, norm_layer=nn.BatchNorm2d):
        """VAE constructor, build the VAE network as a sequence of the encoder and the decoder networks,
        using a sampling in Normal distribution to generate the latent space z.

        Args:
            imageSize (int): image size in pixels
            ngf (int): number of filters of the last conv layer
            nz (int): latent space dimension
            ngpu (int): number of GPUs to use
            nc (int): number of channels of the input image
            norm_layer: normalization layer
        """
        super(VAEGenerator, self).__init__()
        
        n = math.log2(imageSize)
        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.ngpu = len(gpu_ids)
        self.encoder = _Encoder(n, ngf, nz, nc, norm_layer)
        self.sampler = _Sampler(gpu_ids[0])
        self.decoder = _Decoder(n, ngf, nz, nc, norm_layer)


    def forward(self, input):
        output = self.encoder(input)
        output = self.sampler(output)
        output = self.decoder(output)
        return output

    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()
        
# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, zsize, layers=None, in_channels=3, img_size=128, norm_layer=nn.BatchNorm2d):
        super(VAE, self).__init__()
        self.d = img_size
        log_d= math.log2(img_size)
        self.zsize = zsize
        self.norm_layer=norm_layer
        if not layers: # filters (representi features) for each layer
            self.layers=[64,128,256,512]
        else:
            self.layers=layers
        
        # the convolution of the encoder, with a kernel=4, stride=2, padding=1
        # has the effect of reducing the size of the image by a factor of 2
        # so we need to ensure that the number of layers is consistent with the size of the image
        self.d_enc=2**int(log_d-len(self.layers))
        
        assert self.d_enc>2, "The number of layers is not consistent with the size of the image"
        
        self.encoder= nn.Sequential()
        inputs = in_channels
        for i in range(len(self.layers)):
            self.encoder.add_module("conv%d" % (i + 1), nn.Conv2d(inputs, self.layers[i], 4, 2, 1))
            self.encoder.add_module("conv%d_bn" % (i + 1), self.norm_layer(self.layers[i]))
            self.encoder.add_module("conv%d_relu" % (i + 1), nn.LeakyReLU(0.2, inplace=True)) 
            inputs = self.layers[i]
        
        self.fc1 = nn.Linear(inputs * self.d_enc**2, zsize)
        self.fc2 = nn.Linear(inputs * self.d_enc**2, zsize)

        self.d1 = nn.Linear(zsize, inputs * self.d_enc**2)

        self.decoder= nn.Sequential()
        self.layers.reverse()
        for i in range(1, len(self.layers)):
            self.decoder.add_module("deconv%d" % (i), nn.ConvTranspose2d(inputs, self.layers[i], 4, 2, 1))
            self.decoder.add_module("deconv%d_bn" % (i), self.norm_layer(self.layers[i]))
            self.decoder.add_module("deconv%d_relu" % (i), nn.LeakyReLU(0.2, inplace=True))
            inputs = self.layers[i]
        self.decoder.add_module("deconv%d" % (len(self.layers)), nn.ConvTranspose2d(inputs, in_channels, 4, 2, 1))
        self.decoder.add_module("deconv%d_tanh" % (len(self.layers)), nn.Tanh())
        self.layers.reverse()

    def encode(self, x):
        x=self.encoder.forward(x)
        x = x.view(x.shape[0], self.layers[-1] * self.d_enc**2)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.layers[-1], self.d_enc, self.d_enc)
        x= self.decoder.forward(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def make_cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()
        
"""    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()"""
        


def get_norm_layer(channels, norm_type="batch"):
    if norm_type == "batch":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    It does a pipeline of Activation Function, Convolution and Normalization first
    a with stride of 2, that downsamples the image by a factor of 2, and then again
    with stride 1. In between, it does a skip connection within the first and the second half of the volume.
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_out // 2) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    It starts with an up-sampling layer with a scale of 2,
    followed by a pipeline of Activation Function, Convolution and Normalization
    with stride 1, that keep the image size stable. 
    In between, it does a skip connection within the first and the second half of the volume.
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_in // 2) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    """
    "Normal" Residual block - the convolutions here keep the same W*H size of the image
    but increase the number of channels.
    The forward of this block does:
    - {Normalization, Activation function (ELU), Convolution}
    - Skip connection (or identity if channel_in == channel_out, otherwise within the first and the second half of the volume)
    - {Normalization, Activation function (ELU), Convolution}
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = channel_in // 2 if channel_in == channel_out else (channel_in // 2) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Encoder, self).__init__()
        
        # Convolution keeping the image size W*H
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks) # [1,2,4,8]
        widths_out = list(blocks[1:]) + [2 * blocks[-1]] # [2,4,8,16]

        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional "Normal" residual block before down-sampling
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))
            # Add a residual down-sampling block
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]

        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))

        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        return torch.tanh(self.conv_out(x))


class RESVAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in=3, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(RESVAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        ch = number of filters of the first ResBlock layer, multiplied by blocks for later layers
        blocks = multiplicative factor for each ResBlock for up-down sampling
        num_res_blocks = number of pure residual blocks (with skip connections) in the encoder and decoder
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var

# convolutional final dimension= (W-F+2P)/S +1

########################################################################################
########################################################################################

class ResnetVAEGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', nz=256, gpu_ids=[]):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            nz (int)            -- the size of the latent z vector
        """
        assert(n_blocks >= 0)
        super(ResnetVAEGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # First conv layer does padding, then conv, then normalization, then ReLU. 
        # This does not modify the image size
        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers, each time the image size is divided by 2
            mult = 2 ** i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks//2):       # add ResNet blocks

            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        self.encoder = nn.Sequential(*encoder)
        self.conv_mu = nn.Conv2d(ngf * mult, nz, 1)
        self.conv_logvar = nn.Conv2d(ngf * mult, nz, 1)
        
        decoder=[nn.Conv2d(nz, ngf * mult, 1)]
        
        for i in range(n_blocks//2):       # add the other half of ResNet blocks

            decoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]
        
        self.decoder= nn.Sequential(*decoder)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input):
        """Standard forward"""
        x = self.encoder(input)
        mu = self.conv_mu(x)
        log_var = self.conv_logvar(x)
        if self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return self.decoder(z), mu, log_var

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out