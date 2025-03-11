from torch.autograd import Variable
import functools
import math
import torch
import torch.nn as nn

# =======================================================================================================
# ==================================ConVAE===========================================================
# =======================================================================================================
# =======================================================================================================

class VAE(nn.Module):
    def __init__(self, zsize, layers=None, in_channels=3, img_size=128, norm_layer=nn.BatchNorm2d):
        super(VAE, self).__init__()
        self.img_size = img_size
        log_imsize= math.log2(img_size)
        self.latent_dim = zsize
        self.norm_layer=norm_layer
        if not layers: # n° filters for each layer
            self.layers=[64,128,256,512]
        else:
            self.layers=layers
        
        # the convolution of the encoder, with a kernel=4, stride=2, padding=1
        # has the effect of reducing the size of the image by a factor of 2
        # so we need to ensure that the number of layers is consistent with the size of the image
        
        self.size_imenc=2**int(log_imsize-len(self.layers))
        
        assert self.size_imenc>2, "The number of layers is not consistent with the size of the image"
        
        self.encoder= nn.Sequential()
        
        inputs = in_channels
        for i in range(len(self.layers)):
            self.encoder.add_module("conv%d" % (i + 1), nn.Conv2d(inputs, self.layers[i], 4, 2, 1))
            self.encoder.add_module("conv%d_bn" % (i + 1), self.norm_layer(self.layers[i]))
            self.encoder.add_module("conv%d_act" % (i + 1), nn.ELU(0.2, inplace=True)) 
            inputs = self.layers[i]
        
        self.fc1 = nn.Linear(inputs * self.size_imenc**2, zsize)
        self.fc2 = nn.Linear(inputs * self.size_imenc**2, zsize)

        self.d1 = nn.Linear(zsize, inputs * self.size_imenc**2)

        self.decoder= nn.Sequential()
        self.layers.reverse()
        for i in range(1, len(self.layers)):
            self.decoder.add_module("deconv%d" % (i), nn.ConvTranspose2d(inputs, self.layers[i], 4, 2, 1))
            self.decoder.add_module("deconv%d_bn" % (i), self.norm_layer(self.layers[i]))
            self.decoder.add_module("deconv%d_act" % (i), nn.ELU(0.2, inplace=True))
            inputs = self.layers[i]
        self.decoder.add_module("deconv%d" % (len(self.layers)), nn.ConvTranspose2d(inputs, in_channels, 4, 2, 1))
        self.decoder.add_module("deconv%d_tanh" % (len(self.layers)), nn.Tanh())
        self.layers.reverse()

    def encode(self, x):
        x=self.encoder.forward(x)
        x = x.view(x.shape[0], self.layers[-1] * self.size_imenc**2)
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
        x = x.view(x.shape[0], self.latent_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.layers[-1], self.size_imenc, self.size_imenc)
        x= self.decoder.forward(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.latent_dim, 1, 1)), mu, logvar

# /VAE-Cycle-GAN/ from https://github.com/xr-Yang/CycleGAN-VAE-for-reid/
# =======================================================================================================
# ==================================VAEGAN===========================================================
# =======================================================================================================
# =======================================================================================================

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
        for i in range(imageSizeLog - 3):  # i= 0, 1, ..., imageSizeLog-4
            # state size 
            self.encoder.add_module('conv-{}'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)), nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('batchnorm-{}'.format(ngf * 2 ** (i + 1)), norm_layer(ngf * 2 ** (i + 1)))
            self.encoder.add_module('relu-{}'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*2**(imageSizeLog-3)) x 4 x 4

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

# =======================================================================================================
# ==================================ResnetKVAE===========================================================
# =======================================================================================================
# =======================================================================================================

class ResnetVAEGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', nz=64):
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

        n_downsampling = 3
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

# convolution final dimension= (W-F+2P)/S +1