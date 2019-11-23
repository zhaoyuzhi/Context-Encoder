import torch
import torch.nn as nn
from network_module import *

# ----------------------------------------
#                Generator
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottleneck
        self.B1 = Conv2dLayer(opt.start_channels * 8, opt.bottleneck_channels, 4, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.bottleneck_channels, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, activation = 'relu', scale_factor = 4)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, activation = 'relu', scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, activation = 'relu', scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, activation = 'relu', scale_factor = 2)
        self.D5 = TransposeConv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, activation = 'relu', scale_factor = 2)
        self.D6 = TransposeConv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'tanh', scale_factor = 2)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decodernet = Generator()
        x = self.E1(x)                                          # out: batch * 64 * 64 * 64
        x = self.E2(x)                                          # out: batch * 64 * 32 * 32
        x = self.E3(x)                                          # out: batch * 128 * 16 * 16
        x = self.E4(x)                                          # out: batch * 256 * 8 * 8
        x = self.E5(x)                                          # out: batch * 512 * 4 * 4
        # Bottleneck
        x = self.B1(x)                                          # out: batch * 4000 * 1 * 1
        # Decode the center code
        x = self.D1(x)                                          # out: batch * 512 * 4 * 4
        x = self.D2(x)                                          # out: batch * 256 * 8 * 8
        x = self.D3(x)                                          # out: batch * 128 * 16 * 16
        x = self.D4(x)                                          # out: batch * 64 * 32 * 32
        x = self.D5(x)                                          # out: batch * 64 * 32 * 32
        x = self.D6(x)                                          # out: batch * 3 * 64 * 64

        return x

# ----------------------------------------
#       AdversarialDiscriminator
# ----------------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.out_channels, 64, 4, 2, 1, pad_type = opt.pad, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(64, 128, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(128, 256, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(256, 512, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)

    def forward(self, x):
        x = self.block1(x)                                      # out: batch * 64 * 32 * 32
        x = self.block2(x)                                      # out: batch * 128 * 16 * 16
        x = self.block3(x)                                      # out: batch * 256 * 8 * 8
        x = self.block4(x)                                      # out: batch * 512 * 4 * 4
        return x
