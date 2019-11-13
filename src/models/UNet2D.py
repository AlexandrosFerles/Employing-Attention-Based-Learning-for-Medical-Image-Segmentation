import torch
import torch.nn as nn

from network_components import double_convolution, upsampling_block

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes, filters, output_activation='sigmoid'):
        super(UNet, self).__init__()

        self.output_activation = output_activation.lower()

        self.pool = nn.MaxPool2d(2)

        self.inp = double_convolution(num_channels, filters)

        self.downsampling_1 = double_convolution(filters, filters*2)
        self.downsampling_2 = double_convolution(filters*2, filters*4)
        self.downsampling_3 = double_convolution(filters*4, filters*8)
        self.downsampling_4 = double_convolution(filters*8, filters*16)

        self.upsample4 = upsampling_block(filters*16, filters*8)
        self.upsampling_convolution4 = double_convolution(filters*16, filters*8)
        self.upsample3 = upsampling_block(filters*8, filters*4)
        self.upsampling_convolution3 = double_convolution(filters*8, filters*4)
        self.upsample2 = upsampling_block(filters*4, filters*2)
        self.upsampling_convolution2 = double_convolution(filters*4, filters*2)
        self.upsample1 = upsampling_block(filters*2, filters)
        self.upsampling_convolution1 = double_convolution(filters*2, filters)

        self.output = nn.Conv2d(filters, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        inp = self.inp(x)

        # downsampling path
        d1 = self.pool(inp)
        d1 = self.downsampling_1(d1)
        d2 = self.pool(d1)
        d2 = self.downsampling_2(d2)
        d3 = self.pool(d2)
        d3 = self.downsampling_3(d3)
        d4 = self.pool(d3)
        d4 = self.downsampling_4(d4)

        # upsampling path
        u4 = self.upsample4(d4)
        cat4 = torch.cat((d3, u4), dim=1)
        up4 = self.upsampling_convolution4(cat4)
        u3 = self.upsample3(up4)
        cat3 = torch.cat((d2, u3), dim=1)
        up3 = self.upsampling_convolution3(cat3)
        u2 = self.upsample2(up3)
        cat2 = torch.cat((d1, u2), dim=1)
        up2 = self.upsampling_convolution2(cat2)
        u1 = self.upsample1(up2)
        cat1 = torch.cat((inp, u1), dim=1)
        up1 = self.upsampling_convolution1(cat1)

        out = self.output(up1)

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.output_activation == 'softmax':
            import torch.nn.functional as F
            return F.softmax(out, dim=1)
        else:
            raise NotImplementedError('Unknown output activation function')
