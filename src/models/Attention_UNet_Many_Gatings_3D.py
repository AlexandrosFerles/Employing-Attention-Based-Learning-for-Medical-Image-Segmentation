from network_components import *

class Attention_UNet(nn.Module):
    def __init__(self, num_channels, num_classes, filters, output_activation='sigmoid'):
        super(Attention_UNet, self).__init__()

        self.output_activation = output_activation.lower()

        self.pool = nn.MaxPool3d(2)

        self.inp = double_convolution3D(num_channels, filters)

        self.downsampling_1 = double_convolution3D(filters, filters*2)
        self.downsampling_2 = double_convolution3D(filters*2, filters*4)
        self.downsampling_3 = double_convolution3D(filters*4, filters*8)

        self.att3 = GridAttentionGateLocal3D(Fg=filters*8, Fl=filters*4, Fint=filters*4)
        self.att2 = GridAttentionGateLocal3D(Fg=filters*4, Fl=filters*2, Fint=filters*2)
        self.att1 = GridAttentionGateLocal3D(Fg=filters*2, Fl=filters, Fint=filters)

        self.upsample3 = upsampling_block3D(filters*8, filters*4)
        self.upsampling_convolution3 = double_convolution3D(filters*8, filters*4)
        self.upsample2 = upsampling_block3D(filters*4, filters*2)
        self.upsampling_convolution2 = double_convolution3D(filters*4, filters*2)
        self.upsample1 = upsampling_block3D(filters*2, filters)
        self.upsampling_convolution1 = double_convolution3D(filters*2, filters)

        self.output = nn.Conv3d(filters, num_classes, kernel_size=1, stride=1, padding=0)

        # for m in self.modules():
        #     initialize_weights(m)

    def forward(self, x):

        # downsampling path
        inp = self.inp(x)

        d1 = self.pool(inp)
        d1 = self.downsampling_1(d1)
        d2 = self.pool(d1)
        d2 = self.downsampling_2(d2)
        d3 = self.pool(d2)
        d3 = self.downsampling_3(d3)

        # attention coefficients + upsampling path
        g3 = self.att3(d2, d3)
        u3 = self.upsample3(d3)
        cat3 = torch.cat((g3, u3), dim=1)
        up3 = self.upsampling_convolution3(cat3)

        g2 = self.att2(d1, up3)
        u2 = self.upsample2(up3)
        cat2 = torch.cat((g2, u2), dim=1)
        up2 = self.upsampling_convolution2(cat2)

        g1 = self.att1(inp, up2)
        u1 = self.upsample1(up2)
        cat1 = torch.cat((g1, u1), dim=1)
        up1 = self.upsampling_convolution1(cat1)

        out = self.output(up1)

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.output_activation == 'softmax':
            import torch.nn.functional as F
            return F.softmax(out, dim=1)
        else:
            raise NotImplementedError('Unknown output activation function')

