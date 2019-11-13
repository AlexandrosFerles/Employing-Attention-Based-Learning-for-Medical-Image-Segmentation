import torch
import torch.nn as nn
import torch.nn.functional as F


########### CONVENTIONAL U-NET COMPONENTS ###########
class double_convolution(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(double_convolution, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class upsampling_block(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(upsampling_block, self).__init__()

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, up):

        upsampled = self.upsampling(up)
        return self.up_conv(upsampled)


class double_convolution3D(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(double_convolution3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class upsampling_block3D(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(upsampling_block3D, self).__init__()

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up_conv = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, up):

        upsampled = self.upsampling(up)
        return self.up_conv(upsampled)



########### ATTENTION U-NET COMPONENTS ###########
class GatingSignal(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, batchnorm=True):
        super(GatingSignal, self).__init__()

        if batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class GridAttentionGateGlobal(nn.Module):

    def __init__(self, Fg, Fl, Fint, learn_upsampling=False, batchnorm=False):
        super(GridAttentionGateGlobal, self).__init__()

        self.learn_upsampling = learn_upsampling
        self.Fint = Fint

        if batchnorm:
            self.Wg = nn.Sequential(
                nn.Conv2d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(Fint)
            )
            self.Wx = nn.Sequential(
                nn.Conv2d(Fl, Fint, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(Fint)
            )

            self.y = nn.Sequential(
                nn.Conv2d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1)
            )

        else:
            self.Wg = nn.Conv2d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True)
            self.Wx = nn.Conv2d(Fl, Fint, kernel_size=2, stride=2, padding=0, bias=False)

            self.y = nn.Conv2d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=Fl, out_channels=Fl, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(Fl),
        )

    def forward(self, xl, g):

        xl_size_orig = xl.size()
        g_size = g.size()

        xl_ = self.Wx(xl)
        xl_size = xl_.size()

        g = self.Wg(g)

        if self.learn_upsampling:
            # TODO: update padding value
            upsampled_g = nn.ConvTranspose2d(self.Fint, self.Fint, kernel_size=3, stride=xl_size[2] // g_size[2])
        else:
            upsampled_g = F.interpolate(g, size=xl_size[2:], mode='bilinear', align_corners=False)

        relu = F.relu(xl_ + upsampled_g, inplace=True)
        y = self.y(relu)
        sigmoid = torch.sigmoid(y)

        upsampled_sigmoid = F.interpolate(sigmoid, size=xl_size_orig[2:], mode='bilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_sigmoid.expand_as(xl)

        return self.out(attention * xl)


########### ATTENTION U-NET COMPONENTS FOR MULTIPLE GATING ###########
class GridAttentionGateLocal(nn.Module):

    def __init__(self, Fg, Fl, Fint, learn_upsampling=False, batchnorm=False):
        super(GridAttentionGateLocal, self).__init__()

        if batchnorm:
            self.Wg = nn.Sequential(
                nn.Conv2d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(Fint)
            )
            self.Wx = nn.Sequential(
                nn.Conv2d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(Fint),
                nn.MaxPool2d(2)
            )

            self.y = nn.Sequential(
                nn.Conv2d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1)
            )

        else:
            self.Wg = nn.Conv2d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True)
            self.Wx = nn.Sequential(
                nn.Conv2d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(2)
            )

            self.y = nn.Conv2d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=Fl, out_channels=Fl, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(Fl),
        )

    def forward(self, xl, g):

        xl_size_orig = xl.size()
        xl_ = self.Wx(xl)

        g = self.Wg(g)

        relu = F.relu(xl_ + g, inplace=True)
        y = self.y(relu)
        sigmoid = torch.sigmoid(y)

        upsampled_sigmoid = F.interpolate(sigmoid, size=xl_size_orig[2:], mode='bilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_sigmoid.expand_as(xl)

        return self.out(attention * xl)


class GridAttentionGateLocal3D(nn.Module):

    def __init__(self, Fg, Fl, Fint, learn_upsampling=False, batchnorm=False):
        super(GridAttentionGateLocal3D, self).__init__()

        if batchnorm:
            self.Wg = nn.Sequential(
                nn.Conv3d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(Fint)
            )
            self.Wx = nn.Sequential(
                nn.Conv3d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(Fint),
                nn.MaxPool3d(2)
            )

            self.y = nn.Sequential(
                nn.Conv3d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(1)
            )

        else:
            self.Wg = nn.Conv3d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True)
            self.Wx = nn.Sequential(
                nn.Conv3d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.MaxPool3d(2)
            )

            self.y = nn.Conv3d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.out = nn.Sequential(
            nn.Conv3d(in_channels=Fl, out_channels=Fl, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(Fl),
        )

    def forward(self, xl, g):

        xl_size_orig = xl.size()
        xl_ = self.Wx(xl)

        g = self.Wg(g)

        relu = F.relu(xl_ + g, inplace=True)
        y = self.y(relu)
        sigmoid = torch.sigmoid(y)

        upsampled_sigmoid = F.interpolate(sigmoid, size=xl_size_orig[2:], mode='trilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_sigmoid.expand_as(xl)

        return self.out(attention * xl)
