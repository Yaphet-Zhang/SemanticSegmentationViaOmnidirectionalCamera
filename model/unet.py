import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # [4, 3, 480, 480]
        c1 = self.conv1(x)
        # [4, 64, 480, 480]
        p1 = self.pool1(c1)
        # [4, 64, 240, 240]
        c2 = self.conv2(p1)
        # [4, 128, 240, 240]
        p2 = self.pool2(c2)
        # [4, 128, 120, 120]
        c3 = self.conv3(p2)
        # [4, 256, 120, 120]
        p3 = self.pool3(c3)
        # [4, 256, 60, 60]
        c4 = self.conv4(p3)
        # [4, 512, 60, 60]
        p4 = self.pool4(c4)
        # [4, 512, 30, 30]
        c5 = self.conv5(p4)
        # [4, 1024, 30, 30]

        up_6 = self.up6(c5)
        # [4, 512, 60, 60]
        merge6 = torch.cat([up_6, c4], dim=1)
        # [4, 1024, 60, 60]
        c6 = self.conv6(merge6)
        # [4, 512, 60, 60]
        up_7 = self.up7(c6)
        # [4, 256, 120, 120]
        merge7 = torch.cat([up_7, c3], dim=1)
        # [4, 512, 120, 120]
        c7 = self.conv7(merge7)
        # [4, 256, 120, 120]
        up_8 = self.up8(c7)
        # [4, 128, 240, 240]
        merge8 = torch.cat([up_8, c2], dim=1)
        # [4, 256, 240, 240]
        c8 = self.conv8(merge8)
        # [4, 128, 240, 240]
        up_9 = self.up9(c8)
        # [4, 64, 480, 480]
        merge9 = torch.cat([up_9, c1], dim=1)
        # [4, 128, 480, 480]
        c9 = self.conv9(merge9)
        # [4, 64, 480, 480]
        c10 = self.conv10(c9)
        # [4, 2, 480, 480]
        out = nn.Sigmoid()(c10)
        # [4, 2, 480, 480]
        return out


class UNet_VISUAL(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet_VISUAL, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # row 1
        # [4, 3, 480, 480]
        c1 = self.conv1(x)
        # [4, 64, 480, 480]
        p1 = self.pool1(c1)
        # [4, 64, 240, 240]
        c2 = self.conv2(p1)
        # [4, 128, 240, 240]
        p2 = self.pool2(c2)
        # [4, 128, 120, 120]
        c3 = self.conv3(p2)
        # [4, 256, 120, 120]
        p3 = self.pool3(c3)
        # [4, 256, 60, 60]
        c4 = self.conv4(p3)
        # [4, 512, 60, 60]
        p4 = self.pool4(c4)
        # [4, 512, 30, 30]
        c5 = self.conv5(p4)
        # [4, 1024, 30, 30]

        # 2 row
        up_6 = self.up6(c5)
        # [4, 512, 60, 60]
        merge6 = torch.cat([up_6, c4], dim=1)
        # [4, 1024, 60, 60]
        c6 = self.conv6(merge6)
        # [4, 512, 60, 60]
        up_7 = self.up7(c6)
        # [4, 256, 120, 120]
        merge7 = torch.cat([up_7, c3], dim=1)
        # [4, 512, 120, 120]
        c7 = self.conv7(merge7)
        # [4, 256, 120, 120]
        up_8 = self.up8(c7)
        # [4, 128, 240, 240]
        merge8 = torch.cat([up_8, c2], dim=1)
        # [4, 256, 240, 240]
        c8 = self.conv8(merge8)
        # [4, 128, 240, 240]
        up_9 = self.up9(c8)
        # [4, 64, 480, 480]
        merge9 = torch.cat([up_9, c1], dim=1)
        # [4, 128, 480, 480]
        c9 = self.conv9(merge9)
        # [4, 64, 480, 480]
        c10 = self.conv10(c9)
        # [4, 2, 480, 480]
        out = nn.Sigmoid()(c10)
        # [4, 2, 480, 480]
        return out



if __name__=='__main__':

    net = UNet(in_ch=3 , out_ch=2)
    print(net)
    print('====================')

    x = torch.randn(4, 3, 480, 480)
    print('x: {}'.format(x.shape))
    y = net(x)
    print('y: {}'.format(y.shape))
    print('====================')

    print('parameters numbers:', sum(param.numel() for param in net.parameters()))
    print('====================')

