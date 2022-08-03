import torch
import torch.nn as nn
import torch.nn.functional as F


# 基本卷积模块
class Conv_Layer(nn.Module):
    def __init__(self, in_channles, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 Activation_required=True
                 ):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Conv2d(
                in_channels=in_channles, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        self.relu = nn.ReLU(inplace=True)
        self.Activation_required = Activation_required

    def forward(self, x):
        out = self.conv(x)
        if self.Activation_required:
            out = self.relu(out)
        return out


# 注意力模块
class Attention_Module(nn.Module):
    def __init__(self):
        super(Attention_Module, self).__init__()
        self.model = nn.Sequential(
            Conv_Layer(in_channles=128, out_channels=64),
            Conv_Layer(in_channles=64, out_channels=64, Activation_required=False),
            nn.Sigmoid()
        )

    def forward(self, Zi, Zr):
        x = torch.cat([Zi, Zr], dim=1)
        out = self.model(x)
        return out


# 注意力网络
class Attention_Network(nn.Module):
    def __init__(self):
        super(Attention_Network, self).__init__()
        self.encoder = Conv_Layer(in_channles=6, out_channels=64)
        self.Attention_1r = Attention_Module()
        self.Attention_3r = Attention_Module()

    def forward(self, X1, X2, X3):
        Z1 = self.encoder(X1)
        Zr = self.encoder(X2)
        Z3 = self.encoder(X3)

        A1 = self.Attention_1r(Z1, Zr)
        A3 = self.Attention_3r(Z3, Zr)

        Z1_apo = Z1 * A1
        Z3_apo = Z3 * A3

        Zs = torch.cat([Z1_apo, Zr, Z3_apo], dim=1)

        return Zs, Zr


# 扩张密集块 DDB - Dilated Dense Block
class DDB(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DDB, self).__init__()
        self.Dilated_conv = Conv_Layer(
            in_channles=in_channels, out_channels=growth_rate,
            padding=2, dilation=2,
        )

    def forward(self, x):
        out = self.Dilated_conv(x)
        return torch.cat([x, out], dim=1)


# 扩张残差密集块 DRDB - Dilated Residual Dense Block
class DRDB(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, Layer_num=6):
        super(DRDB, self).__init__()
        Current_Channel_num = in_channels

        modules = []
        for i in range(Layer_num):
            modules.append(DDB(in_channels=Current_Channel_num, growth_rate=growth_rate))
            Current_Channel_num += growth_rate
        self.dense_layers = nn.Sequential(*modules)

        self.conv_1x1 = Conv_Layer(
            in_channles=Current_Channel_num, out_channels=in_channels,
            kernel_size=1, padding=0,
        )

    def forward(self, x):
        out1 = self.dense_layers(x)
        out2 = self.conv_1x1(out1)

        return out2 + x


# 用于 HDR 图像估计的合并网络Merging_Network
class Merging_Network(nn.Module):
    def __init__(self):
        super(Merging_Network, self).__init__()
        self.conv1 = Conv_Layer(in_channles=64 * 3, out_channels=64)
        self.DRDB1 = DRDB()
        self.DRDB2 = DRDB()
        self.DRDB3 = DRDB()
        self.conv2 = Conv_Layer(in_channles=64 * 3, out_channels=64)
        self.conv3 = Conv_Layer(in_channles=64, out_channels=64)
        self.conv4 = Conv_Layer(in_channles=64, out_channels=3, Activation_required=False)
        self.tanh = nn.Tanh()

    def forward(self, Zs, Zr):
        F0 = self.conv1(Zs)
        F1 = self.DRDB1(F0)
        F2 = self.DRDB2(F1)
        F3 = self.DRDB3(F2)

        F4 = torch.cat([F1, F2, F3], dim=1)

        F5 = self.conv2(F4)
        F6 = self.conv3(F5 + Zr)
        F7 = self.conv4(F6)

        return self.tanh(F7) * 0.5 + 0.5


# AHDRNet
class AHDRNet(nn.Module):
    def __init__(self):
        super(AHDRNet, self).__init__()
        self.A = Attention_Network()
        self.M = Merging_Network()

    def forward(self, X1, X2, X3):
        Zs, Zr = self.A(X1, X2, X3)
        out = self.M(Zs, Zr)
        return out


from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x1 = torch.randn(1, 6, 1000, 1500).to(device)
    x2 = torch.randn(1, 6, 1000, 1500).to(device)
    x3 = torch.randn(1, 6, 1000, 1500).to(device)

    model = AHDRNet().to(device)
    with torch.no_grad():
        out = model(x1, x2, x3)

    print(model)
    print(out.shape)
    print(out.max(), out.min())

    print("-AHDRNet构建完成，参数量为： {} ".format(sum(x.numel() for x in model.parameters())))
    # -AHDRNet构建完成，参数量为： 1281347

    # writer = SummaryWriter('./logs/AHDRNet')
    # writer.add_graph(model, [x1, x2, x3])
    # writer.close()

