import torch
from torch import nn

class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = nn.Hardsigmoid()  
 
    def forward(self, x):
        r = self.conv(x)
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)
        r = scale.expand_as(r)*r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b,self.dim,-1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b,-1,h,w)        
        out = self.p(out) + r
        return out

class DynamicConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = False
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.p = nn.Conv1d()

        squeeze_dim = in_channels

        self.dynamic_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(in_channels, squeeze_dim)
        )

        self.se = nn.Sequential(
            nn.Linear(squeeze_dim, squeeze_dim, bias),
            nn.Hardsigmoid()
        )

        self.fc_phi = nn.Linear(squeeze_dim, self.dim**2, bias)
        self.fc_scale = nn.Sequential(
            nn.Linear(squeeze_dim, out_channels, bias),
            nn.Hardsigmoid()
        )
    
    def forward(self, x):
        r = self.conv(x)

        y = self.dynamic_branch(x)
        y = y * self.se(y)

        phi = self.fc_phi(y)
        scale = self.fc_scale(y)
        r = scale.expand_as(r) * r

        return w_0
