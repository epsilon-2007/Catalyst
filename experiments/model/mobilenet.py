import torch.nn as nn
from .route import RouteDICE

class InvertedResidual(nn.Module):
    """building block of the MobileNetV2 architecture."""
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expansion_factor
        
        # residual connection is only possible if stride is 1 and dimensions match.
        self.use_residual = self.stride == 1 and in_channels == out_channels

        layers = []
        # add the expansion layer (1x1 Conv) if expansion_factor > 1
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # add the main depthwise and projection layers
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False), # depthwise convolution
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),  # projection convolution (linear)
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # (expansion_factor, out_channels, num_repeats, stride)
        self.config = [
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # build the inverted residual blocks
        self.blocks = nn.ModuleList()
        in_channels = 32
        for t, c, n, s in self.config:
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(InvertedResidual(in_channels, c, stride, t))
                in_channels = c
        
        # final layers before the classifier
        self.dim_in = 512
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.dim_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.dim_in),
            nn.ReLU6(inplace=True)
        )

    def my_encoder(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x # returns a tensor of [batch_size, dim_in, 4, 4]
    
class MyMobileNetV2(MobileNetV2):
    def __init__(self, args):
        super(MyMobileNetV2, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # classifier
        if args.p is None:
            self.output_layer = nn.Linear(self.dim_in, args.num_classes)
        else:
            self.output_layer = RouteDICE(self.dim_in, args.num_classes, device=args.device, p=args.p, info=args.info)

    def my_features(self, x):
        x = self.my_encoder(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(-1, self.dim_in)
        return x

    def forward(self, x):
        x = self.my_features(x)
        x = self.output_layer(x)
        return x