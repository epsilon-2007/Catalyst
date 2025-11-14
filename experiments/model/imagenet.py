import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import copy
import math
from torch import nn, Tensor
from functools import partial
from torch.hub import load_state_dict_from_url
from typing import Any, Callable, List, Optional

from .route import RouteDICE

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
}

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ResNet +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AbstractResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # main layers
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # weight initialization
    def _initial_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def encoder(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x # return a tensor of shape: [batch_size, dim_in, 7, 7])
    
class ResNet(AbstractResNet):
    def __init__(self, block, layers, args):
        super(ResNet, self).__init__(block, layers, args.num_classes)
        self._initial_weight()

        # classifier
        self.dim_in = 512 * block.expansion  # self.dim_in = self.inplanes
        if args.p is None:
            self.fc = nn.Linear(self.dim_in, args.num_classes)
        else:
            self.fc = RouteDICE(self.dim_in, args.num_classes, device=args.device, p=args.p, info=args.info)

    def intermediate_features(self, x):
        """
        Returns outputs from Layer: 1, Layer: 2, Layer: 3, and Layer: 4
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out1, out2, out3, out4

    def my_encoder(self, x):
        x = self.encoder(x)
        # print(x.shape)
        return x
    
    def my_features(self, x):
        x = self.my_encoder(x)
        x = self.avgpool(x)
        x = x.view(-1, self.dim_in)
        return x
    
    def forward(self, x):
        x = self.my_features(x)
        x = self.fc(x)
        return x

def resnet18(args, pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(args, pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(args, pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model  


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ MobileNet-V2 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#  Conv + BatchNorm + ReLU6
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

#  Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # Expansion phase
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        # depthwise convolution
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        # projection phase (linear, no ReLU)
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

#  MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # (t, c, n, s) -> expand_ratio, output_channels, num_blocks, stride
        inverted_residual_settings = [
            # exp, out, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # initial layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # inverted residual blocks
        for t, c, n, s in inverted_residual_settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # final convolution
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def encoder(self, x):
        x = self.features(x)
        return x # return a tensor of shape: [batch_size, dim_in, 7, 7])

    

class MyMobileNetV2(MobileNetV2):
    def __init__(self, args):
        super(MyMobileNetV2, self).__init__(num_classes=args.num_classes, width_mult=1.0)
        self._initialize_weights()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim_in = self.last_channel

        #classifier 
        if args.p is None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, args.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                RouteDICE(self.dim_in, args.num_classes, device=args.device, p=args.p, info=args.info)
            )

    def my_encoder(self, x):
        x = self.encoder(x)
        # print(x.shape)
        return x
    
    def my_features(self, x):
        x = self.my_encoder(x)
        x = self.avgpool(x)
        x = x.view(-1, self.dim_in)
        return x
    
    def forward(self, x):
        x = self.my_features(x)
        x = self.classifier(x)
        return x
    
def mobilenet_v2(args, pretrained=True):
    model = MyMobileNetV2(args=args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2']))
    return model



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ DenseNet-121 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from collections import OrderedDict

class DenseLayer(nn.Module):
    def __init__( self, input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer( input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x


class TransitionLayer(nn.Sequential):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)




class DenseNet(nn.Module):
    def __init__(self, growth_rate = 32, block_config = (6, 12, 24, 16), init_features = 64, bn_size = 4, drop_rate = 0.0, num_classes = 1000):
        super().__init__()
        # first convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # denseblock
        num_features = init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock( num_layers=num_layers, input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(input_features=num_features, output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # classification layer: batch norm + linear layer
        self.dim_in = num_features
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        # weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    
    def encoder(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        return x # return a tensor of shape: [batch_size, dim_in, 7, 7])
    
class MyDenseNet121(DenseNet):
    def __init__(self, args):
        super(MyDenseNet121, self).__init__(num_classes=args.num_classes)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # classifier
        if args.p is None:
            self.classifier = nn.Linear(self.dim_in, args.num_classes)
        else:
            self.classifier = RouteDICE(self.dim_in, args.num_classes, device=args.device, p=args.p, info=args.info)

        # initialize weights
        self._initialize_weights()

    def my_encoder(self, x):
        x = self.encoder(x)
        # print(x.shape)
        return x
    
    def my_features(self, x):
        x = self.my_encoder(x)
        x = self.avgpool(x)
        x = x.view(-1, self.dim_in)
        return x
    
    def forward(self, x):
        x = self.my_features(x)
        x = self.classifier(x)
        return x
    
def densenet121(args, pretrained = False):
    model = MyDenseNet121(args)
    if pretrained:
        import torchvision.models as models
        model = models.densenet121(pretrained=pretrained)
    return model



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EfficientNet-B0 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# ==============================================================================
# 1. official PyTorch vision helper modules
# ==============================================================================

# this is a simplified version of torchvision's ConvNormActivation
class ConvNormActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, groups: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d, 
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU, dilation: int = 1, inplace: Optional[bool] = None,) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=norm_layer is None)
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

# this is a simplified version of torchvision's SqueezeExcitation
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_channels: int, activation: Callable[..., nn.Module] = nn.ReLU, scale_activation: Callable[..., nn.Module] = nn.Sigmoid,) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

# this is a simplified version of torchvision's StochasticDepth
class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        
        survival_rate = 1.0 - self.p
        if self.mode == 'row':
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
            
        noise = torch.empty(size, dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return input * noise

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# ==============================================================================
# 2. official PyTorch efficientNet implementation
# ==============================================================================

class MBConvConfig:
    # stores information listed at table 1 of the efficientNet paper
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int, out_channels: int, num_layers: int, width_mult: float, depth_mult: float) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module], se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                             norm_layer=norm_layer, activation_layer=activation_layer))
        layers.append(ConvNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel,
                                         stride=cnf.stride, groups=expanded_channels,
                                         norm_layer=norm_layer, activation_layer=activation_layer))
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))
        layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                         activation_layer=None))
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(self, args, inverted_residual_setting: List[MBConvConfig], dropout: float, stochastic_depth_prob: float = 0.2, block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None, **kwargs: Any ) -> None:
        super().__init__()
        if block is None: block = MBConv
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        layers: List[nn.Module] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                         activation_layer=nn.SiLU))
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=nn.SiLU))
        self.features = nn.Sequential(*layers)


        self.dim_in = lastconv_output_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        if args.p is None:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(self.dim_in, args.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                RouteDICE(self.dim_in, args.num_classes, device=args.device, p=args.p, info=args.info),
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range); nn.init.zeros_(m.bias)

    def my_encoder(self, x:Tensor) -> Tensor:
        x = self.features(x)
        return x
    
    def my_features(self, x:Tensor) -> Tensor:
        x = self.my_encoder(x)
        x = self.avgpool(x)
        x = x.view(-1, self.dim_in)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.my_features(x)
        x = self.classifier(x)
        return x


def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: Any) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1), bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2), bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3), bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return inverted_residual_setting


def _efficientnet_model(args, arch: str, inverted_residual_setting: List[MBConvConfig], dropout: float, pretrained: bool, progress: bool, **kwargs: Any) -> EfficientNet:
    model = EfficientNet(args, inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def efficientnet_b0(args, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    return _efficientnet_model(args, "efficientnet_b0", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)
