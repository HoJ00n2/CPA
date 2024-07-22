import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora

from collections import OrderedDict
from utils import *
from torch.nn.modules.batchnorm import BatchNorm2d

torch.hub.set_dir('/cmlscratch/manlis/hub')

model_urls = {
    'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def convert_ckpt(model, weight, adv_key="adv_bn"):
    """
    Args:
        model (torch.nn.Module): a model with additional adv_bn layers
        weight (state_dict): the statedict without adv_bn layers
        adv_key (str): the keyword for adv_bn layers in the model
    """
    sd = model.state_dict()
    kv = weight

    values = []
    kv_keys = []
    for k, value in kv.items():
        kv_keys.append(k)
        values.append(value)

    nkv = OrderedDict()
    index = 0
    
    for k in sd:
        n = k.split('.')[1]
        if n == 0:
            n = 'Conv2d'
        elif n == 1:
            n = 'BatchNorm2d'
        elif n == 2:
            n = 'ReLU'
        elif n == 3:
            n = 'MaxPool2d'
        else:
            n = 'Sequential'
            
        if adv_key in k:
            # initialize the adv_bn layer using statistcs from its clean counterparts
            nkv[k] = nkv[k.replace(adv_key, adv_key.replace("adv", "clean"))]
        elif adv_key.replace("adv", "clean") in k or 'feature_x.1' in k or n=='BatchNorm2d': 
            #isinstance(getattr(model, '.'.join(k.split('.')[:-1])), BatchNorm2d) 
            # in case bn stats are stored in different order (wt/bias, mean/var vs. mean/var, wt/bias) 
            if ('weight' in k and 'weight' not in kv_keys[index]) or \
                ('bias' in k and 'bias' not in kv_keys[index]):
                nkv[k] = values[index+2]
                index += 1
            elif ('mean' in k and 'mean' not in kv_keys[index]) or \
                ('var' in k and 'var' not in kv_keys[index]):
                nkv[k] = values[index-2]
                index += 1
            elif "num_batches_tracked" in k and "num_batched_tracked" not in kv_keys[index]:
                # for converting models weights from older pytorch version
                nkv[k] = torch.BoolTensor([True])
            else:
                nkv[k] = values[index]
                index += 1
        elif "num_batches_tracked" in k and "num_batched_tracked" not in kv_keys[index]:
            # for converting models weights from older pytorch version
            nkv[k] = torch.BoolTensor([True])
        else:
            nkv[k] = values[index]
            index += 1

    return nkv

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, momentum=0.1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.clean_bn1 = norm_layer(planes, momentum=momentum)
        self.adv_bn1 = norm_layer(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.clean_bn2 = norm_layer(planes, momentum=momentum)
        self.adv_bn2 = norm_layer(planes, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, tag='clean'):
        identity = x

        out = self.conv1(x)
        if tag == 'clean':
            out = self.clean_bn1(out)
        else:
            out = self.adv_bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if tag == 'clean':
            out = self.clean_bn2(out)
        else:
            out = self.adv_bn2(out)

        if self.downsample is not None:
            identity = self.downsample.conv(x)
            if tag == 'clean':
                identity = self.downsample.clean_bn(identity)
            else:
                identity = self.downsample.adv_bn(identity)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, momentum=0.1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) #conv1x1(inplanes, width)
        self.clean_bn1 = norm_layer(planes, momentum=momentum) #norm_layer(width, momentum=momentum)
        self.adv_bn1 = norm_layer(planes, momentum=momentum) #norm_layer(width, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #conv3x3(width, width, stride, groups, dilation)
        self.clean_bn2 = norm_layer(planes, momentum=momentum) #norm_layer(width, momentum=momentum)
        self.adv_bn2 = norm_layer(planes, momentum=momentum) #norm_layer(width, momentum=momentum)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False) #conv1x1(width, planes * self.expansion)
        self.clean_bn3 = norm_layer(planes * self.expansion, momentum=momentum)
        self.adv_bn3 = norm_layer(planes * self.expansion, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, tag='clean'):
        identity = x

        out = self.conv1(x)
        if tag == 'clean':
            out = self.clean_bn1(out)
        else:
            out = self.adv_bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if tag == 'clean':
            out = self.clean_bn2(out)
        else:
            out = self.adv_bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if tag == 'clean':
            out = self.clean_bn3(out)
        else:
            out = self.adv_bn3(out)

        if self.downsample is not None:
            identity = self.downsample.conv(x)
            if tag == 'clean':
                identity = self.downsample.clean_bn(identity)
            else:
                identity = self.downsample.adv_bn(identity)

        out += identity
        out = self.relu(out)

        return out
        
class Head(nn.Module):
    def __init__(self, layers, cut, num_classes):
        super(Head, self).__init__()
        block_num = []
        for i, layer in enumerate(layers):
            block_num.append(len(layer))
            for j, block in enumerate(layer):
                setattr(self, 'layer{}_{}'.format(str(int(cut)+i+1), str(j)), block)
        self.layer_num = len(layers)
        self.block_num = block_num
        self.cut = cut
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x, tag='clean'):
        for i in range(self.layer_num):
            for j in range(self.block_num[i]):
                m = getattr(self, 'layer{}_{}'.format(str(int(self.cut)+i+1), str(j)))
                x = m(x, tag)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    '''
        Additional args:
            train_all (bool): Fix the first few layers during training if train_all==False. 
            cut (int): the index of layer that cut between 'feature extractor (g^{1, l})'(input -> layer{cut-1}) 
                       and 'downstream layers (g^{l+1, L})'(layer{cut} -> output)
    '''
    def __init__(self, block, layers, num_classes=1000, train_all=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, cut=2):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if num_classes == 1000:
            conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif num_classes == 10:
            conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        bn1 = norm_layer(self.inplanes)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer_list = []
        layer_list.append(self._make_layer(block, 64, layers[0], stride=1))
        layer_list.append(self._make_layer(block, 128, layers[1], stride=2,
                          dilate=replace_stride_with_dilation[0]))
        layer_list.append(self._make_layer(block, 256, layers[2], stride=2,
                          dilate=replace_stride_with_dilation[1]))
        layer_list.append(self._make_layer(block, 512, layers[3], stride=2,
                          dilate=replace_stride_with_dilation[2]))
        self.cut = cut

        # first few layers (g^{1, l}) for extracting features
        feature_x = []
        feature_x.append(conv1)
        feature_x.append(bn1)
        feature_x.append(relu)
        if num_classes == 1000:
            feature_x.append(maxpool)
        for i in range(self.cut):
            layer = layer_list[i]
            feature_x.append(nn.Sequential(*layer))
        self.feature_x = nn.Sequential(*feature_x)

        # fix the g^{1, l} when training with advbn
        if not train_all:
            for params in getattr(self, 'feature_x').parameters():
               params.requires_grad = False
            for m in self.feature_x.modules():
               if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                   m.eval()

        # downstream layers for deeper feature extraction and classification
        head = []
        if self.cut + 1 <= 4:
            for i in range(self.cut, 4):
                layer = layer_list[i]
                head.append(layer)
        self.head = Head(head, cut, num_classes)

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
                ('clean_bn', norm_layer(planes * block.expansion)),
                ('adv_bn', norm_layer(planes * block.expansion)),
            ]))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layers

    def _forward_impl(self, x, stage, tag='clean'):
        if stage == 'feature_x':
            x = self.feature_x(x)
        elif stage == 'head':
            x = self.head(x, tag)
        else:
            x = self.feature_x(x)
            x = self.head(x, tag)
        return x

    def forward(self, x, stage, tag='clean'):
        return self._forward_impl(x, stage, tag)

class BottleneckChen2020AdversarialNet(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckChen2020AdversarialNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        pre = F.relu(self.bn0(x))
        out = F.relu(self.bn1(self.conv1(pre)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        if len(self.shortcut) == 0:
            out += self.shortcut(x)
        else:
            out += self.shortcut(pre)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_shortcut=False):
        super(PreActBlock, self).__init__()
        self.out_shortcut = out_shortcut
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out if self.out_shortcut else x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBlockV2(nn.Module):
    '''Pre-activation version of the BasicBlock (slightly different forward pass)'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_shortcut=False):
        super(PreActBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn_before_fc=False, out_shortcut=False):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.bn_before_fc = bn_before_fc
        self.out_shortcut = out_shortcut
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if bn_before_fc:
            self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, out_shortcut=self.out_shortcut))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.bn_before_fc:
            out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def _ResNet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        kv = torch.hub.load_state_dict_from_url(model_urls[arch])
        nkv = convert_ckpt(model, kv, 'adv_bn')
        model.load_state_dict(nkv)
    return model

def ResNet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ResNet('ResNet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def ResNet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ResNet('ResNet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ResNet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ResNet('ResNet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ResNet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ResNet('ResNet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def ResNet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ResNet('ResNet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
