import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
from .plugins import *
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3, plugins=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)
        
        self.plugins = plugins
        self.with_plugins = plugins is not None
        
        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if (plugin['position'] == 'after_conv2' or plugin['position'] == 'after_conv3')
            ]
        
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
        
    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv1_plugin_names)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv2_plugin_names)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3, plugins=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.plugins = plugins
        
        self.stride = stride
        
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2','after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]
        
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * 4, self.after_conv3_plugins)
        self.downsample = downsample

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv1_plugin_names)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv2_plugin_names)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv3_plugin_names)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3], plugins=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        stages_plugins_inchannels = [self.inplanes, 64*block.expansion, 128*block.expansion, 256*block.expansion, 512*block.expansion]
        if plugins is not None:
            stage_blocks_plugins = []
            self.stages_plugins = []
            for i in range(5):
                stage_block_plugins, stage_plugins = self.make_stage_plugins(plugins, i)
                stage_blocks_plugins.append(stage_block_plugins)   
                self.stages_plugins.append(self.make_block_plugins(stages_plugins_inchannels[i], stage_plugins, 'stage' + str(i)))            
        else:
            stage_blocks_plugins = [None for i in range(5)]
            self.stages_plugins = [None for i in range(5)]

        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]), plugins=stage_blocks_plugins[0])
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2, plugins=stage_blocks_plugins[1])
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2, plugins=stage_blocks_plugins[2])
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2, plugins=stage_blocks_plugins[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1, plugins=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size, plugins=plugins))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size, plugins=plugins))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.stages_plugins[0] is not None and len(self.stages_plugins[0])>0: 
            x = self.forward_plugin(x, self.stages_plugins[0])

        x = self.maxpool(x)

        x = self.layer1(x)
        if self.stages_plugins[1] is not None and len(self.stages_plugins[1])>0: 
            x = self.forward_plugin(x, self.stages_plugins[1])

        x = self.layer2(x)
        if self.stages_plugins[2] is not None and len(self.stages_plugins[2])>0: 
            x = self.forward_plugin(x, self.stages_plugins[2])

        x = self.layer3(x)
        if self.stages_plugins[3] is not None and len(self.stages_plugins[3])>0: 
            x = self.forward_plugin(x, self.stages_plugins[3])

        x = self.layer4(x)
        if self.stages_plugins[4] is not None and len(self.stages_plugins[4])>0:
            x = self.forward_plugin(x, self.stages_plugins[4])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_block_plugins = []
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            position = plugin.get('position', None)
            if position == 'stage_output':
                stages = plugin.pop('stages', None)
                # whether to insert plugin into current stage
                if stages is not None and stage_idx<len(stages) and stages[stage_idx]:
                    stage_plugins.append(plugin['cfg'])
            else:
                stages = plugin.pop('stages', None)
                assert stages is None or len(stages) >=4
                # whether to insert plugin into current stage
                if stages is None or stages[stage_idx]:
                    stage_block_plugins.append(plugin)

        return stage_block_plugins, stage_plugins
    
    def make_block_plugins(self, in_channels, plugins, postfix = ''):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=postfix)
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names
    
    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out


def resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False, plugins=None):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False, plugins=None):
    """Constructs a ResNet-34 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False, plugins=None):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing attention_resnet50......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
def resnet50_ablation(k_size=[3, 3, 3, 3], num_classes=250, pretrained=False, plugins=None):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing attention_resnet50......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
def resnet26_ablation(k_size=[3, 3, 3, 3], num_classes=250, pretrained=False, plugins=None):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing attention_resnet50......")
    model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
def resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False, plugins=None):
    """Constructs a ResNet-101 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False, plugins=None):
    """Constructs a ResNet-152 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size, plugins=plugins)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
