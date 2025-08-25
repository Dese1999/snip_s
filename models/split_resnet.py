import torch
import torch.nn as nn
from models.builder import get_builder
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel  #


    def forward(self, x):
        b, c, _, _ = x.size()
        assert c == self.channel, f"Input channels ({c}) do not match expected channels ({self.channel})"
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


    def compute_average_attention(self, dataloader):
        device = next(self.parameters()).device
        cumulative_a = torch.zeros(self.fc[2].out_features).to(device)
        count = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                b, c, _, _ = inputs.size()
                assert c == self.channel, f"Input channels ({c}) do not match expected channels ({self.channel})"
                x = self.avg_pool(inputs).view(b, c)
                y = self.fc(x).view(b, c)
                cumulative_a += y.mean(dim=0)
                count += b
        average_a = cumulative_a / count if count > 0 else cumulative_a
        return average_a


class BasicBlock(nn.Module):
    M = 2
    expansion = 1


    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64, slim_factor=1):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")


        self.conv1 = builder.conv3x3(int(inplanes * slim_factor), int(planes * slim_factor), stride)
        self.bn1 = builder.batchnorm(int(planes * slim_factor))
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(int(planes * slim_factor), int(planes * slim_factor))
        self.bn2 = builder.batchnorm(int(planes * slim_factor), last_bn=True)
        self.se = SEBlock(int(planes * slim_factor))  # اضافه شده
        self.relu2 = builder.activation()
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.se(out)  # اضافه شده
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class BasicBlockNorm(nn.Module):
    M = 2
    expansion = 1


    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64, slim_factor=1, LW_norm=False):
        super(BasicBlockNorm, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")


        self.conv1 = builder.conv3x3(int(inplanes * slim_factor), int(planes * slim_factor), stride)
        self.bn1 = builder.batchnorm(int(planes * slim_factor))
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(int(planes * slim_factor), int(planes * slim_factor))
        self.bn2 = builder.batchnorm(int(planes * slim_factor), last_bn=True)
        self.se = SEBlock(int(planes * slim_factor))  # اضافه شده
        self.relu2 = builder.activation()
        self.downsample = downsample
        self.stride = stride
        self.norm = nn.BatchNorm2d(int(planes * slim_factor))
        self.LW_norm = LW_norm


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.se(out)  # اضافه شده
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        if self.LW_norm:
            out = self.norm(out)
        return out


class Bottleneck(nn.Module):
    M = 3
    expansion = 4


    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64, slim_factor=1, is_last_conv=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64))
        self.conv1 = builder.conv1x1(int(inplanes * slim_factor), int(width * slim_factor))
        self.bn1 = builder.batchnorm(int(width * slim_factor))
        self.conv2 = builder.conv3x3(int(width * slim_factor), int(width * slim_factor), stride=stride)
        self.bn2 = builder.batchnorm(int(width * slim_factor))
        self.conv3 = builder.conv1x1(int(width * slim_factor), int(planes * self.expansion * slim_factor))
        self.bn3 = builder.batchnorm(int(planes * self.expansion * slim_factor))
        self.se = SEBlock(int(planes * self.expansion * slim_factor))  # اضافه شده
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)  # اضافه شده
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckNorm(nn.Module):
    M = 3
    expansion = 4


    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64, slim_factor=1, is_last_conv=False, LW_norm=False):
        super(BottleneckNorm, self).__init__()
        width = int(planes * (base_width / 64))
        self.conv1 = builder.conv1x1(int(inplanes * slim_factor), int(width * slim_factor))
        self.bn1 = builder.batchnorm(int(width * slim_factor))
        self.conv2 = builder.conv3x3(int(width * slim_factor), int(width * slim_factor), stride=stride)
        self.bn2 = builder.batchnorm(int(width * slim_factor))
        self.conv3 = builder.conv1x1(int(width * slim_factor), int(planes * self.expansion * slim_factor))
        self.bn3 = builder.batchnorm(int(planes * self.expansion * slim_factor))
        self.se = SEBlock(int(planes * self.expansion * slim_factor))  # اضافه شده
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride
        self.norm = nn.BatchNorm2d(int(planes * self.expansion * slim_factor))
        self.LW_norm = LW_norm


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)  # اضافه شده
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.LW_norm:
            out = self.norm(out)
        return out


class ResNet(nn.Module):
    def __init__(self, cfg, builder, block, layers, base_width=64):
        super(ResNet, self).__init__()
        self.inplanes = 64
        slim_factor = cfg.slim_factor
        if slim_factor < 1:
            cfg.logger.info('WARNING: You are using a slim network')


        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")
        self.last_layer = cfg.last_layer


        self.conv1 = builder.conv7x7(3, int(64 * slim_factor), stride=2, first_layer=True)
        self.bn1 = builder.batchnorm(int(64 * slim_factor))
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0], slim_factor=slim_factor)
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2, slim_factor=slim_factor)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2, slim_factor=slim_factor)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2, slim_factor=slim_factor)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.linear(int(512 * block.expansion * slim_factor), cfg.num_cls, last_layer=True)


    def _make_layer(self, builder, block, planes, blocks, stride=1, slim_factor=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(int(self.inplanes * slim_factor), int(planes * block.expansion * slim_factor), stride=stride)
            dbn = builder.batchnorm(int(planes * block.expansion * slim_factor))
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv


        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width, slim_factor=slim_factor))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width, slim_factor=slim_factor))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.last_layer:
            x = self.fc(x)
        return x


    def get_params(self):
        return torch.cat([p.view(-1) for p in self.parameters()])


    def set_params(self, new_params):
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in self.parameters():
            cand_params = new_params[progress: progress + torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params


    def get_grads(self):
        grads = []
        for pp in self.parameters():
            if pp.grad is None:
                grads.append(torch.zeros(pp.shape).view(-1).to(pp.device))
            else:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)


    def get_grads_list(self):
        return [pp.grad.view(-1) if pp.grad is not None else torch.zeros_like(pp).view(-1) for pp in self.parameters()]


def Split_ResNet18(cfg, progress=True):
    model = ResNet(cfg, get_builder(cfg), BasicBlock, [2, 2, 2, 2])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet18'
        print('loading pretrained resnet')
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model


def Split_ResNet18Norm(cfg, progress=True):
    model = ResNet(cfg, get_builder(cfg), BasicBlockNorm, [2, 2, 2, 2])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet18'
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model


def Split_ResNet34(cfg, progress=True):
    model = ResNet(cfg, get_builder(cfg), BasicBlock, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet34'
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model


def Split_ResNet50(cfg, progress=True):
    model = ResNet(cfg, get_builder(cfg), Bottleneck, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet50'
        print('loading pretrained resnet')
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model


def Split_ResNet50Norm(cfg, progress=True):
    model = ResNet(cfg, get_builder(cfg), BottleneckNorm, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet50'
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model


def Split_ResNet101(cfg, progress=True):
    model = ResNet(cfg, get_builder(cfg), Bottleneck, [3, 4, 23, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet101'
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model



