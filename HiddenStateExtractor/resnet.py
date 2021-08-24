from collections import OrderedDict
import torchvision.models as models
import torch
from torch import nn
from HiddenStateExtractor.losses import AllTripletMiner

class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers, layer_planes=(64, 128, 256, 512), num_inputs=2, cifar_head=False):
        """
        Args:
            block (nn.Module): block to build the network
            layers (list): number to repeat each block
            num_inputs (int): number of input channels
            cifar_head (bool): Use modified network for cifar-10 data if True
        """
        super().__init__(block, layers)
        self.inplanes = layer_planes[0]
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(num_inputs, layer_planes[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(num_inputs, layer_planes[0], kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(layer_planes[0])
        self.layer1 = self._make_layer(block, layer_planes[0], layers[0])
        self.layer2 = self._make_layer(block, layer_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, layer_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, layer_planes[3], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNet18(ResNetEncoder):
    def __init__(self, num_inputs=2, cifar_head=True, width=1):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], layer_planes=[2 ** (x + width) for x in range(5, 9)],
                         num_inputs=num_inputs, cifar_head=cifar_head)

class ResNet50(ResNetEncoder):
    def __init__(self, num_inputs=2, cifar_head=True, width=1):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], layer_planes=[2 ** (x + width) for x in range(5, 9)],
                         num_inputs=num_inputs, cifar_head=cifar_head)

class ResNet101(ResNetEncoder):
    def __init__(self, num_inputs=2, cifar_head=True, width=1):
        super().__init__(models.resnet.Bottleneck, [3, 4, 23, 3], layer_planes=[2 ** (x + width) for x in range(5, 9)],
                         num_inputs=num_inputs, cifar_head=cifar_head)

class ResNet152(ResNetEncoder):
    def __init__(self, num_inputs=2, cifar_head=True, width=1):
        super().__init__(models.resnet.Bottleneck, [3, 8, 36, 3], layer_planes=[2 ** (x + width) for x in range(5, 9)],
                         num_inputs=num_inputs, cifar_head=cifar_head)

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self,
                 arch='ResNet50',
                 width=1,
                 loss=AllTripletMiner(margin=1),
                 num_inputs=2,
                 cifar_head=False,
                 device='cuda:0'):

        super().__init__()

        if arch == 'ResNet50':
            self.convnet = ResNet50(num_inputs=num_inputs, cifar_head=cifar_head, width=width)
            self.encoder_dim = 2048 * width
        elif arch == 'ResNet101':
            self.convnet = ResNet101(num_inputs=num_inputs, cifar_head=cifar_head, width=width)
            self.encoder_dim = 2048 * width
        elif arch == 'ResNet152':
            self.convnet = ResNet152(num_inputs=num_inputs, cifar_head=cifar_head, width=width)
            self.encoder_dim = 2048 * width
        elif arch == 'ResNet18':
            self.convnet = ResNet18(num_inputs=num_inputs, cifar_head=cifar_head, width=width)
            self.encoder_dim = 512 * width
        else:
            raise NotImplementedError

        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)

        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))
        self.loss = loss
        self.device = device

    def encode(self, x, out='z'):
        h = self.convnet(x)
        if out == 'h':
            return h
        elif out == 'z':
            z = self.projection(h)
            return z
        else:
            raise ValueError('"out" can only be "h" or "z", not {}'.format(out))


    def forward(self, x, labels=None, time_matching_mat=None, batch_mask=None):
        z = self.encode(x)
        loss, f_pos_tri = self.loss(labels, z)
        loss_dict = {'total_loss': loss, 'positive_triplet': f_pos_tri}
        return z, loss_dict

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, n_class, device='cuda:0'):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_class)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.device = device

    def forward(self, x, labels=None, time_matching_mat=None, batch_mask=None):
        z = self.linear(x)
        loss = nn.functional.cross_entropy(z, labels)
        torch.nn.CrossEntropyLoss()
        acc = (z.argmax(1) == labels).float().mean()
        loss_dict = {'total_loss': loss, 'acc': acc}
        return z, loss_dict