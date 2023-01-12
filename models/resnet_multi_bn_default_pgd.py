import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from pdb import set_trace


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


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
                 base_width=64, dilation=1, norm_layer=None, bn_names=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batch_norm_multiple(norm_layer, planes, bn_names=bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batch_norm_multiple(norm_layer, planes, bn_names=bn_names)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = x[0]
        bn_name = x[1]

        # debug
        # print("bn_name: {}".format(bn_name))

        out = self.conv1(out)
        out = self.bn1([out, bn_name])

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2([out, bn_name])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity[0]
        out = self.relu(out)

        return [out, bn_name]


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_names=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = batch_norm_multiple(norm_layer, width, bn_names=bn_names)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = batch_norm_multiple(norm_layer, width, bn_names=bn_names)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = batch_norm_multiple(norm_layer, planes * self.expansion, bn_names=bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = x[0]
        bn_name = x[1]

        out = self.conv1(out)
        out = self.bn1([out, bn_name])

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2([out, bn_name])

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3([out, bn_name])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity[0]
        out = self.relu(out)

        return [out, bn_name]


class Downsample_multiple(nn.Module):
    def __init__(self, inplanes, planes, expansion, stride, norm_layer, bn_names=None):
        super(Downsample_multiple, self).__init__()
        self.conv = conv1x1(inplanes, planes * expansion, stride)
        self.bn = batch_norm_multiple(norm_layer, planes * expansion, bn_names=bn_names)

    def forward(self, x):
        out = x[0]
        bn_name = x[1]
        # debug
        # print("adv attack: {}".format(flag_adv))
        # print("out is {}".format(out))

        out = self.conv(out)
        out = self.bn([out, bn_name])

        return [out, bn_name]


class batch_norm_multiple(nn.Module):
    def __init__(self, norm, inplanes, bn_names=None):
        super(batch_norm_multiple, self).__init__()

        self.norm = norm

        # if no bn name input, by default use single bn
        self.bn_names = bn_names
        if self.bn_names is None:
            self.bn_list = norm(inplanes)
            return

        len_bn_names = len(bn_names)
        if norm == nn.GroupNorm:
            self.bn_list = nn.ModuleList([norm(32, inplanes) for _ in range(len_bn_names)])                
        elif norm == "dual_bn_oct_clean":         
            self.bn_list = nn.ModuleList([CrossTrainingBN_clean(inplanes), nn.BatchNorm2d(inplanes)])
        else:
            self.bn_list = nn.ModuleList([norm(inplanes) for _ in range(len_bn_names)])
        
        self.bn_names_dict = {bn_name: i for i, bn_name in enumerate(bn_names)}

        if self.norm == Disentangling_LP:
            self.bn_layer = Disentangling_LP(inplanes)

        if self.norm == Disentangling_StatP:
            self.bn_layer = Disentangling_StatP(inplanes)
        
        if self.norm == CrossTraining_DualBN_swap:
            self.bn_layer = CrossTraining_DualBN_swap(inplanes)
        

    def forward(self, x):
        out = x[0]
        name_bn = x[1]

        # debug
        # name_bn = "normal"

        if self.norm == Disentangling_LP:
            out = self.bn_layer(out, name_bn)
            return out
        
        if self.norm == Disentangling_StatP:
            out = self.bn_layer(out, name_bn)
            return out
        
        if self.norm == CrossTraining_DualBN_swap:
            out = self.bn_layer(out, name_bn)
            return out
        

        if self.norm == BN2dStrongerAT or self.norm == BN2dStrongerATMomentum1:
            if "AE_generate" in name_bn:
                bn_index = self.bn_names_dict["pgd"]

            else:
                bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out, name_bn)
            return out


        if name_bn is None:
            out = self.bn_list(out)
        else:
            bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, bn_names=["pgd", "normal"], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, vq_in=False):
        """
        :param bn_names: list, the name of bn that would be employed
        """

        super(ResNet, self).__init__()
        self.vq_in = vq_in
        if self.vq_in:
            n_embeddings = 512
            embedding_dim = 3
            commitment_cost = 0.25
            decay = 0.99
            self.vq_in = VectorQuantizerEMA(n_embeddings, embedding_dim, commitment_cost, decay)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # self.normalize = NormalizeByChannelMeanStd(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.inplanes = 64
        self.dilation = 1
        self.bn_names = bn_names

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                        bias=False)
        
        self.bn1 = batch_norm_multiple(norm_layer, self.inplanes, bn_names=self.bn_names)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.Identity()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], bn_names=self.bn_names)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):            
            #     import ipdb; ipdb.set_trace()
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        self.forward_bn_name = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bn_names=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample_multiple(self.inplanes, planes, block.expansion, stride, norm_layer, bn_names=bn_names)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bn_names=bn_names))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bn_names=bn_names))

        return nn.Sequential(*layers)

    # def _forward_impl(self, x, bn_name=None):
    def _forward_impl(self, x, bn_name=None):


        # debug
        # print("bn name: {}".format(bn_name))
        # import ipdb; ipdb.set_trace()

        # for auto attack. we need to change the default forward bn name, since we can not modify the original AA algorithm
        if self.forward_bn_name in ["pgd", "normal"]:
            bn_name = self.forward_bn_name
        if self.vq_in:
            _, quantized, _, _ = self.vq_in(x)
            x = quantized
        # normalize
        # x = self.normalize(x)

        # See note [TorchScript super()]
        x = self.conv1(x)
        # import ipdb;ipdb.set_trace()
        x = self.bn1([x, bn_name])

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1([x, bn_name])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x[0])
        x = torch.flatten(x, 1)
        

        x = self.fc(x)

        return x

    # def forward(self, x, bn_name=None):
    def forward(self, x, bn_name="pgd"):

        return self._forward_impl(x, bn_name)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def multi_bn_vq_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, vq_in=True,
                   **kwargs)

def multi_bn_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def multi_bn_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)




# class CrossTrainingBN(nn.Module):

#     def __init__(self, num_features,  eps=1e-05, momentum = 0.1):

#         super(CrossTrainingBN, self).__init__()

#         self.eps = eps
#         self.momentum = torch.tensor( (momentum), requires_grad = False)

#         self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
#         self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

#         self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
#         self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

#     def forward(self, x):
#         """
#         x: a cat tensor torch.cat([X_adv, X_clean])
#         """

#         device = self.gamma.device

#         if self.training:
#             bs = int(x.size(0)/2)
#             # bs = int(x.size(0))

#             #use adv
#             # batch_ch_mean = torch.mean(x[:bs], dim=(0,2,3), keepdim=True).to(device)
#             # batch_ch_std = torch.clamp(torch.std(x[:bs], dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

#             # use clean
#             batch_ch_mean = torch.mean(x[bs:], dim=(0,2,3), keepdim=True).to(device)
#             batch_ch_std = torch.clamp(torch.std(x[bs:], dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

#             # batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
#             # batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

#         self.running_avg_std = self.running_avg_std.to(device)
#         self.running_avg_mean = self.running_avg_mean.to(device)
#         self.momentum = self.momentum.to(device)

#         if self.training:
#             x = (x - batch_ch_mean) / batch_ch_std
#             x = x * self.gamma + self.beta
#             self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
#             self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)
#         else:
#             x = (x - self.running_avg_mean) / self.running_avg_std
#             x = self.gamma * x + self.beta

#         return x


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class CrossTrainingBN_clean(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum: float = 0.1):

        super(CrossTrainingBN_clean, self).__init__()

        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

        self.running_avg_mean = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=False))
        self.running_avg_std = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=False))

    def forward(self, x):
        """
        x: a cat tensor torch.cat([X_adv, X_clean])
        """

        device = self.gamma.device

        if self.training:
            bs = int(x.size(0)/2)
            # bs = int(x.size(0))

            #use adv
            # batch_ch_mean = torch.mean(x[:bs], dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.clamp(torch.std(x[:bs], dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

            # use clean
            batch_ch_mean = torch.mean(x[bs:], dim=(0,2,3), keepdim=True).to(device)
            batch_ch_std = torch.std(x[bs:], dim=(0,2,3), keepdim=True)


            # batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)

        # import ipdb; ipdb.set_trace()
        if self.training:
            x = (x - batch_ch_mean) / batch_ch_std
            x = x * self.gamma + self.beta
            # self.running_avg_mean.data = (self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)).data
            # self.running_avg_std.data = (self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)).data
            self.running_avg_mean = nn.Parameter(self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean), requires_grad=False)
            self.running_avg_std = nn.Parameter(self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std), requires_grad=False)
        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x

class CrossTrainingBN_adv(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum: float = 0.1):

        super(CrossTrainingBN_adv, self).__init__()

        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

        self.running_avg_mean = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=False))
        self.running_avg_std = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=False))

    def forward(self, x):
        """
        x: a cat tensor torch.cat([X_adv, X_clean])
        """

        device = self.gamma.device

        if self.training:
            bs = int(x.size(0)/2)
            # bs = int(x.size(0))

            #use adv
            batch_ch_mean = torch.mean(x[:bs], dim=(0,2,3), keepdim=True).to(device)
            batch_ch_std = torch.std(x[:bs], dim=(0,2,3), keepdim=True)

            # use clean
            # batch_ch_mean = torch.mean(x[bs:], dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.std(x[bs:], dim=(0,2,3), keepdim=True)


            # batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)


        if self.training:
            x = (x - batch_ch_mean) / batch_ch_std
            x = x * self.gamma + self.beta
            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)
        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x


class CrossTrainingBN_all(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum: float = 0.1):

        super(CrossTrainingBN_all, self).__init__()

        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

        self.register_buffer('running_mean', torch.ones((1, num_features, 1, 1), requires_grad=False))
        self.register_buffer('running_var', torch.zeros((1, num_features, 1, 1), requires_grad=False))

    def forward(self, x):
        """
        x: a cat tensor torch.cat([X_adv, X_clean])
        """

        device = self.gamma.device

        if self.training:
            bs = int(x.size(0)/2)
            # bs = int(x.size(0))

            #use adv
            # batch_ch_mean = torch.mean(x[:bs], dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.std(x[:bs], dim=(0,2,3), keepdim=True)

            # use clean
            # batch_ch_mean = torch.mean(x[bs:], dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.std(x[bs:], dim=(0,2,3), keepdim=True)

            # use all
            batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
            batch_ch_std = torch.std(x, dim=(0,2,3), keepdim=True)

        # self.running_avg_std = self.running_avg_std.to(device)
        self.running_var = self.running_var.to(device)

        # self.running_avg_mean = self.running_avg_mean.to(device)
        self.running_mean = self.running_mean.to(device)



        if self.training:
            x = (x - batch_ch_mean) / batch_ch_std
            x = x * self.gamma + self.beta
            # self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
            self.running_mean = self.running_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_mean)

            # self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)
            self.running_var = self.running_var + self.momentum * (batch_ch_std.data.to(device) - self.running_var)

        else:
            # x = (x - self.running_avg_mean) / self.running_avg_std
            x = (x - self.running_mean) / self.running_var

            x = self.gamma * x + self.beta

        return x



class CrossTraining_DualBN_swap(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum: float = 0.1):

        super(CrossTraining_DualBN_swap, self).__init__()

        self.eps = eps
        self.momentum = momentum

        # init bn for clean
        self.gamma_clean = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.beta_clean = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        # NOTE: Buffers won’t be returned in model.parameters()
        self.register_buffer("running_avg_mean_clean", torch.ones((1, num_features, 1, 1)))
        self.register_buffer("running_avg_std_clean", torch.zeros((1, num_features, 1, 1)))
        self.running_avg_mean_clean: Optional[Tensor]
        self.running_avg_std_clean: Optional[Tensor]
        
        # init bn for adv
        self.gamma_adv = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        self.beta_adv = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))
        self.register_buffer("running_avg_mean_adv", torch.ones((1, num_features, 1, 1)))
        self.register_buffer("running_avg_std_adv", torch.zeros((1, num_features, 1, 1)))
        self.running_avg_mean_adv: Optional[Tensor]
        self.running_avg_std_adv: Optional[Tensor]
        
    def forward(self, x, bn_name):
        """
        x: a cat tensor torch.cat([X_adv, X_clean])
        """

        if self.training:
            bs = int(x.size(0)/2)
            # bs = int(x.size(0))

            #use adv
            batch_ch_mean_adv = torch.mean(x[:bs], dim=(0,2,3), keepdim=True)
            batch_ch_std_adv = torch.std(x[:bs], dim=(0,2,3), keepdim=True)

            # use clean
            batch_ch_mean_clean = torch.mean(x[bs:], dim=(0,2,3), keepdim=True)
            batch_ch_std_clean = torch.std(x[bs:], dim=(0,2,3), keepdim=True)


            # batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
            # batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

            x_clean = x[bs:]
            x_adv = x[:bs]
            
            # x_clean = (x_clean - batch_ch_mean_adv) / batch_ch_std_adv
            # x_clean = x_clean * self.gamma_adv + self.beta_adv
            
            # x_adv = (x_adv - batch_ch_mean_clean) / batch_ch_std_clean
            # x_adv = x_adv * self.gamma_clean + self.beta_clean
            
            # for verify
            x_adv = (x_adv - batch_ch_mean_adv) / batch_ch_std_adv
            x_adv = x_adv * self.gamma_adv + self.beta_adv
            
            x_clean = (x_clean - batch_ch_mean_clean) / batch_ch_std_clean
            x_clean = x_clean * self.gamma_clean + self.beta_clean
            
            self.running_avg_mean_clean = self.running_avg_mean_clean + self.momentum * (batch_ch_mean_clean.data - self.running_avg_mean_clean)
            self.running_avg_std_clean = self.running_avg_std_clean + self.momentum * (batch_ch_std_clean.data - self.running_avg_std_clean)
        
            self.running_avg_mean_adv = self.running_avg_mean_adv + self.momentum * (batch_ch_mean_adv.data - self.running_avg_mean_adv)
            self.running_avg_std_adv = self.running_avg_std_adv + self.momentum * (batch_ch_std_adv.data - self.running_avg_std_adv)
            
            x = torch.cat([x_clean, x_adv])
            
            return x
        
        else:
            if bn_name == "pgd":
                x = (x - self.running_avg_mean_adv) / self.running_avg_std_adv
                x = x * self.gamma_adv + self.beta_adv
            
            elif bn_name == "normal":
                x = (x - self.running_avg_mean_clean) / self.running_avg_std_clean
                x = x * self.gamma_clean + self.beta_clean
                
            return x




# class Disentangling_LP(nn.Module):

#     def __init__(self, num_features,  eps=1e-05, momentum: float = 0.1):

#         super(Disentangling_LP, self).__init__()

#         self.eps = eps
#         self.momentum = momentum

#         self.gamma_pgd = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
#         self.beta_pgd = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

#         self.gamma_normal = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
#         self.beta_normal = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

#         self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
#         self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

    # def forward(self, x, bn_names):
    #     """
    #     x: a cat tensor torch.cat([X_adv, X_clean])
    #     """

    #     device = self.gamma_pgd.device

    #     if self.training:
    #         bs = int(x.size(0))

    #         #use adv
    #         # batch_ch_mean = torch.mean(x[:bs], dim=(0,2,3), keepdim=True).to(device)
    #         # batch_ch_std = torch.std(x[:bs], dim=(0,2,3), keepdim=True)

    #         # use clean
    #         # batch_ch_mean = torch.mean(x[bs:], dim=(0,2,3), keepdim=True).to(device)
    #         # batch_ch_std = torch.std(x[bs:], dim=(0,2,3), keepdim=True)

    #         # use all
    #         batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
    #         batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)
    #         # torch.std(x, dim=(0,2,3), keepdim=True)
    #     self.running_avg_std = self.running_avg_std.to(device)
    #     self.running_avg_mean = self.running_avg_mean.to(device)

    #     if self.training:
    #         x = (x - batch_ch_mean) / batch_ch_std

    #         pgd_out = x[:bs] * self.gamma_pgd + self.beta_pgd
    #         normal_out = x[bs:] * self.gamma_normal + self.beta_normal

    #         x = torch.cat([pgd_out, normal_out])



    #         self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
    #         self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)
    #     else:

    #         x = (x - self.running_avg_mean) / self.running_avg_std
    #         if bn_names == "pgd":
    #             x = x * self.gamma_pgd + self.beta_pgd
    #         elif bn_names == "normal":
    #             x = x * self.gamma_normal + self.beta_normal

    #     return x


from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional, Any
from torch.nn import functional as F

class Disentangling_LP(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Disentangling_LP, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )


    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # self.weight_pgd = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_pgd = Parameter(torch.empty(num_features, **factory_kwargs))

            # self.weight_normal = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_normal = Parameter(torch.empty(num_features, **factory_kwargs))

            self.weight_pgd = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_pgd = Parameter(torch.zeros(num_features, **factory_kwargs))

            self.weight_normal = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_normal = Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        # self.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))



    def forward(self, input: Tensor, bn_name) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        bs = int(input.size(0)/2)

        if self.training:
            bn_training = True

            out_pgd = F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight_pgd,
                self.bias_pgd,
                bn_training,
                exponential_average_factor,
                self.eps,
            )[:bs]

            out_normal = F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight_normal,
                self.bias_normal,
                bn_training,
                exponential_average_factor,
                self.eps,
            )[bs:]

            return torch.cat([out_pgd, out_normal])

        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
            
            if bn_name == "pgd":
                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight_pgd,
                    self.bias_pgd,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
            elif bn_name == "normal":
                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight_normal,
                    self.bias_normal,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )



class BN2dStrongerAT(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        # momentum=1,

        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BN2dStrongerAT, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )


    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # self.weight_pgd = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_pgd = Parameter(torch.empty(num_features, **factory_kwargs))

            # self.weight_normal = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_normal = Parameter(torch.empty(num_features, **factory_kwargs))

            self.weight_pgd = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_pgd = Parameter(torch.zeros(num_features, **factory_kwargs))

            self.weight_temp = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_temp = Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        # self.reset_parameters()



    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))



    def forward(self, input: Tensor, bn_name) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        # bs = int(input.size(0)/2)

        if self.training:

            if "AE_generate" in bn_name:
                bn_training = False

                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight_temp,
                    self.bias_temp,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
            else:
                bn_training = True
                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight_pgd,
                    self.bias_pgd,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )



        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
            

            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight_pgd,
                self.bias_pgd,
                bn_training,
                exponential_average_factor,
                self.eps,
            )



class BN2dStrongerATMomentum1(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=1,
        # momentum=1,

        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BN2dStrongerATMomentum1, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )


    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # self.weight_pgd = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_pgd = Parameter(torch.empty(num_features, **factory_kwargs))

            # self.weight_normal = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_normal = Parameter(torch.empty(num_features, **factory_kwargs))

            self.weight_pgd = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_pgd = Parameter(torch.zeros(num_features, **factory_kwargs))

            self.weight_temp = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_temp = Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        # self.reset_parameters()



    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))



    def forward(self, input: Tensor, bn_name) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        # bs = int(input.size(0)/2)

        if self.training:

            if "AE_generate" in bn_name:
                bn_training = False

                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight_temp,
                    self.bias_temp,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
            else:
                bn_training = True
                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight_pgd,
                    self.bias_pgd,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )



        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
            

            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight_pgd,
                self.bias_pgd,
                bn_training,
                exponential_average_factor,
                self.eps,
            )

class Disentangling_StatP(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Disentangling_StatP, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )


    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # self.weight_pgd = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_pgd = Parameter(torch.empty(num_features, **factory_kwargs))

            # self.weight_normal = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.bias_normal = Parameter(torch.empty(num_features, **factory_kwargs))

            self.weight_pgd = Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias_pgd = Parameter(torch.zeros(num_features, **factory_kwargs))

            # self.weight_temp = Parameter(torch.ones(num_features, **factory_kwargs))
            # self.bias_temp = Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer('running_mean_pgd', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var_pgd', torch.ones(num_features, **factory_kwargs))

            self.register_buffer('running_mean_normal', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var_normal', torch.ones(num_features, **factory_kwargs))

            self.running_mean_pgd: Optional[Tensor]
            self.running_var_pgd: Optional[Tensor]
            self.running_mean_normal: Optional[Tensor]
            self.running_var_normal: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        # self.reset_parameters()



    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))



    def forward(self, input: Tensor, bn_name) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean_pgd is None) and (self.running_var_pgd is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        if bn_name == "pgd":

            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean_pgd
                if not self.training or self.track_running_stats
                else None,
                self.running_var_pgd if not self.training or self.track_running_stats else None,
                self.weight_pgd,
                self.bias_pgd,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean_normal
                if not self.training or self.track_running_stats
                else None,
                self.running_var_normal if not self.training or self.track_running_stats else None,
                self.weight_pgd,
                self.bias_pgd,
                bn_training,
                exponential_average_factor,
                self.eps,
            )


# class Disentangling_LP(_NormBase):
#     def __init__(
#         self,
#         num_features,
#         eps=1e-5,
#         momentum=0.1,
#         affine=True,
#         track_running_stats=True,
#         device=None,
#         dtype=None
#     ):
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(Disentangling_LP, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
#         )


#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-5,
#         momentum: float = 0.1,
#         affine: bool = True,
#         track_running_stats: bool = True,
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(_NormBase, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight_pgd = Parameter(torch.empty(num_features, **factory_kwargs))
#             self.bias_pgd = Parameter(torch.empty(num_features, **factory_kwargs))

#             self.weight_normal = Parameter(torch.empty(num_features, **factory_kwargs))
#             self.bias_normal = Parameter(torch.empty(num_features, **factory_kwargs))

#             # self.weight = Parameter(torch.ones(num_features, **factory_kwargs))
#             # self.bias = Parameter(torch.zeros(num_features, **factory_kwargs))


#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)

#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
#             self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
#             self.running_mean: Optional[Tensor]
#             self.running_var: Optional[Tensor]
#             self.register_buffer('num_batches_tracked',
#                                  torch.tensor(0, dtype=torch.long,
#                                               **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
#         else:
#             self.register_buffer("running_mean", None)
#             self.register_buffer("running_var", None)
#             self.register_buffer("num_batches_tracked", None)
#         # self.reset_parameters()

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError("expected 4D input (got {}D input)".format(input.dim()))



#     def forward(self, input: Tensor, bn_name) -> Tensor:
#         self._check_input_dim(input)

#         # exponential_average_factor is set to self.momentum
#         # (when it is available) only so that it gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum

#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore[has-type]
#                 self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)

#         r"""
#         Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#         passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#         used for normalization (i.e. in eval mode when buffers are not None).
#         """

#         bs = int(input.size(0)/2)

#         if self.training:
#             bn_training = True

#             if bn_name == "pgd":

#                 out = F.batch_norm(
#                     input,
#                     # If buffers are not to be tracked, ensure that they won't be updated
#                     self.running_mean
#                     if not self.training or self.track_running_stats
#                     else None,
#                     self.running_var if not self.training or self.track_running_stats else None,
#                     self.weight,
#                     self.bias,
#                     bn_training,
#                     exponential_average_factor,
#                     self.eps,
#                 )


#             return out

#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)
            
#             if bn_name == "pgd":
#                 return F.batch_norm(
#                     input,
#                     # If buffers are not to be tracked, ensure that they won't be updated
#                     self.running_mean
#                     if not self.training or self.track_running_stats
#                     else None,
#                     self.running_var if not self.training or self.track_running_stats else None,
#                     self.weight_pgd,
#                     self.bias_pgd,
#                     bn_training,
#                     exponential_average_factor,
#                     self.eps,
#                 )
#             elif bn_name == "normal":
#                 return F.batch_norm(
#                     input,
#                     # If buffers are not to be tracked, ensure that they won't be updated
#                     self.running_mean
#                     if not self.training or self.track_running_stats
#                     else None,
#                     self.running_var if not self.training or self.track_running_stats else None,
#                     self.weight_normal,
#                     self.bias_normal,
#                     bn_training,
#                     exponential_average_factor,
#                     self.eps,
                # )



class BatchNorm2dWOAffine(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()


    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
