import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import csv

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, gamma):
        # 在前向传播中保存输入，ctx 是一个用于存储中间值的上下文对象
        ctx.save_for_backward(weight, gamma)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播中修改计算图，这里只是一个示例，可以根据需求修改计算图
        weight, gamma, = ctx.saved_tensors
        if weight < 1e-3:
            # grad_input =grad_output+grad_output * weight  * gamma # 举例：修改梯度计算
            grad_input =grad_output * weight


        else:
            grad_input = grad_output+grad_output * weight

                         # + torch.zeros_like(grad_output).normal_(0,(grad_output ).std().item() + 1e-8) # 举例：修改梯度计算

        grad_weight = torch.tensor(1.0)
        grad_gamma = torch.tensor(1.0)
        return grad_input, grad_weight, grad_gamma


class GradScale_std(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        # 在前向传播中保存输入，ctx 是一个用于存储中间值的上下文对象
        ctx.save_for_backward(weight)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播中修改计算图，这里只是一个示例，可以根据需求修改计算图
        weight, = ctx.saved_tensors
        # grad_input = grad_output + grad_output * weight  # 举例：修改梯度计算
        grad_input = grad_output * weight  # 举例：修改梯度计算

        grad_weight = torch.tensor(1.0)
        return grad_input, grad_weight

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, n_head)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.w_1.weight)
    #     nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PCME(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return out


class ResNet(nn.Module):

    def __init__(self, args, block, layers, modality, num_classes=1000, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
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
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'depth':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'rgb':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # if self.pool == 'avgpool':
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192

        if args.pe:
            self.mu_dul_backbone = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                # nn.ELU(),

            )
            self.logvar_dul_backbone = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                # nn.ELU(),
            )

            # if modality=='visual':
            #     self.mu_dul_backbone = nn.Sequential(
            #         nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            #         nn.LayerNorm([512,7,7]),
            #         # nn.ELU(),
            #
            #     )
            #     self.logvar_dul_backbone = nn.Sequential(
            #         nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            #         nn.LayerNorm([512,7,7]),
            #         # nn.ELU(),
            #
            #     )
            # else:
            #     self.mu_dul_backbone = nn.Sequential(
            #         nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            #         nn.LayerNorm([512, 9, 6]),
            #         # nn.ELU(),
            #
            #     )
            #     self.logvar_dul_backbone = nn.Sequential(
            #         nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            #         nn.LayerNorm([512, 9, 6]),
            #         # nn.ELU(),
            #     )
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # import ipdb; ipdb.set_trace();
        if self.modality == 'visual' or self.modality == 'depth':

            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)

            x = self.conv1(x)

            # print("conv1 weight",self.conv1.weight)
            # print("visual,output",torch.sum(x))

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            out = x

            # print(out.shape)
            # print(self.args.epoch_now)

            if self.args.pe:
                # print(1)
                mu_dul = self.mu_dul_backbone(out)
                # mu_dul = F.normalize(mu_dul, dim=-1,p=2)
                logvar_dul = self.logvar_dul_backbone(out)
                logvar_dul = logvar_dul
                # logvar_dul = F.normalize(logvar_dul, dim=-1,p=2)
                std_dul = (logvar_dul * 0.5).exp()

                epsilon = torch.randn_like(std_dul)

                if self.training:
                    out = mu_dul + epsilon * std_dul
                    # out = mu_dul

                else:
                    out = mu_dul
                return out, mu_dul, std_dul
            else:
                out = out

                return out
        
        # elif self.modality == 'rgb' or self.modality == 'depth':
        #     (B, C, T, H, W) = x.size()
        #     x = x.permute(0, 2, 1, 3, 4).contiguous()
        #     x = x.view(B * T, C, H, W)

        #     x = self.conv1(x)

        #     # print("conv1 weight",self.conv1.weight)
        #     # print("visual,output",torch.sum(x))

        #     x = self.bn1(x)
        #     x = self.relu(x)
        #     x = self.maxpool(x)

        #     x = self.layer1(x)
        #     x = self.layer2(x)
        #     x = self.layer3(x)
        #     x = self.layer4(x)
        #     out = x

        #     return out
        
        else:
            # import ipdb; ipdb.set_trace();
            if len(x.size()) == 3:
                x = x.unsqueeze(1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            out = x

            # print(out.shape)

            if self.args.pe:
                # print(1)
                mu_dul = self.mu_dul_backbone(out)
                # mu_dul = F.normalize(mu_dul, dim=-1,p=2)
                logvar_dul = self.logvar_dul_backbone(out)
                logvar_dul = logvar_dul
                # logvar_dul = F.normalize(logvar_dul, dim=-1,p=2)
                std_dul = (logvar_dul * 0.5).exp()

                epsilon = torch.randn_like(std_dul)

                # mask = torch.gt(std_dul, 10)

                # with open('audio_single_pe_'+str(self.args.beta)+'.csv', 'a+', newline='') as f:
                #     csv_writer = csv.writer(f)
                #     std_bacthmean = torch.mean(std_dul, dim=0)
                #     std_bacthmean = std_bacthmean.view(1, -1)
                #     csv_writer.writerow(std_bacthmean.detach().cpu().numpy()[0])
                # #
                # # print(std_dul)
                # std_dul = torch.clamp(std_dul, min=0, max=2)

                # if self.args.epoch_now<2:
                std_dul = torch.clamp(std_dul, min=0, max=self.args.max)

                if self.training:

                    out = mu_dul + epsilon * std_dul

                else:
                    out = mu_dul
                return out, mu_dul, std_dul
            else:
                out = out

                return out


# --------------------------
# 1. 定义可学习类中心模块
# --------------------------
class LearnableClassCenters(nn.Module):
    def __init__(self, num_classes, feature_dim, init_centers=None):
        """
        初始化可学习类中心
        :param num_classes: 类别数量
        :param feature_dim: 特征维度
        :param init_centers: 初始类中心 (num_classes, feature_dim)，若为None则随机初始化
        """
        super().__init__()
        # 将类中心定义为可学习参数
        if init_centers is not None:
            # 用预计算的中心（如干净样本的均值）初始化
            self.centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))
        else:
            # 随机初始化（服从正态分布）
            self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward(self, labels=None):
        """
        获取类中心（可根据标签筛选特定类的中心）
        :param labels: 标签列表 (batch_size,)，若为None则返回所有类中心
        :return: 类中心特征 (batch_size, feature_dim) 或 (num_classes, feature_dim)
        """
        if labels is not None:
            # 根据标签获取对应类别的中心
            return self.centers[labels]
        return self.centers


# --------------------------
# 2. 定义损失函数（类内紧凑 + 类间分离）
# --------------------------
class CenterContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, intra_weight=1.0, inter_weight=1.0):
        """
        混合损失：类内损失（样本到中心的距离）+ 类间损失（中心之间的距离）
        :param margin: 类间最小距离阈值
        :param intra_weight: 类内损失权重
        :param inter_weight: 类间损失权重
        """
        super().__init__()
        self.margin = margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight

    def forward(self, features, labels, centers, confidences=None):
        """
        计算损失
        :param features: 样本特征 (batch_size, feature_dim)
        :param labels: 样本标签 (batch_size,)
        :param centers: 所有类中心 (num_classes, feature_dim)
        :param confidences: 样本置信度（无噪声样本=1，噪声样本<1）(batch_size,)，默认为1
        """
        batch_size = features.shape[0]
        num_classes = centers.shape[0]
        
        # 默认为全置信（无噪声）
        if confidences is None:
            confidences = torch.ones(batch_size, device=features.device)
        
        # --------------------------
        # 类内损失：样本特征到同类中心的距离（加权MSE）
        # --------------------------
        # 获取每个样本对应的类中心 (batch_size, feature_dim)
        class_centers = centers[labels]
        # 计算距离（支持模态缺失场景：用掩码屏蔽缺失维度）
        intra_dist = torch.norm(features - class_centers, p=2, dim=1)  # (batch_size,)
        # 置信度加权（噪声样本权重低，减少对中心的干扰）
        intra_loss = torch.mean(confidences * torch.square(intra_dist))
        
        # --------------------------
        # 类间损失：不同类中心之间的距离（确保大于margin）
        # --------------------------
        # 计算所有类中心之间的距离矩阵 (num_classes, num_classes)
        center_dist = torch.cdist(centers, centers, p=2)
        # 取出上三角部分（排除对角线）
        mask = torch.triu(torch.ones(num_classes, num_classes), diagonal=1).bool()
        inter_dist = center_dist[mask]  # (num_pairs,)
        # 类间距离小于margin则产生损失
        inter_loss = torch.mean(torch.max(torch.zeros_like(inter_dist), self.margin - inter_dist))
        
        # 总损失
        total_loss = self.intra_weight * intra_loss + self.inter_weight * inter_loss
        return total_loss, intra_loss, inter_loss

class ResNet_weight(nn.Module):

    def __init__(self, args, block, layers, modality, num_classes=1000, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_weight, self).__init__()
        self.modality = modality
        self.pool = pool
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
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if args.pe:
            self.mu_dul_backbone = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),

            )
            self.logvar_dul_backbone = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                # nn.ELU(),
            )

        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        self.class_centers = LearnableClassCenters(num_classes, feature_dim=512, init_centers=None)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,labels):

        if self.modality == 'visual':

            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)

            x = self.conv1(x)

            # print("conv1 weight",self.conv1.weight)
            # print("visual,output",torch.sum(x))

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            out = x

            # print(out.shape)
            # print(self.args.epoch_now)

            centers = self.class_centers(labels)

            if self.args.pe:
                # print(1)
                mu_dul = self.mu_dul_backbone(out)
                logvar_dul = self.logvar_dul_backbone(out)
                logvar_dul = logvar_dul
                std_dul = (logvar_dul * 0.5).exp()

                epsilon = torch.randn_like(std_dul)

                if self.training:
                    out = mu_dul + epsilon * std_dul

                else:
                    out = mu_dul
                return out, mu_dul, std_dul,centers
            else:
                out = out

                return out,centers
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            out = x

            # print(out.shape)


            centers = self.class_centers(labels)

            if self.args.pe:
                # print(1)
                mu_dul = self.mu_dul_backbone(out)
                logvar_dul = self.logvar_dul_backbone(out)
                logvar_dul = logvar_dul
                std_dul = (logvar_dul * 0.5).exp()

                epsilon = torch.randn_like(std_dul)

                if self.training:
                    out = mu_dul + epsilon * std_dul

                else:
                    out = mu_dul
                return out, mu_dul, std_dul,centers
            else:
                out = out

                return out,centers


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _resnet(arch, args, block, layers, modality, progress, **kwargs):
    model = ResNet(args, block, layers, modality, **kwargs)
    if args.pretrain and modality == 'visual':
    # if modality == 'visual': 
        print("pretrain")
        state_dict = torch.load("/root/autodl-tmp/daima/used/General_IML/pretrain_weights/resnet18-5c106cde.pth")
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(modality, args, progress=True, **kwargs):
    return _resnet('resnet18', args, BasicBlock, [2, 2, 2, 2], modality, progress,
                   **kwargs)


def _resnet_weight(arch, args, block, layers, modality, progress, **kwargs):
    model = ResNet_weight(args, block, layers, modality, **kwargs)
    if args.pretrain and args.modality == 'visual':
        state_dict = torch.load("resnet18-5c106cde.pth")
        model.load_state_dict(state_dict)
    return model


def resnet18_weight(modality, args, progress=True, **kwargs):
    return _resnet_weight('resnet18', args, BasicBlock, [2, 2, 2, 2], modality, progress,
                          **kwargs)


def resnet50(modality, args, progress=True, **kwargs):
    return _resnet('resnet50', args, BasicBlock, [3, 4, 6, 3], modality, progress,
                   **kwargs)
