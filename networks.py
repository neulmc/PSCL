"""
This file contains the UNet structure, its modules,
and the head functions used for contrast learning in our PSCL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the pretrained model
def load_model(base_encoder, load_checkpoint_dir):
    checkpoint = torch.load(load_checkpoint_dir, map_location="cpu")
    state_dict = checkpoint['finetune']
    # Load the parameters
    base_encoder.load_state_dict(state_dict, strict=True)
    return base_encoder

# DoubleConv in UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    """Double convolution block with optional normalization"""
    def __init__(self, in_channels, out_channels, mid_channels=None, normName = ['BN','LN'], droprate = None, size = 0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.normName = normName
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        # First conv + norm + relu
        self.con1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        if normName[0] == 'BN':
            self.norm1 = nn.BatchNorm2d(mid_channels)
        elif normName[0] == 'LN':
            self.norm1 = nn.LayerNorm([mid_channels])
        self.act1 = nn.LeakyReLU(inplace=True)
        # Second conv + norm + relu
        self.con2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if normName[1] == 'BN':
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normName[1] == 'LN':
            self.norm2 = nn.LayerNorm([out_channels])
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        B, _, H, W = x.shape
        # First conv block
        x = self.con1(x)
        if self.normName[0] == 'BN':
            x = self.norm1(x)
        elif self.normName[0] == 'LN':
            x = x.view(B, self.mid_channels, H*W).permute(0, 2, 1)
            x = self.norm1(x)
            x = x.permute(0, 2, 1).view(B, self.mid_channels, H, W)
        x = self.act1(x)
        # Second conv block
        x = self.con2(x)
        if self.normName[1] == 'BN':
            x = self.norm2(x)
        elif self.normName[1] == 'LN':
            x = x.view(B, self.out_channels, H*W).permute(0, 2, 1)
            x = self.norm2(x)
            x = x.permute(0, 2, 1).view(B, self.out_channels, H, W)
        x = self.act2(x)
        return x

# SingleConv in UNet
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, droprate = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# Downsampling block in UNet
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, ConvBlock = DoubleConv, droprate = 0.5, size = 0, normName =[]):
        super().__init__()
        self.down = nn.MaxPool2d(2) # Downsample with maxpool
        self.conv = ConvBlock(in_channels, out_channels, droprate = droprate, size = size, normName = normName)

    def forward(self, x):
        return self.conv(self.down(x)) # Apply maxpool then conv block

# Upsampling block in UNet
class Up(nn.Module):
    """Upscaling block with either bilinear or transposed conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, ConvBlock = SingleConv, droprate = 0.5, size = 0, normName =[]):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        bilinear = False # Force transposed conv
        if bilinear:
            #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0, bias=False))
            self.conv = ConvBlock(in_channels, out_channels, droprate = droprate, size = size, normName = normName)
        else:
            # Use transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1)
            self.conv = ConvBlock(in_channels, out_channels, droprate = droprate, size = size, normName = normName)

    def forward(self, x1, x2):
        """Forward pass with skip connection"""
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate with skip connection and apply conv block
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output convolution block in UNet
class OutConv(nn.Module):
    """Output convolution block that converts features to final output"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.fc2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # Permute dimensions and apply linear layer
        x = x.permute(0, 2, 3, 1)
        #x = self.relu1(self.fc1(x))
        #x = self.fc2(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        return x

# Decoder part of UNet that produces final output
# Strictly, the "decoder" here is just the final mapping,
# the upsampling is in the encoder code below
class UNet_decode(nn.Module):
    def __init__(self, drop, channel, n_classes = 4, bilinear=False, droprate = 0.5):
        super(UNet_decode, self).__init__()
        self.outc = OutConv(channel[0], n_classes)

    def forward(self, xlist):
        x4, x4up1, x4up2, x4up3 = xlist
        x = self.outc(x4up3)
        return x

# Encoder part of UNet architecture adapted for our PSCL
class UNet_encode(nn.Module):
    def __init__(self, drop, channel, n_channels = 3, n_classes = 4, bilinear=False, droprate = 0, IncNorm = ['',''],
                 DownNorm = ['',''], UpNorm = ['','']):
        super(UNet_encode, self).__init__()
        ConvBlock = DoubleConv
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # Initial convolution block
        self.inc = ConvBlock(n_channels, channel[0], droprate=drop, size=376, normName = IncNorm)
        # Downsampling blocks
        self.down1 = Down(channel[0], channel[1], ConvBlock, droprate = drop, size = 188, normName = DownNorm)
        self.down2 = Down(channel[1], channel[2], ConvBlock, droprate = drop, size = 94, normName = DownNorm)
        self.down3 = Down(channel[2], channel[3], ConvBlock, droprate = drop, size = 47, normName = DownNorm)
        # Upsampling blocks
        self.up1 = Up(channel[3], channel[2], bilinear, ConvBlock, droprate = drop, size = 94, normName = UpNorm)
        self.up2 = Up(channel[2], channel[1], bilinear, ConvBlock, droprate = drop, size = 188, normName = UpNorm)
        self.up3 = Up(channel[1], channel[0], bilinear, ConvBlock, droprate = drop, size = 376, normName = UpNorm)

    def forward(self, x, sample = True):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # Decoder path with skip connections
        x4up1 = self.up1(x4, x3)
        x4up2 = self.up2(x4up1, x2)
        x4up3 = self.up3(x4up2, x1)
        # Return features at different scales for patch contrast learning
        return [x4, x4up1, x4up2, x4up3]

# Projection head for contrastive learning
class projection_MLP(nn.Module):
    def __init__(self):
        '''Projection head for the pretraining of the encoder.
            - Uses the dataset and model size to determine encoder output
                representation dimension.
            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_MLP, self).__init__()
        # Input dimension
        n_channels = 512
        # Two-layer MLP with ReLU activation
        self.projection_head = nn.Sequential()
        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        # Project to 128-dim space
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))
    def forward(self, x):
        return self.projection_head(x)

# Projection head for UNet features in contrastive learning
# This projection head is for global contrastive learning
class projection_UNET_MLP(nn.Module):
    def __init__(self):
        super(projection_UNET_MLP, self).__init__()
        # Input dimension from UNet
        n_channels = 1024
        # Global pooling
        self.avg_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        #self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Two-layer MLP similar to projection_MLP
        self.projection_head = nn.Sequential()
        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))

    def forward(self, x):
        B, C, H, W = x[0].size()
        x = self.avg_pooling(x[0]).view(B, -1) # Pool and flatten
        return self.projection_head(x) # Apply MLP

# Projection head for UNet features in contrastive learning
# This projection head is for local(patch) & global(image) contrastive learning for our PSCL
# It also contains multi-scale feature outputs
class Denseproj_UNET_MLP(nn.Module):
    """Dense projection head for multi-scale UNet features"""
    def __init__(self, channel=[64, 128, 256, 512], hidfea_num = 128, confea_num = 64, multihead = 4, normName = ['BN']):
        super(Denseproj_UNET_MLP, self).__init__()
        self.normName = normName
        n_hide = hidfea_num
        n_out = confea_num
        # Global average pooling and MLP for deepest features
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        if normName[1] == 'gBN':
            self.mlp = nn.Sequential(
                nn.Linear(channel[-1], min(n_hide, channel[-1])),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(min(n_hide, channel[-1])),  ## haha
                nn.Linear(min(n_hide, channel[-1]), n_out))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(channel[-1], min(n_hide, channel[-1])),
                nn.ReLU(inplace=True),
                nn.Linear(min(n_hide, channel[-1]), n_out))
        # Projection heads for different scale features
        # Each has conv -> norm -> relu -> conv structure
        self.conv_x4up1_conv1 = nn.Conv2d(channel[-2], min(n_hide, channel[-2]), 1, groups=multihead)
        if normName[0] == 'BN':
            self.conv_x4up1_norm1 = nn.BatchNorm2d(min(n_hide, channel[-2]))
        elif normName[0] == 'LN':
            self.conv_x4up1_norm1 = nn.LayerNorm([min(n_hide, channel[-2])])
        self.conv_x4up1_act = nn.ReLU(inplace=True)
        self.conv_x4up1_conv2 = nn.Conv2d(min(n_hide, channel[-2]), n_out, 1, groups=multihead)
        # Similar structure for other scales
        self.conv_x4up2_conv1 = nn.Conv2d(channel[-3], min(n_hide, channel[-3]), 1, groups=multihead)
        if normName[0] == 'BN':
            self.conv_x4up2_norm1 = nn.BatchNorm2d(min(n_hide, channel[-3]))
        elif normName[0] == 'LN':
            self.conv_x4up2_norm1 = nn.LayerNorm([min(n_hide, channel[-3])])
        self.conv_x4up2_act = nn.ReLU(inplace=True)
        self.conv_x4up2_conv2 = nn.Conv2d(min(n_hide, channel[-3]), n_out, 1, groups=multihead)
        # Similar structure for other scales
        self.conv_x4up3_conv1 = nn.Conv2d(channel[-4], min(n_hide, channel[-4]), 1, groups=multihead)
        if normName[0] == 'BN':
            self.conv_x4up3_norm1 = nn.BatchNorm2d(min(n_hide, channel[-4]))
        elif normName[0] == 'LN':
            self.conv_x4up3_norm1 = nn.LayerNorm([min(n_hide, channel[-4])])
        self.conv_x4up3_act = nn.ReLU(inplace=True)
        self.conv_x4up3_conv2 = nn.Conv2d(min(n_hide, channel[-4]), n_out, 1, groups=multihead)

    def forward(self, xlist):
        x4, x4up1, x4up2, x4up3 = xlist # Unpack multi-scale features
        B, _, _, _ = x4.size()
        # Process deepest features with global pooling and MLP
        avgpooled_x = self.avg_pooling(x4).view(B, -1)
        avgpooled_x = self.mlp(avgpooled_x)
        # Process each scale with its projection head
        x4up1 = self.conv_x4up1_conv1(x4up1)
        if self.normName[0] == 'BN':
            x4up1 = self.conv_x4up1_norm1(x4up1)
        elif self.normName[0] == 'LN':
            B, cc, H, W = x4up1.shape
            x4up1 = x4up1.view(B, cc, H*W).permute(0, 2, 1)
            x4up1 = self.conv_x4up1_norm1(x4up1)
            x4up1 = x4up1.permute(0, 2, 1).view(B, cc, H, W)
        x4up1 = self.conv_x4up1_act(x4up1)
        x4up1 = self.conv_x4up1_conv2(x4up1)
        x4up1 = x4up1.view(x4up1.size(0), x4up1.size(1), -1) # # Flatten spatial dimensions
        # Process other scales similarly
        x4up2 = self.conv_x4up2_conv1(x4up2)
        if self.normName[0] == 'BN':
            x4up2 = self.conv_x4up2_norm1(x4up2)
        elif self.normName[0] == 'LN':
            B, cc, H, W = x4up2.shape
            x4up2 = x4up2.view(B, cc, H * W).permute(0, 2, 1)
            x4up2 = self.conv_x4up2_norm1(x4up2)
            x4up2 = x4up2.permute(0, 2, 1).view(B, cc, H, W)
        x4up2 = self.conv_x4up2_act(x4up2)
        x4up2 = self.conv_x4up2_conv2(x4up2)
        x4up2 = x4up2.view(x4up2.size(0), x4up2.size(1), -1) # bxdxs^2
        # Process other scales similarly
        x4up3 = self.conv_x4up3_conv1(x4up3)
        if self.normName[0] == 'BN':
            x4up3 = self.conv_x4up3_norm1(x4up3)
        elif self.normName[0] == 'LN':
            B, cc, H, W = x4up3.shape
            x4up3 = x4up3.view(B, cc, H * W).permute(0, 2, 1)
            x4up3 = self.conv_x4up3_norm1(x4up3)
            x4up3 = x4up3.permute(0, 2, 1).view(B, cc, H, W)
        x4up3 = self.conv_x4up3_act(x4up3)
        x4up3 = self.conv_x4up3_conv2(x4up3)
        x4up3 = x4up3.view(x4up3.size(0), x4up3.size(1), -1) # bxdxs^2
        # Return projected features at all scales
        return [avgpooled_x, x4up1, x4up2, x4up3]

# The following code is related to the ResNet structure.
# This may be valuable for potential future classification tasks.

# Convolutional layer with 3x3 kernel
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# Convolutional layer with 1x1 kernel
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Basic residual block in resnet
class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18 and ResNet-34"""
    expansion = 1  # Expansion factor for output channels
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # First convolutional layer
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # Downsample layer if needed
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x # Save input for residual connection
        # First conv + relu
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply downsample to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Bottleneck residual block in resnet
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Three convolutional layers with bottleneck structure
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
        # First conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Second conv + bn + relu
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # Third conv + bn
        out = self.conv3(out)
        out = self.bn3(out)
        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        # Add residual and apply final relu
        out += identity
        out = self.relu(out)

        return out

# ResNet backbone architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
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

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)  # For CIFAR

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)  # For CIFAR

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # Create the four residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # Final pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch
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
        # Create downsample layer if needed
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # Add first block
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv + bn + relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Pass through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Final pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Constructs a ResNet-18 model
def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)