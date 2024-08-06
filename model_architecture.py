import torch
import torch.nn as nn 
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Each layer adds growth_rate number of filters
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(ConvBlock(layer_in_channels, growth_rate, kernel_size=3, stride=1, padding=1))
        
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        return torch.cat(features, dim=1)

class RPSNet(nn.Module):
    
    def __init__(self, num_classes=4):
        super(RPSNet, self).__init__()
        
        self.layer1 = ConvBlock(3, 8, kernel_size=3, stride=1, padding=1) # having padding=1 makes it so that a feature map of 3 doesnt reduce the image size.
        self.layer2 = ConvBlock(8, 8, kernel_size=3, stride=2, padding=1) # having stride=2 halves the size of the output feature map
        self.layer3 = DenseBlock(8, growth_rate=2, num_layers=4)
        self.layer4 = ConvBlock(16, 16, kernel_size=3, stride=1, padding=1)
        self.layer5 = ConvBlock(16, 16, kernel_size=3, stride=2, padding=1)
        self.layer6 = DenseBlock(16, growth_rate=4, num_layers=4)
        self.layer7 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer8 = ConvBlock(32, 32, kernel_size=3, stride=2, padding=1)
        self.layer9 = DenseBlock(32, growth_rate=8, num_layers=4)
        self.layer10 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer11 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer12 = DenseBlock(64, growth_rate=16, num_layers=4)
        self.layer13 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer14 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1)
        self.layer15 = DenseBlock(128, growth_rate=32, num_layers=4)
        self.layer16 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer17 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer18 = DenseBlock(256, growth_rate=64, num_layers=4)
 
        self.conv1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        
        x = self.conv1x1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(dim=2).squeeze(dim=2)
        
        return x
