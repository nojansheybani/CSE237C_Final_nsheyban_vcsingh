from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType

class QuantLeNet(Module):
    def __init__(self, input_width, weight_width, act_width):
        
        super(QuantLeNet, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=input_width, min_val=-1.0, max_val=1.0)
        
        self.conv1 = QuantConv2d(1, 6, 5, weight_bit_width=weight_width)
        self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=weight_width)
        self.fc1   = QuantLinear(16*4*4, 120, bias=True, weight_bit_width=weight_width)
        self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=weight_width)
        self.fc3   = QuantLinear(84, 10, bias=False, weight_bit_width=weight_width)
        
        self.relu1 = QuantReLU(bit_width=act_width, max_val=6)
        self.relu2 = QuantReLU(bit_width=act_width, max_val=6)
        self.relu3 = QuantReLU(bit_width=act_width, max_val=6)
        self.relu4 = QuantReLU(bit_width=act_width, max_val=6)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
    
class QuantLeNetMinusRelu(Module):
    def __init__(self, input_width, weight_width, act_width):
        
        super(QuantLeNetMinusRelu, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=input_width, min_val=-1.0, max_val=1.0)
        
        self.conv1 = QuantConv2d(1, 6, 5, weight_bit_width=weight_width)
        self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=weight_width)
        self.fc1   = QuantLinear(16*4*4, 120, bias=True, weight_bit_width=weight_width)
        self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=weight_width)
        self.fc3   = QuantLinear(84, 10, bias=False, weight_bit_width=weight_width)
        
    def forward(self, x):
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out