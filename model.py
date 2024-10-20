from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.layer1 = nn.Linear(784, 2048, device=device, bias=False)
        self.layer2 = nn.Linear(2048, 512, device=device, bias=False)
        self.layer3 = nn.Linear(512, 64, device=device, bias=False)
        self.layer4 = nn.Linear(64, 32, device=device, bias=False)
        self.outputLayer = nn.Linear(32, 10, device=device, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.layer1.weight)
        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.kaiming_normal_(self.layer3.weight)
        nn.init.kaiming_normal_(self.layer4.weight)
        nn.init.kaiming_normal_(self.outputLayer.weight)
        # self.layer1.bias.data.fill_(0)
        # self.layer2.bias.data.fill_(0)
        # self.layer3.bias.data.fill_(0)
        # self.layer4.bias.data.fill_(0)
        # self.outputLayer.bias.data.fill_(0)

    def forward(self, x):
        x = torch.flatten(x, 1)
        out1 = self.layer1(x)
        out1 = nn.functional.relu(out1)
        out2 = self.layer2(out1)
        out2 = nn.functional.relu(out2)
        out3 = self.layer3(out2)
        out3 = nn.functional.relu(out3)
        out4 = self.layer4(out3)
        out4 = nn.functional.relu(out4)
        output = self.outputLayer(out4)
        
        return output









