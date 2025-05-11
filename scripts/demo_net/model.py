"""
此文件用于自定义网络模型
"""
import torch
from torch import nn
from torch.nn import Sequential

class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, net_input):
        net_output = self.model1(net_input)
        return net_output

if __name__ == "__main__":
    demo_net = DemoNet()
    input_test = torch.ones((64, 3, 32, 32))
    output_test = demo_net(input_test)
    print(output_test.shape)
