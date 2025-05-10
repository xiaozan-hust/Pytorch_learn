"""
该代码用于PyTorch的学习
"""
from time import sleep

import torch.nn
import torchvision
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 神经网络搭建实践 Sequential的使用 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ######################################################################################################################
# Sequential序列：简化网络搭建方式，无需手动编写forward方法，不过仅支持线性顺序执行，且较难定位中间层问题
# 下面的代码搭建了一个用于图像分类的神经网络，但是目前该网络并没有进行训练，其中的参数没有进行优化，所以只是一个框架，并不具备实际的图像分类能力
########################################################################################################################

# 搭建一个图像分类的神经网络 -----------------------------------------------------------------------------------------------
from torch import nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 搭建方式一
        # self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        # self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        # self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        # self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        # self.pool3 = nn.MaxPool2d(2, ceil_mode=True)
        # self.flatten = nn.Flatten()                                     # 此处展开后特征为64*4*4，所以后面应该是跟了两个线性层
        # self.line1 = nn.Linear(1024, 64)
        # self.line2 = nn.Linear(64, 10)

        # 搭建方式二
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
        # 对应搭建方式一
        # net_output = self.conv1(net_input)
        # net_output = self.pool1(net_output)
        # net_output = self.conv2(net_output)
        # net_output = self.pool2(net_output)
        # net_output = self.conv3(net_output)
        # net_output = self.pool3(net_output)
        # net_output = self.flatten(net_output)
        # net_output = self.line1(net_output)
        # net_output = self.line2(net_output)

        # 对应搭建方式二
        net_output = self.model1(net_input)
        return net_output
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试搭建的一个网络：创建一个输入，查看网络最终的输出是否符合预期size
    my_net = MyNet()
    print(my_net)

    input_test = torch.ones((64, 3, 32, 32))    # 搭建的网络的输入是[3, 32, 32](不考虑batch_size下)
    output_test = my_net(input_test)
    print(output_test.shape)                    # 此处应输出[64, 10](不考虑batch_size时即为[10]，代表最终的10个类别所属)

    writer = SummaryWriter("../runs/test_6")
    writer.add_graph(my_net, input_test)        # 该方式可以将模型在TensorBoard中可视化查看
    writer.close()

    # 使用图像去测试这个网络
    dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True)
    for data in dataloader:
        images, targets = data
        output_my_net = my_net(images)
        print(output_my_net)
