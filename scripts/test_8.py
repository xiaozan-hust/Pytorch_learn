"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 损失函数 反向传播 优化器 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ###
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ######################################################################################################################
# 损失函数用于衡量模型预测值与真实标签之间的差异，是模型优化的核心，在PyTorch中分为内置损失函数和自定义损失函数
#   内置损失函数：如L1Loss()表示计算平均绝对误差，MSELoss()表示计算均方误差
#   自定义损失函数：
# 反向传播用于计算损失函数相对于模型参数的梯度，进而对网络模型的参数进行优化，从而使损失函数最小化，这个过程就叫做"梯度下降"
#   在调用损失函数后，执行loss.backward()后，梯度会累积在参数的.grad属性中，每次迭代前必须要先调用zero_grad()方法将之前的梯度清零
#   参数的.grad属性：一般的当创建带有参数的模型时，会自动给模型的参数添加.grad属性，该属性专门用于存储损失函数关于该参数的梯度(即导数)
# 优化器用于根据损失函数的梯度来更新模型的参数
#   学习率(lr)：控制参数更新的步长
#   step方法：执行该方法后将根据梯度实现参数更新
# 实现模型优化的基本步骤：
#   ①: 创建一个损失函数-自定义或者使用内置损失函数：
#   ②: 创建一个网络实例-自己搭建的网络，由于继承自nn.Module，所以其类对象会存在parameters()成员
#   ③: 创建一个优化器-自定义或使用内置优化器，需要传入两个参数如：torch.optim.SGD(my_net.parameters(), lr=0.01)
#   ④: 加载数据-得到待训练图像与对应标签
#   ⑤: 运行网络正向传播-得到网络的输出结果
#   ⑥: 运行损失函数-得到网络输出结果与标签之间的损失值，需要传入两个参数如：result_loss = my_net_loss(my_net_output, targets)
#   ⑦: 运行优化器的置零功能-将网络实例的参数的.grad属性置零：optimizer.zero_grad()
#   ⑧: 运行损失函数的反向传播-计算梯度值并加载到.grad中，如：result_loss.backward()
#   ⑨: 运行优化器的更新功能-根据梯度值更新网络模型的参数，如：optimizer.step()
#   ⑩: 至此完成一轮优化，重复④-⑨步骤，进行多次训练(即epoch)
########################################################################################################################

# 搭建一个神经网络(与上一节同) -----------------------------------------------------------------------------------------------
import torch, torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader

class MyNet(nn.Module):

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
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试内置损失函数
    input_test = torch.tensor([1, 2, 3], dtype=torch.float32)   # 一般要指定数据类型
    target_test = torch.tensor([1, 2, 5], dtype=torch.float32)  # 一般要指定数据类型
    lose_test = torch.nn.L1Loss()
    result_test = lose_test(input_test, target_test)
    print(result_test)

    # 测试优化器
    dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

    my_net_loss = nn.CrossEntropyLoss()                             # ①创建一个损失函数
    my_net = MyNet()                                                # ②创建一个网络实例
    optimizer = torch.optim.SGD(my_net.parameters(), lr=0.01)       # ③创建一个优化器

    # ================一般情况下我们会训练上百次数据，这里我们简单的设置为设置20次训练================#
    for epoch in range(20):
        running_loss = 0.0
        for data in dataloader:
            images, targets = data                                  # ④加载数据
            my_net_output = my_net(images)                          # ⑤运行网络正向传播
            result_loss = my_net_loss(my_net_output, targets)       # ⑥运行损失函数
            optimizer.zero_grad()                                   # ⑦运行优化器的置零功能
            result_loss.backward()                                  # ⑧运行损失函数的反向传播
            optimizer.step()                                        # ⑨运行优化器的更新功能
            running_loss += result_loss
        print(f"第{epoch}次训练后的损失值为：{running_loss}")          # ⑩完成一轮循环，重复④-⑨步骤，进行多次训练

    #========================该写法在所有的数据中只优化了一次，即只训练了一轮========================#
    # for data in dataloader:
    #     images, targets = data
    #     my_net_output = my_net(images)
    #     result_loss = my_net_loss(my_net_output, targets)
    #     optimizer.zero_grad()
    #     result_loss.backward()
    #     optimizer.step()

