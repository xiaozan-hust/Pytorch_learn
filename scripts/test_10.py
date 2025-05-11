"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 完整的模型训练过程 利用GPU训练 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

########################################################################################################################
# 完整的模型训练过程：
#   ①: 加载数据
#   ②: 初始化模型，损失函数，优化器，以及想要记录的一些数据
#   ③: 开始训练模型+开始测试模型
#   ④: 打印或可视化模型效果
# 如何利用GPU进行训练：
#   利用GPU进行训练是比较简单的，只需要将数据、模型、损失函数移动到GPU上即可
#   方式一：.cuda()
#       数据移动到GPU: images = images.cuda()  targets = targets.cuda()
#       模型移动到GPU: demo_net = demo_net.cuda()
#       损失函数移动到GPU: loss_demo = loss_demo.cuda()
#   方式二：.to(device)
#       首先选择设备：device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # python三元表达式语法
#       数据移动到GPU: images = images.to(device)  targets = targets.to(device)
#       模型移动到GPU: demo_net = demo_net.to(device)
#       损失函数移动到GPU: loss_demo = loss_demo.to(device)
########################################################################################################################

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from demo_net.model import *
from torch.utils.data import DataLoader

print(f"当前设备是否支持CUDA：{torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 单GPU写法，如果是多GPU可以为cuda:1, cuda:2这种写法
print(f"当前训练设备选择为：{device}")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 初始化
demo_net = DemoNet()
loss_demo = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(demo_net.parameters(), lr=0.01)
epochs = 50                                                       # 训练轮数
total_train_steps = 0                                             # 记录训练次数
total_test_steps = 0                                              # 记录测试次数
train_loss = 0.0                                                  # 记录每次训练后的损失值
test_loss = 0.0                                                   # 记录每次测试后的损失值

# 将模型和损失函数移动到GPU
if torch.cuda.is_available():
    # 方式一：.cuda()
    # demo_net = demo_net.cuda()
    # loss_demo = loss_demo.cuda()
    # 方式二：.to(device)
    demo_net = demo_net.to(device)
    loss_demo = loss_demo.to(device)

# 可视化
writer = SummaryWriter("../runs/test_9")

# 开始训练网络
for epoch in range(epochs):
    print(f"============================开始第{epoch+1}轮训练============================")

    # 训练步骤
    demo_net.train()                                               # 如果网络中有Dropout层或者BatchNorm等层时需要注意加上
    train_loss = 0.0
    for data in train_loader:
        images, targets = data
        if torch.cuda.is_available():                              # 如果电脑支持GPU加速，将数据移动到GPU进行训练
            # 方式一：.cuda()
            # images = images.cuda()
            # targets = targets.cuda()
            # 方式二：.to(device)
            images = images.to(device)
            targets = targets.to(device)

        output_train = demo_net(images)
        result_loss = loss_demo(output_train, targets)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        total_train_steps += 1
        train_loss += result_loss
        if total_train_steps % 100 == 0:
            print(f"第{total_train_steps}次训练，损失值为：{train_loss}")
            writer.add_scalar("train_loss", train_loss, total_train_steps)
            train_loss = 0.0

    # 测试步骤
    demo_net.eval()                                                 # 如果网络中有Dropout层或者BatchNorm等层时需要注意加上
    test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():                                           # 固定参数，避免测试时影响梯度，再进行测试
        for data in test_loader:
            images, targets = data
            if torch.cuda.is_available():                           # 如果电脑支持GPU加速，将数据移动到GPU进行训练
                # 方式一：.cuda()
                # images = images.cuda()
                # targets = targets.cuda()
                # 方式二：.to(device)
                images = images.to(device)
                targets = targets.to(device)

            output_test = demo_net(images)
            result_loss = loss_demo(output_test, targets)
            test_loss += result_loss
            accuracy = (output_test.argmax(1) == targets).sum()     # argmax(1)将会在维度1上找到最大值的索引
            total_accuracy += accuracy
        total_test_steps += 1
        print(f"第{total_test_steps}次测试，损失值为：{test_loss}")
        print(f"第{total_test_steps}次测试，正确率为：{total_accuracy/len(test_data)}")
        writer.add_scalar("test_loss", test_loss, total_test_steps)
        writer.add_scalar("test_accuracy", total_accuracy/len(test_data), total_test_steps)

    print(f"----------------------------完成第{epoch+1}轮训练----------------------------")

# 保存模型与关闭事件文件
torch.save(demo_net, "../weights/demo_net.pth")
writer.close()