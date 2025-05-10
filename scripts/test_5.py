"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# nn.Module Conv2d MaxPool2d ReLU Sigmoid Linear # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ######################################################################################################################
# nn.Module：所有神经网络模块的基类，通过继承它来构建神经网络，并重写__init__和__forward__
#   __init__：定义网络的各个层和子模块
#   __forward__：定义输入到输出的计算流程，即输入数据在网络中的前向传播过程
# 卷积层-主要使用Conv2d()：
#   torch.nn.Conv2d()不需要手动指定卷积核(这与torch.nn.functional.conv2d用法不同)，其卷积核参数将会在训练中被不断改变
#   in_channels(int)(必需)：输入图像的通道数
#   out_channels(int)(必需)：输出图像通道数，有多少个输出通道数就会有多少个卷积核存在(卷积核之间可不同)
#   kernel_size(int or tuple)(必需)：卷积核大小(比如3表示3*3卷积核)
#   stride(int or tuple)(可选)：卷积核滑动步长，默认为1
#   padding(int or tuple)(可选)：输入边缘填充的大小，默认为0
#   dilation(int or tuple)(可选)：卷积核元素之间的间距，默认为1
#   groups(int or tuple)(可选)：输入和输出通道的分组数，默认为1
#   bias(bool)(可选)：是否添加偏置项，默认为True
#   padding_mode(string)(可选)：填充模式，默认为零填充
# 最大池化层-主要是MaxPool2d()：
#   torch.nn.MaxPool2d()会取窗口内的元素最大值，从而生成一个更小尺寸的特征图
#   kernel_size()()：池化核的大小
#   stride()()：池化核的滑动步长，默认等于kernel_size
#   padding()()：输出边缘的填充大小，默认不填充
#   dilation()()：池化核元素之间的间距，默认为1
#   return_indices()()：是否返回最大值的索引，常用于反池化操作
#   ceil_mode()()
# 非线性激活-主要是ReLU()和Sigmoid()
#   在网络中引入非线性特征，从而使模型能够学习复杂的函数映射
#   ReLU()：对数据进行截断：若大于零则不变，若小于零则置为零
#   Sigmoid()：将数据压缩到(0,1)之间
# 线性层及其它层-更多网络层的介绍请参考官网
#   Linear Layers(线性层或全连接层)：通过对输入数据执行线性变换来实现特征的加权组合，是神经网络中最基本的模块之一
#   Normalization Layers(正则化层)：对数据进行标准化处理，从而加速训练，提高模型稳定性并减少过拟合
#   Recurrent Layers(循环层)：处理序列数据，在处理序列时保留历史信息，使网络能够捕捉序列中的时序依赖关系
#   Transformer Layers：Transformer模型的核心组件，引入自注意力机制，被广泛应用于自然语言处理、计算机视觉、语音识别等领域
#   Dropout Layers：在训练过程中随机"丢弃"一部分神经元，迫使模型学习更鲁棒的特征表示，从而防止模型过拟合并提高泛化能力
#   Sparse Layers(稀疏层)：神经元之间的连接是稀疏的，即大部分连接的权重为零
########################################################################################################################

# 写一个简单的神经网络骨架 -------------------------------------------------------------------------------------------------
import torch
import torchvision.datasets
from torch import nn
from torch.ao.nn.quantized import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class TestNet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, test_input):
        """
        在nn.Module中的__call__函数中会调用forward函数(还会实现一些别的功能，比如日志记录，性能监控等)，
        我们一般只需要重写这个forward函数，当执行：类对象(输入数据)时就会实现正向传播
        :param test_input:
        :return:
        """
        test_output = test_input + 1
        return test_output
#-----------------------------------------------------------------------------------------------------------------------

# 写一个带有卷积层的神经网络 ----------------------------------------------------------------------------------------------
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)  # 主要是这几个参数比较重要

    def forward(self, conv_test):
        """
        nn.Conv2d类继承自nn.Module，因此同样实现了__call__方法，所以可以执行self.conv1(conv_test)这样的操作
        而该操作的本质其实是调用Conv2d类中的forward函数，从而实现卷积的计算
        :param conv_test:
        :return:
        """
        conv_output = self.conv1(conv_test)
        return conv_output
#-----------------------------------------------------------------------------------------------------------------------

# 写一个带有最大池化层的神经网络 -------------------------------------------------------------------------------------------
class PoolNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, pool_input):
        pool_output = self.pool1(pool_input)
        return pool_output
#-----------------------------------------------------------------------------------------------------------------------

# 写一个带有非线性激活的神经网络 -------------------------------------------------------------------------------------------
class NLANet(nn.Module):

    def __init__(self):
        super().__init__()
        self.nla1 = ReLU()
        self.nla2 = Sigmoid()

    def forward(self, nla_input):
        # nla_output = self.nla1(nla_input)     # 测试ReLU
        nla_output = self.nla2(nla_input)       # 测试Sigmoid
        return nla_output
#-----------------------------------------------------------------------------------------------------------------------

# 写一个带有线性层(全连接层)的神经网络 -------------------------------------------------------------------------------------
class LineNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.line1 = Linear(196608, 10)

    def forward(self, line_input):
        line_output = self.line1(line_input)
        return line_output
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试神经网络骨架的实现过程-------------------------------------------------------------------------------------------
    test_net = TestNet()
    x = torch.tensor(1.0)
    output = test_net(x)
    print(output)

    # 测试卷积操作(这里用的torch.nn.functional与一般搭建网络时用的torch.nn.Conv2d的用法稍有不同，但本质上数学操作是一致的)--------
    import torch.nn.functional as F

    conv_input = torch.tensor(
        [[1, 2, 0, 3, 1],
         [0, 1, 2, 3, 1],
         [1, 2, 1, 0, 0],
         [5, 2, 3, 1, 1],
         [2, 1, 0, 1, 1]]
    )
    conv_kernel = torch.tensor(
        [[1, 2, 1],
         [0, 1, 0],
         [2, 1, 0]]
    )
    conv_input = torch.reshape(conv_input, (1, 1, 5, 5))
    conv_kernel = torch.reshape(conv_kernel, (1, 1, 3, 3))
    print(conv_input.shape, conv_kernel.shape)

    conv_output_1 = F.conv2d(conv_input, conv_kernel, stride=1)
    print(conv_output_1)

    conv_output_2 = F.conv2d(conv_input, conv_kernel, stride=2)
    print(conv_output_2)

    conv_output_3 = F.conv2d(conv_input, conv_kernel, stride=1, padding=1)
    print(conv_output_3)

    # 准备数据，用于后面的卷积，池化，非线性激活，全连接的测试-----------------------------------------------------------------
    dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True)     # 这里将drop_last设置为True，是避免了最后不足64批次的部分数据在进行全连接层时会报错(因为全连接层要求输入特征是确定的)
    writer = SummaryWriter("../runs/test_5")

    input_test = torch.tensor(
        [[1, 2, 0, 3, 1],
         [0, 1, 2, 3, 1],
         [1, 2, 1, 0, 0],
         [5, 2, 3, 1, 1],
         [2, 1, 0, 1, 1]], dtype=torch.float32
    )
    pool_input_tensor = torch.reshape(input_test, [-1, 1, 5, 5])   # 对于卷积池化非线性激活等操作，前提都需要将其满足其格式
                                                                          # 需要包含四个数据：[batch_size, 通道数, 高, 宽]

    # 测试卷积层---------------------------------------------------------------------------------------------------------
    conv_net = ConvNet()
    print(conv_net)                 # 打印可以查看该网络的结构

    step = 0
    for data in dataloader:
        images, targets = data
        conv_net_output = conv_net(images)  # 该步骤将会首先调用ConvNet()类中的__call__，从而调用了我们自己重新定义的forward()函数
                                            # 在forward函数中又会调用卷积层Conv2d的类对象，并且执行其的__call__，从而调用其的forward()函数，从而最终调用了卷积计算
        print(f"输入数据大小：{images.shape}，卷积后数据大小：{conv_net_output.shape}")
        writer.add_images("conv_input", images, step)
        # TensorBoard中只会显示3通道图像，因此在这里使用了一种并不严谨的方法，将6通道改为3通道(其实就是将batch_size加倍了)，参数-1表示该处的参数自行计算
        # 修改大小：torch_size([64, 3, 30, 30]) -> [xxx, 3, 30, 30]
        conv_net_output = torch.reshape(conv_net_output, (-1, 3, 30, 30))
        writer.add_images("conv_output", conv_net_output, step)
        step += 1
    # writer.close()

    # 测试最大池化层------------------------------------------------------------------------------------------------------
    pool_net = PoolNet()
    print(pool_net)

    step = 0
    for data in dataloader:
        images, targets = data
        pool_net_output = pool_net(images)
        print(f"输入数据大小：{images.shape}，最大池化后数据大小：{pool_net_output.shape}")
        writer.add_images("pool_output", pool_net_output, step)
        step += 1
    # writer.close()

    # 测试非线性激活------------------------------------------------------------------------------------------------------
    nla_net = NLANet()
    print(nla_net)

    step = 0
    for data in dataloader:
        images, targets = data
        nla_net_output = nla_net(images)
        print(f"输入数据大小：{images.shape}，非线性激活后数据大小：{nla_net_output.shape}")
        writer.add_images("nla_output", nla_net_output, step)
        step += 1
    writer.close()

    # 测试线性层(全连接层)-------------------------------------------------------------------------------------------------
    line_net = LineNet()
    print(line_net)

    step = 0
    for data in dataloader:
        images, targets = data
        # images = torch.reshape(images, [1, 1, 1, -1])   # 一般的，全连接层要求输入数据的特征数是固定的
                                                               # 这里将图像数据进行"展平"，转换成torch.size([1, 1, 1, 196608])
        images = torch.flatten(images)                       # 该函数意为"展平"，其实现的效果与上行代码类似
        line_net_output = line_net(images)
        print(f"输入数据大小：{images.shape}，全连接后数据大小：{line_net_output.shape}")