"""
该代码用于PyTorch的学习
"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# nn.Module # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ###########################################################################################################
# nn.Module：所有神经网络模块的基类，通过继承它来构建神经网络，并重写__init__和__forward__
#   __init__：定义网络的各个层和子模块
#   __forward__：定义输入到输出的计算流程，即输入数据在网络中的前向传播过程
# 卷积
########################################################################################################################

# 写一个简单的神经网络骨架 -------------------------------------------------------------------------------------------------
import torch
from torch import nn
from torch.nn import Conv2d


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
# 写一个带有卷积层的神经网络 ----------------------------------------------------------------------------------------------
class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, conv_test):
        conv_output = self.conv1(conv_test)
        return conv_output

conv_net = ConvNet()
conv_net_output = conv_net(x)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试神经网络骨架的实现过程
    test_net = TestNet()
    x = torch.tensor(1.0)
    output = test_net(x)
    print(output)

    # 测试卷积操作(这里用的torch.nn.functional与一般搭建网络时用的torch.nn.Conv2d的用法稍有不同，但本质上数学操作是一致的)
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

    # 测试卷积层