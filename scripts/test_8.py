"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 现有模型的使用与修改 模型的保存与加载 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch, torchvision
from torch import nn

if __name__ == "__main__":
    # 获取现有模型
    os.environ['TORCH_HOME'] = '../weights'                         # 指定自定义路径
    vgg16_true = torchvision.models.vgg16(pretrained=True)          # True表示下载预训练的模型，可以直接使用
    vgg16_false = torchvision.models.vgg16(pretrained=False)        # False表示下载参数随机的模型，需要自行训练

    # 查看模型结构
    print(vgg16_true)

    #==============================================修改模型=============================================================#
    # 例如原模型是1000个类别，要修改成最终只有10个类别
    # 方式一：通过刚才的打印模型结构发现，可以在该网络的classifier的最后添加一个线性层
    vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
    print(vgg16_true)
    # 方式二：或者选择直接将最后一个线性层修改为10
    vgg16_true.classifier[6] = nn.Linear(4096, 10)
    print(vgg16_true)

    # 以上为对模型进行新加层与修改层，除此之外还会有比如删除层，指定位置新加层等，可以自行去网上了解

    #============================================模型保存与加载==========================================================#
    # 方式一
    torch.save(vgg16_true, "../weights/vgg16_new.pth")                      # 保存模型，同时保存了模型结构与参数
    vgg16_new = torch.load("../weights/vgg16_new.pth")                         # 加载模型，对应于保存方式一
    print(vgg16_new)
    # 方式二
    torch.save(vgg16_true.state_dict(), "../weights/vgg16_dict_new.pth")    # 保存模型，只保存了模型参数(官方推荐)
    vgg16_dict_new = torchvision.models.vgg16(pretrained=False)                # 加载模型，该方式需要先定义模型结构，再加载权重参数
    vgg16_dict_new.load_state_dict(torch.load("../weights/vgg16_dict_new.pth"))

    #============================================模型保存与加载==========================================================#
    ### 例如我们现在对自己搭建的网络进行了训练，最终我们想保存这个训练后的模型
    # my_net = MyNet()
    # """
    # 此处省略模型的训练过程
    # """
    # torch.save(my_net, "保存路径/模型名称.pth")
    ### 但是当想加载这个保存的模型时，需要将你自行搭建的网络那个类同时写在加载模型的代码里
    # from 自定义网络的py文件 import *