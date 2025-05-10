"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 数据加载 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Dataset Dataloader ###################################################################################################
# Dataset和Dataloader都是torch中的抽象类，主要用于数据加载
# Dataset: 提供一种方式去获取数据及其label
# Dataloader: 为后面的网络提供不同的数据形式，如数据打乱，并行加载等
# 使用Dataset时应继承该抽象类，并在子类中重写__getitem__和__len__方法，前者用于提供方式去获取数据，后者用于提供这个数据集的数量有多少
#   __getitem__: 执行类对象[下标]时其实就是在调用该方法
#   __len__: 执行len(类对象)时其实就是在调用该方法
# 使用Dataloader时：实例 = Dataloader(Dataset类对象，batch_size=xx, shuffle=xx)
#   batch_size: 指定每个批次中样本的数量，比如2，4，6，8等
#   shuffle: 可设置为True或False，True表示在每个epoch开始时打乱数据集
########################################################################################################################

# 写一个加载数据类 -------------------------------------------------------------------------------------------------------
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Union
from os import PathLike

class MyData(Dataset):

    def __init__(self, root_path, img_path, label_path):
        self.root_path: Union[str, PathLike[str]] = root_path      # 存放数据集的路径，添加类型注释可以避免os.path.join的警告
        self.img_path: Union[str, PathLike[str]] = img_path        # 训练集或测试集，添加类型注释可以避免os.path.join的警告
        self.label_path: Union[str, PathLike[str]] = label_path    # 不同标签文件夹，添加类型注释可以避免os.path.join的警告
        self.image_path = os.path.join(self.root_path, self.img_path, self.label_path)   # 存放图像的路径
        self.image_listdir = os.listdir(self.image_path)           # 该函数返回的文件名顺序与文件系统相关(通常是无序的)
                                                                   # 所以通过索引可视化的图像顺序与在文件夹中看到的顺序并不一致

    def __getitem__(self, idx):
        """
        魔术方法，该方法定义后当类对象执行：类对象[下标索引]时实质上就是在调用该函数
        :param idx: 下标索引
        :return: 图像和图像对应的标签
        """
        img_name = self.image_listdir[idx]
        img_name_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_name_path)
        label = self.label_path
        return img, label

    def __len__(self):
        return len(self.image_listdir)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试数据加载功能
    test_root_path = "../data/hymenoptera_data/"
    test_img_path = "train/"
    test_label_path = "ants/"
    dataset = MyData(test_root_path, test_img_path, test_label_path)
    test_img, test_label = dataset[0]
    print(f"文件夹路径：{dataset.image_path}，当前显示的图片：{dataset.image_listdir[0]}")
    test_img.show()


