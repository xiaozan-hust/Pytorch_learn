"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# torchvision数据集 Dataloader # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# torchvision数据集 Dataloader ##########################################################################################
# 利用torchvision.datasets可以直接从网上(官方网站地址)获得相应的数据集用于训练和测试
#   其为每个数据集都定义了对应的类，这些类继承自torch.utils.data.Dataset基类，实现了__init__，__getitem__和__len__等必要的方法
#   关于数据集更详细的介绍可以查看官方网站：https://docs.pytorch.org/vision/stable/datasets.html，获取数据集的具体函数使用可以直接查看源码(写的非常详细)
#   例如获得CIFAR10数据集：torchvision.datasets.CIFAR10(root=指定数据集存储路径, train=True代表加载训练集False代表加载测试集, download=True自动下载数据集, transform=图像变换(可选))
#   例如获得COCO数据集：torchvision.datasets.CocoDetection(root=指定数据集存储路径, annFile=指定数据集标注文件存储路径, transform=图像变换(可选))
# 获得数据集后往往要和Dataloader结合使用
#   Dataloader用于对Dataset对象中的数据进行批量加载，打乱顺序，并行加载等
#   dataset参数(必需)：指定要加载的Dataset对象
#   batch_size参数(可选)：每个批次中包含的样本数量，默认为1
#   shuffle参数(可选)：是否在每个epoch开始时打乱数据集，默认为False
#   num_workers(可选)：数据加载的子进程数量，默认为0表示在主进程加载数据，1，2，3代表创建1个，2个，3个子进程并行去加载数据
#   drop_last(可选)：如果数据集中的样本不能被batch_size整除，则丢弃最后一个不完整的批次，默认为False
#   pin_memory(可选)：是否将数据加载到CUDA固定内存中，默认为False
########################################################################################################################

# 写一个使用数据集的例子 -------------------------------------------------------------------------------------------------
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义图像变换(一般在数据加载时都会增加该参数)
tensor_trans = transforms.ToTensor()

# 得到数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=tensor_trans)    # 获取一个数据集
img, tar = train_data[0]                                                                                        # 调用魔术方法，得到图像和对应标签
print(f"图像类型：{img.shape}, 对应标注标签：{tar}, 标签对应的类别：{train_data.classes[0]}")                        # 打印图像，标签，类别

# Dataloader数据预处理
dataloader = DataLoader(dataset=train_data, batch_size=64)
for data in dataloader:
    images, targets = data
    print(f"图像类型：{images.shape}, 对应标注标签：{targets}")

# 在TensorBoard中查看
writer = SummaryWriter("../runs/test_4")
step = 0
for data in dataloader:
    images, targets = data                                                                                       # 如果shuffle被设置为True，那么这里每次取的是不一样的
    writer.add_images("batch_images", images, step, dataformats="NCHW")                                     # 注意这里要用add_images()函数
    step += 1
writer.close()
#-----------------------------------------------------------------------------------------------------------------------
