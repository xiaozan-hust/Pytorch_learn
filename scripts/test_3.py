"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# transforms # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# transforms ###########################################################################################################
# transforms用于图像的预处理和数据增强，如转换为张量，归一化，尺寸调整，图像裁剪，调整亮度、对比度、饱和度等，图像翻转等
# ToTensor():
# Normalize():
# Resize():
# Compose(): 用于将多个图像变换操作组合在一起按顺序执行
# RandomCrop(): 对图像进行随机裁剪
########################################################################################################################

# 写一个transforms的例子 -------------------------------------------------------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("../runs/test_3")
img_pil = Image.open("../data/hymenoptera_data/train/ants/0013035.jpg")

# ToTensor
tensor_trans = transforms.ToTensor()    # 返回一个类对象
img_tensor = tensor_trans(img_pil)      # 本质上为调用__call__函数

# Normalize
norm_trans = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = norm_trans(img_tensor)

# Resize
resize_trans = transforms.Resize((512, 512))
img_resize = resize_trans(img_pil)
img_resize = tensor_trans(img_resize)

# Compose
com_trans = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为 224x224 大小
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行归一化处理
])
img_com = com_trans(img_pil)

# RandomCrop
random_trans = transforms.RandomCrop((512, 512))
# img_random = random_trans(img_pil)
# img_random = tensor_trans(img_random)

writer.add_image("tensor_img", img_tensor, 0, dataformats="CHW")
writer.add_image("norm_img", img_norm, 0, dataformats="CHW")
writer.add_image("resize_img", img_resize, 0, dataformats="CHW")
writer.add_image("com_img", img_com, 0, dataformats="CHW")
for i in range(10):
    img_random = random_trans(img_pil)
    img_random = tensor_trans(img_random)
    writer.add_image("random_img", img_random, i, dataformats="CHW")

writer.close()
#-----------------------------------------------------------------------------------------------------------------------
