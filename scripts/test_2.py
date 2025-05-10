"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TensorBoard # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# TensorBoard ##########################################################################################################
# SummaryWriter: 用于创建日志文件，可以在TensorBoard中可视化
# add_scalar: 将单个数值按时间步记录到日志中，语法为add_scalar(图表标签, 数值, 时间步)，其中数值即y轴，时间步即x轴
# add_image: 将图像数据按时间步记录到日志中，语法为add_image(图表标签, 图像, global_step=时间步, dataformats=根据图像类型确定)
# add_images: 将批次处理后的图像数据按时间步记录到日志中，语法为add_images(图表标签, 图像, global_step=时间步, dataformats=根据图像类型确定)
# TensorBoard可视化：
#   一般用法：终端输入tensorboard --logdir=事件文件所在文件夹名，然后在浏览器中访问端口即可
#   指定端口：终端输入tensorboard --logdir=事件文件所在文件夹名 --port=端口号(如6007)
########################################################################################################################

# 写一个TensorBoard的例子 ------------------------------------------------------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
import math
from PIL import Image
import numpy as np

writer = SummaryWriter("../runs/test_2")    # 指定结果存放位置
x = range(0, 10)                            # 生成时间步(假设这就是epoch)

# 将数值记录到日志中
for i in x:
    writer.add_scalar('y=2x', math.cos(0.1*i) * 2, i)

# 将图像记录到日志中
img_pil = Image.open("../data/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")  # 读取图像，该方式读后存储的类型为pil
img_np = np.array(img_pil)                                                          # 将图像类型转换为被支持的格式
for i in range(5):
    writer.add_image("image", img_np, global_step=i, dataformats="HWC")

writer.close()
