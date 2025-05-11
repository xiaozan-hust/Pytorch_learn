"""
该代码用于PyTorch的学习
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 完整的模型验证过程 开源项目学习 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

########################################################################################################################
# 完整的模型验证过程：
#   ①: 加载想要测试的数据
#   ②: 使数据满足网络的输入要求，如尺寸大小，通道数，batch_size数，是否为张量类型等
#   ③: 加载待验证的模型，这里要保证模型和待测试数据在同一设备上，不相同时可以将某一个迁移到另一个所在设备
#   ④: 开始测试，建议加上：模型.eval()，开始测试时建议以：with torch.no_grad()方式
# 开源项目学习：
#   一定要阅读该仓库的README.md文件
#   add_argument()的用法-定义命令行参数，如果开源项目中有必须添加的参数，可以直接添加default值，这样就不用在命令行中添加一堆参数了
########################################################################################################################

import torchvision
from PIL import Image
from demo_net.model import *

# 加载数据
# img_test = Image.open("../data/net_test_dog.png")
img_test = Image.open("../data/net_test_airplane.png")
print(img_test)
img_test = img_test.convert("RGB")                              # 只保留RGB通道，使之符合模型输入(3通道)

# 使输入数据符合模型输入要求
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),                   # 对图像进行尺寸修改，但是不会修改图像的类型(此时仍为PIL类型)
     torchvision.transforms.ToTensor()]                         # 将PIL转换为张量类型
)
img_test = transform(img_test)
img_test = torch.reshape(img_test, [1, 3, 32, 32])        # 需要对图像指定batch_size，因为网络的输入都是包含这个参数的
print(img_test.shape)

# 加载模型
model = torch.load("../weights/demo_net.pth", weights_only=False)
print(model)

# 由于模型在GPU上，而输入图像在CPU上，所以需要保证一致
# 方式一：将图像移动到GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# img_test = img_test.to(device)
# 方式二：将模型移动到CPU
model = torch.load("../weights/demo_net.pth", weights_only=False, map_location="cpu")

# 开始测试
model.eval()
with torch.no_grad():
    output = model(img_test)
class_id = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(output)
print(output.argmax(1))
result_eval = class_id[output.argmax(1)]
print(f"模型预测结果为：{result_eval}")