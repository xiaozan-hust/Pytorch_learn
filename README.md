# Pytorch_learn

该仓库为Pytorch学习笔记

参考学习课程链接为：[小土堆Pytorch快速学习](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.1387.favlist.content.click)

## 文件树

* data: 存放测试用数据集
* note: 记录代码之外的一些笔记
* runs: 存放训练结果与TensorBoard可视化用
* scripts: 存放代码
* weights: 存放模型文件

## 一些说明

由于数据文件和部分事件文件比较大，因此未上传至github，但是在执行 `scripts`中的代码时会自动下载数据文件并且生成事件文件。

具体的，没有上传至github，但是在代码中被提到的文件有：

* data/cifar-10：CIFAR10数据集，若代码运行时需要，执行代码后会自动下载至data文件夹
* 部分runs/test_x：test_2至test_9在运行后皆会生成事件文件，github中只上传了test_2和test_3，若代码运行时需要，执行代码后会自动下载至runs文件夹
* weights/hub和vgg16模型：hub文件夹保存了VGG16模型，若代码运行时需要，执行代码后会自动下载至hub文件夹