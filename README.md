智能医学图像处理软件
一、项目简介
医学图像处理是医学影像分析的重要基础，能够辅助完成图像增强、去噪、目标区域提取以及定量分析等任务。
本项目设计并实现了一款基于 Python 的医学图像处理软件。
软件采用 PyQt5 构建可视化交互界面，集成传统数字图像处理算法与深度学习方法，实现医学图像读取、增强、去噪、传统分割以及基于 U-Net 网络的智能分割功能。
二、实际应用价值
本软件面向医学影像分析场景，可辅助完成：
1. 医学图像预处理
   通过图像增强和去噪算法改善医学图像质量，提高组织结构和病灶区域的可见性。
2. 病灶区域自动分割
   基于 U-Net 深度学习网络，实现医学图像中目标区域的自动提取，为后续定量分析提供基础。
3. 辅助医学影像分析
   软件能够输出分割区域面积、区域比例等定量指标，为医学图像研究和辅助诊断提供参考。
---
三、技术路线
1. 医学图像读取
支持 PNG、JPG、BMP、TIFF 等医学图像格式加载。
2. 图像预处理
包括：
图像增强
图像去噪
用于改善图像质量，提高组织结构显示效果。
3. 图像分割
传统分割：
Otsu阈值分割
Canny边缘检测
AI分割：
采用 U-Net 深度学习网络，实现医学图像自动区域分割。
流程：
医学图像 → 预处理 → U-Net → Mask预测 → 分割结果展示
---
四、数据集
采用公开数据集：
LGG MRI Segmentation Dataset
来源：
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
包含：
MRI医学图像
对应人工标注Mask
用于训练和测试U-Net模型。

由于原始数据规模较大，本仓库未上传完整数据。

dataset/filtered目录中提供少量示例图像，用于展示软件运行效果。
---
五、项目结构
```
MedImagePro

├── ai
│   ├── unet.py
│   ├── train.py
│   ├── predict.py
│   └── filter_dataset.py
│
├── core
│   ├── image_reader.py
│   ├── enhancement.py
│   ├── denoise.py
│   ├── segmentation.py
│   └── feature.py
│
├── utils
│   ├── visualization.py
│   └── metrics.py
│
├── dataset
├── weights
├── results
├── gui.py
└── main.py
```
---
六、运行环境
Python >= 3.10
主要依赖：
```
PyQt5
numpy
opencv-python
torch
torchvision
Pillow
```
安装：
```
pip install -r requirements.txt
```
---
七、软件运行
运行：
```
python main.py
```
流程：
打开医学图像
↓
图像增强/去噪
↓
传统分割或AI智能分割
↓
结果展示
↓
特征分析
---
八、结果评价指标
1. 图像处理效果评价
通过处理前后图像对比评价：
- 图像清晰度变化
- 对比度改善情况
- 噪声降低效果
2. AI分割性能评价
采用医学图像分割常用指标：
Dice系数
用于评价预测区域与真实标注区域的重合程度。
公式：
Dice = 2|A∩B|/(|A|+|B|)
IoU（Intersection over Union）
用于评价预测区域与真实区域的交并比。
公式：
IoU = |A∩B|/|A∪B|
3. 软件运行效果
实现：
- 医学图像加载
- 图像增强
- 图像去噪
- 传统分割
- U-Net智能分割
- 分割结果可视化
九、未来改进
支持DICOM医学影像格式
增加3D医学图像处理
引入Attention U-Net、nnU-Net等模型
增加自动医学报告生成
---
十、项目说明
本项目用于医学图像处理学习与研究展示。
所有医学影像数据均来自公开数据集，仅用于科研学习用途。