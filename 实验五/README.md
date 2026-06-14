# 实验五：深度学习图像分类

本实验包含两个子实验，均基于 TensorFlow / Keras 实现图像分类任务。

---

## 目录结构

```
实验五/
├── README.md                  # 本文件
├── 5.1/                       # 实验 5.1：花卉图像分类（迁移学习 + TFLite 部署）
│   ├── shiyan5.1.ipynb        # Jupyter Notebook 源码
│   ├── shiyan5.1.md           # Notebook 导出的 Markdown
│   └── exported_flower_model/ # 训练产物目录
│       ├── flower_classifier.keras  # Keras 原始模型
│       ├── model.tflite             # TFLite 量化模型
│       └── labels.txt               # 类别标签文件
└── 5.2/                       # 实验 5.2：石头剪刀布手势识别（自定义 CNN）
    ├── shiyan5.2.ipynb        # Jupyter Notebook 源码
    └── shiyan5.2.md           # Notebook 导出的 Markdown
```

---

## 环境

- **Python**: 3.10
- **TensorFlow**: 2.21.0
- **Conda 环境**: `tf10`

---

## 5.1 花卉图像分类（迁移学习 + TFLite 部署）

### 简介

使用 **MobileNetV2** 作为预训练骨干网络，通过迁移学习的方式在 TensorFlow 官方花卉数据集上进行微调，训练一个 5 类花卉分类器，并将模型转换为 **TFLite** 格式以便在移动端/边缘设备部署。

### 数据集

| 属性 | 说明 |
|------|------|
| 来源 | [TensorFlow 官方 flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) |
| 类别数 | 5（daisy 雏菊、dandelion 蒲公英、roses 玫瑰、sunflowers 向日葵、tulips 郁金香） |
| 总样本量 | 3,670 张图片 |
| 划分 | 训练集 80% / 验证集 10% / 测试集 10% |

### 模型架构

| 组件 | 说明 |
|------|------|
| 输入层 | 224 x 224 x 3 RGB 图像 |
| 骨干网络 | MobileNetV2（ImageNet 预训练，`include_top=False`，冻结参数） |
| 分类头 | Dropout(0.2) -> Dense(5, softmax) |
| 可训练参数 | 6,405（仅分类头） |
| 总参数量 | 2,264,389 (~8.64 MB) |

### 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 5 |
| Batch Size | 32 |
| 学习率 | 1e-3 |
| 优化器 | Adam |
| 损失函数 | SparseCategoricalCrossentropy |

### 训练结果

| 指标 | 值 |
|------|-----|
| 最终验证准确率 | ~91.36% |
| 测试集准确率 | **88.64%** |
| 测试集 Loss | 0.3090 |

### TFLite 转换与部署

模型支持多种量化方式导出：

- **dynamic**（默认）：动态范围量化，最常用且容易成功
- **float16**：半精度浮点，适合部分移动端/GPU 场景
- **int8**：全整数量化，体积最小但需要代表性数据集校准
- **none**：不量化，保留原始浮点精度

导出产物位于 [`exported_flower_model/`](5.1/exported_flower_model/) 目录：
- `flower_classifier.keras` — Keras 原始模型，可用于继续训练或重新转换
- `model.tflite` — TFLite 量化模型，可直接用于边缘设备推理
- `labels.txt` — 类别标签映射文件

### 快速推理测试

TFLite 模型在测试集上的抽样推理结果：

```
真实类别=daisy,      预测类别=daisy
真实类别=tulips,     预测类别=tulips
真实类别=sunflowers, 预测类别=sunflowers
真实类别=daisy,      预测类别=daisy
真实类别=sunflowers, 预测类别=sunflowers
```

---

## 5.2 石头剪刀布手势识别（自定义 CNN）

### 简介

从零构建一个 **自定义卷积神经网络（CNN）**，对石头-剪刀-布（Rock-Paper-Scissors）手势图片进行三分类。使用 `ImageDataGenerator` 进行数据增强以提升模型泛化能力。

### 数据集

| 属性 | 说明 |
|------|------|
| 来源 | [Laurence Moroney's RPS Dataset](https://storage.googleapis.com/learning-datasets/rps.zip) |
| 类别数 | 3（rock 石头、paper 布、scissors 剪刀） |
| 训练集 | 2,520 张（每类 840 张） |
| 测试集 | 372 张 |
| 图片尺寸 | 150 x 150 |

### 模型架构（自定义 CNN）

| 层 | 配置 | 输出形状 |
|----|------|----------|
| Conv2D | 64 filters, 3x3, relu | 148x148x64 |
| MaxPooling2D | 2x2 | 74x74x64 |
| Conv2D | 64 filters, 3x3, relu | 72x72x64 |
| MaxPooling2D | 2x2 | 36x36x64 |
| Conv2D | 128 filters, 3x3, relu | 34x34x128 |
| MaxPooling2D | 2x2 | 17x17x128 |
| Conv2D | 128 filters, 3x3, relu | 15x15x128 |
| MaxPooling2D | 2x2 | 7x7x128 |
| Flatten | — | 6272 |
| Dropout | 0.5 | 6272 |
| Dense | 512, relu | 512 |
| Dense | 3, softmax | 3 |

- **总参数量**: 3,473,475 (~13.25 MB)
- **全部可训练**

### 数据增强

训练集使用了丰富的数据增强策略：

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 25 |
| Batch Size | 126 |
| Steps per epoch | 20 |
| Validation steps | 3 |
| 优化器 | RMSprop |
| 损失函数 | CategoricalCrossentropy |

### 训练结果

| 指标 | 值 |
|------|-----|
| 最终训练准确率 | ~97.18% |
| 最终验证准确率 | **~100%** |
| 最终验证 Loss | ~0.0046 |
| 模型保存格式 | `rps.h5`（HDF5） |

> 注：验证集准确率达到 100% 可能表明验证集规模较小（仅 372 张），模型存在一定过拟合风险。实际应用中建议扩大验证集或使用交叉验证。

---

## 两个实验的对比

| 对比维度 | 5.1 花卉分类 | 5.2 手势识别 |
|---------|-------------|-------------|
| 方法 | 迁移学习（MobileNetV2） | 自定义 CNN（从头训练） |
| 类别数 | 5 | 3 |
| 图片尺寸 | 224 x 224 | 150 x 150 |
| 参数量 | 2.26M（大部分冻结） | 3.47M（全部可训练） |
| 数据增强 | 无 | 丰富（旋转/平移/缩放/翻转等） |
| 模型导出 | Keras + TFLite（支持量化） | HDF5 (.h5) |
| 测试准确率 | 88.64% | ~100%（验证集） |
| 适用场景 | 迁移学习入门、边缘端部署 | CNN 基础实践、数据增强演示 |

---

## 使用说明

1. 安装依赖：确保已安装 TensorFlow 2.x 及相关库（numpy, matplotlib 等）
2. 进入对应子目录，用 Jupyter 打开 `.ipynb` 文件：
   ```bash
   jupyter notebook 5.1/shiyan5.1.ipynb
   jupyter notebook 5.2/shiyan5.2.ipynb
   ```
3. 按顺序执行每个 Cell 即可复现实验结果
