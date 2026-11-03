# wjw Image Processing

基于边界感知注意力 U-Net 的行人语义分割课程实验。项目在 Penn-Fudan 行人数据集上对比标准 U-Net 与改进模型，完整保留数据划分、训练日志、模型权重、逐图测试指标和可视化结果。

## 任务描述

输入校园或城市街景 RGB 图像，输出像素级行人前景掩码。实验采用固定随机种子 42，将 170 张图像划分为 119 张训练集、26 张验证集和 25 张测试集。评价指标包括 IoU、Dice、Precision、Recall、Pixel Accuracy 和 Boundary F1。

## 数据描述

- 数据集：Penn-Fudan Database for Pedestrian Detection and Segmentation
- 来源：[University of Pennsylvania 官方页面](https://www.cis.upenn.edu/~jshi/ped_html/)
- 规模：170 张图像、345 个已标注行人；96 张拍摄于宾夕法尼亚大学周边，74 张拍摄于复旦大学周边
- 标注：`PedMasks` 中不同非零灰度值代表不同实例，本项目将其合并为“行人/背景”二分类语义掩码
- 官方压缩包 SHA-256：`9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`

下载并校验数据：

```bash
python scripts/download_data.py
```

## 代码修改与创新

本项目没有直接复制第三方项目代码；训练、数据读取、模型、损失、指标和可视化均在本仓库中实现。方法设计参考 U-Net 和 CBAM 的公开论文思想，相对基线做了以下修改：

1. 用带投影捷径的全卷积残差块替换普通双卷积块，改善深层特征的梯度传播。
2. 在三级跳跃连接前加入通道与空间联合注意力，抑制背景纹理，突出行人区域。
3. 增加独立边界预测头，从标注掩码动态生成边界监督。
4. 将单一 BCE 损失改为 `0.35 BCE + 0.55 Dice + 0.10 Boundary BCE`，同时约束区域重叠和轮廓质量。
5. 新增固定划分、确定性训练、早停、逐图测试指标、推理耗时统计和定性对比图。

## 运行方法

推荐 Python 3.11 或更高版本。

```bash
pip install -r requirements.txt
python scripts/download_data.py
python run_experiment.py
python -m unittest discover -s tests
```

单独训练或评估：

```bash
python train.py --variant baseline
python train.py --variant improved
python evaluate.py --variant improved \
  --checkpoint runs/improved/checkpoints/best.pt \
  --output runs/improved/test_metrics.json
```

若 PyTorch 检测到 CUDA，脚本会自动使用 GPU；否则使用 CPU。默认设置为 128×128、batch size 4、最多 20 轮，并依据验证集 IoU 早停。

## 结果对比

固定测试集 25 张图像的正式结果如下：

| 指标 | 基线 U-Net | 改进模型 | 相对变化 |
|---|---:|---:|---:|
| IoU | 0.6666 | **0.7244** | **+8.67%** |
| Dice | 0.7919 | **0.8318** | **+5.05%** |
| Precision | 0.8064 | **0.8150** | +1.07% |
| Recall | 0.7970 | **0.8678** | **+8.89%** |
| Pixel Accuracy | 0.9382 | **0.9477** | +1.00% |
| Boundary F1 | 0.7712 | **0.8248** | **+6.94%** |

基线最优权重来自第 19 轮，改进模型来自第 20 轮。改进模型可训练参数为 292,453，基线为 274,849；本机 CPU 平均前向耗时分别为 30.99 ms/张和 7.96 ms/张。精度提升伴随了注意力与边界分支的计算开销。

完整结果位于：

- `runs/baseline/test_metrics.json`
- `runs/improved/test_metrics.json`
- `results/training_curves.png`
- `results/qualitative_comparison.png`
- `results/demo.gif`

报告中的所有数值均从这些文件自动读取，不使用手工编造的数据。

## 项目结构

```text
wjw-image-processing/
├── data/                    # 数据说明、固定划分；原始数据默认不提交 Git
├── report/                  # 实验报告
├── results/                 # 曲线、对比图、演示动画
├── runs/                    # 配置、日志、权重和逐图指标
├── scripts/                 # 数据下载和可视化脚本
├── src/                     # 数据集、模型、损失、指标
├── tests/                   # 核心单元测试
├── train.py
├── evaluate.py
└── run_experiment.py
```

## 参考资料

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
2. Woo S, Park J, Lee J Y, Kweon I S. CBAM: Convolutional Block Attention Module. ECCV, 2018.
3. Wang L, Shi J, Song G, Shen I. Object Detection Combining Recognition and Segmentation. ACCV, 2007.
4. [TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

## 许可

代码采用 MIT License。Penn-Fudan 数据集版权与使用条件以其官方发布页面为准。

