# 数据目录

运行 `python scripts/download_data.py` 后，官方 Penn-Fudan 数据集会保存在：

```text
data/raw/PennFudanPed/
├── Annotation/
├── PedMasks/
└── PNGImages/
```

数据集来源：University of Pennsylvania, Penn-Fudan Database for Pedestrian
Detection and Segmentation：<https://www.cis.upenn.edu/~jshi/ped_html/>

官方页面说明数据集包含 170 张图像和 345 个标注行人，压缩包约 51 MB。
原始数据不直接提交到 Git，以避免仓库膨胀；课程提交压缩包中另附原始数据。

