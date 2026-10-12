# PH-MCW（先验直方图标记控制分水岭）

用于从 SAR 影像提取洪水水体的算法实现，与论文
《Paddy Rice Inundation Mapping Based on Multi-Resolution Remote Sensing Imagery and
Semantic Segmentation with a Dual-Attention and Edge-Aware Attention Network》
第 3.2.3 节的描述一一对应。

## 方法要点

1. **水体种子**：后向散射低于先验水体百分位 T_water 的像元。先验比例取自对应行政区的公开土地调查数据（仅内陆水体），并经目视修正：
   - 英德 3.2%（清远 2020 年度国土变更调查）
   - 乐昌 2.2%（乐昌 2024 年度国土变更调查）
   - 平远 2.6%（平远县第三次全国国土调查）
   - 台山 8.4%（台山市第三次全国国土调查；滨海影像含大片海面，需较大调整）
2. **非水体种子**：后向散射高于 T_non-water 的像元。T_non-water 为直方图众数（256 个等宽区间，一维高斯平滑 σ = 3 区间），代表占多数的非水体表面的典型后向散射水平。
3. **双向标记控制分水岭**：直接以后向散射强度（而非梯度图）作为分水岭地形面——水体对应低"海拔"盆地，保留绝对散射水平这一最判别信息。前向在强度图上、后向在其反演图（max − intensity）上进行，两次使用相同种子集、compactness = 0。
4. **冲突仲裁**：前向判为水、后向判为非水的像元，按欧氏距离变换分配给更近的种子集。
5. **极化**：使用交叉极化通道（Sentinel-1 用 VH、GF-3 用 HV），水体镜面反射在交叉极化下明显更弱，水陆可分性更好。

## 安装

```
pip install numpy scipy scikit-image rasterio
```

## 用法

```
# 仅提取水体
python ph_mcw.py --input sar.tif --prior 3.2 --output water.tif

# 叠加 AOI（潜在洪泛区）滤波（1 = 候选区）
python ph_mcw.py --input sar.tif --prior 3.2 --aoi aoi.tif --output water.tif
```

其他参数：`--min-valid`（有效后向散射下限，dB 产品默认 −100）、`--min-size`（去除小于该像素数的连通域，默认 64）。

## 重要说明

- 洪水淹没水稻制图需**变化检测**：用同生长季的事件前影像与事件期影像各跑一次 PH-MCW，
  水稻图范围内相减，只保留新增水体为受淹水稻。
- 先验百分位应视为**可迁移的初始约束**而非普适常数：换传感器或换研究区时，
  建议在峰值区间内（约 0.3%–0.6%）先做局部敏感性测试再确定取值。
