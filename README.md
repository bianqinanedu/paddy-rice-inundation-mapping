# Paddy Rice Inundation Mapping — Data and Code

This repository accompanies the paper:

> **Paddy Rice Inundation Mapping Based on Multi-Resolution Remote Sensing Imagery and Semantic Segmentation with a Dual-Attention and Edge-Aware Attention Network** (*Remote Sensing*, under review)

## Contents

| Path | Description |
|---|---|
| `UAV代码/` | DAEA-Net semantic segmentation model (TensorFlow-Keras): training, inference, and evaluation scripts (`model256.py` implements the AMSA, EAAB and DualA modules) |
| `PH-MCW/` | PH-MCW (Prior-Histogram Marker-Controlled Watershed) SAR water-extraction code (`ph_mcw.py`, README and requirements) |
| `paddyricemap-*.shp` | Final paddy rice maps of the four study areas (Yingde, Taishan, Lechang, Pingyuan) as polygon shapefiles (raster originals are omitted because of their file size; the polygons preserve the field-scale boundaries) |
| `paddyriceinundation-*.shp` | Final flood-inundated paddy rice maps of the four events (shapefile polygons) |

## How to use

### DAEA-Net (rice extraction)

```bash
cd UAV代码
pip install tensorflow numpy opencv-python gdal
python demo_uav_train.py        # training on 256 x 256 tiles
python demo_test_tif3.py        # inference on large GeoTIFFs
```

The model takes an eight-band input: RGB (0.5 m) + four quarterly NDVI (10 m) + DEM (5 m),
all resampled to 0.5 m by nearest-neighbour interpolation (early fusion).

### PH-MCW (flood-water extraction)

```bash
cd PH-MCW
pip install -r requirements.txt
python ph_mcw.py --input sar.tif --prior 3.2 --aoi aoi.tif --output water.tif
```

Prior water percentiles: Yingde 3.2%, Lechang 2.2%, Pingyuan 2.6%, Taishan 8.4%.
Inundated paddy rice = water increment (during-event minus pre-event) within the rice map.

## Data availability

* Sentinel-1/Sentinel-2: Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu)
* GF-3 SAR: China Platform of Earth Observation System (https://www.cpeos.org.cn/)
* VHR optical imagery: 2023 basemap of Tianditu Guangdong (https://guangdong.tianditu.gov.cn/), produced from the Jilin-1 constellation
* Flood reference datasets: National Earth Observation Data Center (https://www.chinageoss.cn/)
* AIR-PolSAR-Seg-2.0: https://doi.org/10.12000/JR24237
* CHM_PRE V2 precipitation: https://doi.org/10.11888/Atmos.tpdc.300523

## Citation

If you use the code or data, please cite the paper above.
