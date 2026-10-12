from osgeo import gdal
import numpy as np
import os


def get_raster_info(raster_path):
    """读取栅格基本信息：数据、地理变换、投影、NoData"""
    dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if not dataset:
        raise Exception(f"无法打开栅格文件：{raster_path}")

    # 读取第一波段（默认单波段处理，多波段需循环）
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    # 地理变换（左上角坐标、分辨率等）、投影
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # 栅格尺寸（行、列）
    rows, cols = data.shape

    dataset = None  # 关闭数据集
    return data, geo_transform, projection, nodata, rows, cols


def resample_raster(src_data, src_geo, dst_geo, dst_rows, dst_cols, resample_method=gdal.GRA_NearestNeighbor):
    """重采样栅格：将src_data按dst_geo（目标地理变换）调整尺寸"""
    # 创建内存数据集
    driver = gdal.GetDriverByName('MEM')
    dst_ds = driver.Create('', dst_cols, dst_rows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(dst_geo)
    dst_ds.SetProjection(projection)

    # 重采样（最近邻插值，适合分类数据/掩码，保持0值不变）
    gdal.ReprojectImage(
        src_ds=None, dst_ds=dst_ds,
        src_wkt=projection, dst_wkt=projection,
        eResampleAlg=resample_method
    )

    resampled_data = dst_ds.GetRasterBand(1).ReadAsArray()
    dst_ds = None
    return resampled_data


def mask_tif2_by_tif1(tif1_path, tif2_path, output_path):
    """
    核心函数：tif1中值为0的像素，对应设置tif2为0
    :param tif1_path: 10m分辨率栅格路径（大范围）
    :param tif2_path: 0.5m分辨率栅格路径（小范围）
    :param output_path: 输出结果路径
    """
    # 1. 读取两个栅格的基础信息
    tif1_data, tif1_geo, projection, tif1_nodata, tif1_rows, tif1_cols = get_raster_info(tif1_path)
    tif2_data, tif2_geo, _, tif2_nodata, tif2_rows, tif2_cols = get_raster_info(tif2_path)

    # 2. 验证坐标系统一致性（若不一致需先投影，此处按需求跳过）
    if projection != gdal.Open(tif2_path).GetProjection():
        raise Warning("两个栅格投影不一致，可能导致空间匹配错误！")

    # 3. 重采样tif1到tif2的分辨率和范围（关键步骤）
    # 目标地理变换：沿用tif2的左上角坐标、分辨率，确保空间对齐
    dst_geo = tif2_geo  # (x0, dx=0.5, 0, y0, 0, dy=-0.5)
    resampled_tif1 = resample_raster(tif1_data, tif1_geo, dst_geo, tif2_rows, tif2_cols)

    # 4. 生成掩码：tif1重采样后值为0的位置（排除NoData）
    mask = (resampled_tif1 == 0) & (resampled_tif1 != tif1_nodata)

    # 5. 处理tif2：重叠区域掩码位置设为0，其他保留原值
    # 先复制tif2原始数据
    output_data = tif2_data.copy()
    # 掩码位置赋值为0（非重叠区域mask为False，不影响）
    output_data[mask] = 0

    # 6. 写入输出栅格（保持tif2的分辨率、投影、格式）
    driver = gdal.GetDriverByName('GTiff')
    # 创建输出数据集（与tif2尺寸、波段数一致）
    output_ds = driver.Create(
        output_path, tif2_cols, tif2_rows, 1,
        gdal.GDT_Float32,  # 可根据原始数据类型调整（如GDT_UInt16）
        options=['COMPRESS=LZW', 'TILED=YES']  # 压缩选项
    )
    output_ds.SetGeoTransform(tif2_geo)
    output_ds.SetProjection(projection)

    # 写入数据并设置NoData
    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(output_data)
    if tif2_nodata is not None:
        output_band.SetNoDataValue(tif2_nodata)
    output_band.FlushCache()

    # 关闭数据集
    output_ds = None
    print(f"处理完成！结果保存至：{output_path}")


# -------------------------- 执行代码 --------------------------
if __name__ == "__main__":
    # 替换为你的文件路径
    tif1_path = r"/zhangx/data/UAV20260311/sen10_guangdong/classified-Guangdong-2024-rice-WGS84-v1.tif"
    tif2_path = r"/zhangx/data/UAV20260311/bigtif/img1_tile0003_y0_x60000_checkpoint-08e-val_loss_0.2610--val_accuracy_0.9015.h5predict.tif"
    output_path = r"/zhangx/data/UAV20260311/bigtif/img1_tile0003_y0_x60000_processed.tif"

    # 运行处理
    mask_tif2_by_tif1(tif1_path, tif2_path, output_path)