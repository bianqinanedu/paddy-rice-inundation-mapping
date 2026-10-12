# coding:utf-8
import os
import cv2
from osgeo import gdal
# # coding:utf-8
import numpy
numpy.object = numpy.object_
numpy.typeDict = numpy.sctypeDict

from model256 import *
import glob
from data import *
gdal.PushErrorHandler('CPLQuietErrorHandler')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gdal.UseExceptions()

def readtifdata(fileName):
    dataset_img = gdal.Open(fileName.strip())
    if dataset_img == None:
        print(fileName + "文件无法打开")
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    # print(img.shape,len(img.shape))
    if len(img.shape) == 3:
        img =np.moveaxis(img, 0, 2)
    return img,proj,geotrans,width,height

def write_img(filename,im_proj,im_geotrans,im_data):
    #gdal数据类型包括
    #gdal.GDT_Byte,
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
      datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
      datatype = gdal.GDT_UInt16
    else:
      datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
      im_bands, im_height, im_width = im_data.shape
    else:
      im_bands, (im_height, im_width) = 1,im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")      #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)       #写入仿射变换参数
    dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
    else:
      for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def testoverlap(model, img, patch_size = 1024, overlap_rate = 0.01, isroad=True):

    boder_value = int(patch_size * overlap_rate / 2)
    double_bv = boder_value * 2  ##  重叠区域长度
    stride_value = patch_size - double_bv  ##步长
    most_value = stride_value + boder_value

    print("====正在填充调整=====")

    m, n, c = img.shape
    tmp_m = (m - double_bv) // stride_value  ##计算横向数量（整除）
    new_m = tmp_m if (m - double_bv) % stride_value == 0 else tmp_m + 1  ##如果刚好能整除和不能整除
    tmp_n = (n - double_bv) // stride_value
    new_n = tmp_n if (m - double_bv) % stride_value == 0 else tmp_n + 1
    ##将图像扩展到刚好能被完全裁剪的大小，补0
    new_full_img = np.zeros((new_m * stride_value + double_bv, new_n * stride_value + double_bv, c), dtype=np.float32)
    new_full_img[:m, :n, :] = img
    ###预测的结果也是完全裁剪的大小，3个波段，thin，thick，background
    # FullPredict = np.zeros((new_m * stride_value + double_bv, new_n * stride_value + double_bv, 3), dtype=np.uint8)
    FullPredict = np.zeros((new_m * stride_value + double_bv, new_n * stride_value + double_bv), dtype=np.uint8)
    # Fullcloud = np.zeros((new_m * stride_value + double_bv, new_n * stride_value + double_bv, 3), dtype=np.uint8)
    imgdatas = np.ndarray((1, patch_size, patch_size, c), dtype=np.float32)
    print(np.max(img), np.max(new_full_img))

    print("===正在裁剪拼接===")
    newname = 1
    num = 0
    for i in range(new_m):
        for j in range(new_n):
            tmp_img = new_full_img[
                      i * stride_value:((i + 1) * stride_value + double_bv),
                      j * stride_value:((j + 1) * stride_value + double_bv),
                      :
                      ]
            # tmp_img = img_to_array(tmp_img).astype('float32')
            imgdatas[0] = tmp_img
            num += 1
            print('image------------', num, np.max(tmp_img))
            mask = model.predict(imgdatas)## 得到b, h, w, c
            # pred = np.argmax(mask[0], axis=-1).astype(np.uint8)

            # pred = (mask[0,:,:,0]*255).astype('uint8')##二分类
            pred = (mask[0,:,:,0]*255).astype('uint8')##二分类
            ret, pred = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            pred[pred == 255] = 1

            newname = newname + 1
            if i == 0 and j == 0:  # 左上角
                FullPredict[0:most_value, 0:most_value] = pred[0:most_value, 0:most_value]
            elif i == 0 and j == new_n - 1:  # 右上角
                FullPredict[0:most_value, -most_value:] = pred[0:most_value, boder_value:]
            elif i == 0 and j != 0 and j != new_n - 1:  # 第一行
                FullPredict[0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    pred[0:most_value, boder_value:most_value]

            elif i == new_m - 1 and j == 0:  # 左下角
                FullPredict[-most_value:, 0:most_value] = pred[boder_value:, :-boder_value]
            elif i == new_m - 1 and j == new_n - 1:  # 右下角
                FullPredict[-most_value:, -most_value:] = pred[boder_value:, boder_value:]
            elif i == new_m - 1 and j != 0 and j != new_n - 1:  # 最后一行
                FullPredict[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    pred[boder_value:, boder_value:-boder_value]

            elif j == 0 and i != 0 and i != new_m - 1:  # 第一列
                FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                    pred[boder_value:-boder_value, 0:-boder_value]
            elif j == new_n - 1 and i != 0 and i != new_m - 1:  # 最后一列
                FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                    pred[boder_value:-boder_value, boder_value:]
            else:  # 中间情况
                FullPredict[
                boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                    pred[boder_value:-boder_value, boder_value:-boder_value]


    predict = FullPredict[0:m, 0:n]
    return predict

def geotrans_change(geotrans):
    geotrans_list = list(geotrans)
    geotrans_list10 = geotrans_list
    if geotrans_list[1]>10:
        geotrans_list10[1] = 10.0
        geotrans_list10[5] = -10.0

    geotrans10 = tuple(geotrans_list10)
    return geotrans10

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('-i', '--input_path', type=str, default='/zhangx/data/ChinaLandsat/jiangsu_543_8bit/LC08_L2SP_119037_20230901_20230911_02_T1_SR_B543_8bit_EDSR.tif')
    parser.add_argument('-w', '--weight_path', type=str, default='./models/2024_10_25_03h_15m_unet/bestweight/checkpoint-41e-val_loss_0.0432.h5')#very good

    parser.add_argument('-p', '--patch_size', type=int, default=2048)

    args = parser.parse_args()
    input_path = args.input_path
    weight_path = args.weight_path
    patch_size = args.patch_size

    weight_path = './models/2026_04_27_04h_unet_cce_260413/bestweight/checkpoint-08e-val_loss_0.2610--val_accuracy_0.9015.h5'##260422

    weight_path = './models/2026_05_21_09h_unet_cce_260413/bestweight/checkpoint-07e-val_loss_0.3049--val_accuracy_0.8821.h5'
    model_name = 'unet'

    weight_path = './models/2026_05_21_09h_deeplabv3+_cce_260413/bestweight/checkpoint-33e-val_loss_0.2632--val_accuracy_0.9099.h5'
    model_name = 'deeplabv3+'

    weight_path = './models/2026_05_22_02h_seg_hrnet_cce_260413/bestweight/checkpoint-34e-val_loss_0.2474--val_accuracy_0.9012.h5'
    model_name = 'seg_hrnet'

    weight_path = './models/2026_05_21_07h_build_innovative_unet_cce_260413/bestweight/checkpoint-51e-val_loss_0.2251--val_accuracy_0.9183.h5'
    weight_path = './models/2026_06_20_03h_build_innovative_unet_FTT/bestweight/checkpoint-63e-val_loss_0.2044--val_accuracy_0.9215.h5'
    model_name = 'build_innovative_unet'


    weight_path = '/zhangx/code/UAV_JIANHUA/models/2026_09_22_06h_deeplabv3plus_improve_TTT/bestweight/checkpoint-55e-val_loss_0.3613--val_accuracy_0.9066.h5'
    model_name = 'deeplabv3plus_improve'


    input_path_list = [
        '/zhangx/data/UAV20260311/20260422_band8_class1/lechangAsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/lechangBsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/lechangCsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/pingyuanAsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/pingyuanBsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/taishanAsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/taishanBsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/taishanCsrc.tif',
        '/zhangx/data/UAV20260311/20260422_band8_class1/taishanDsrc.tif']

    model = ONet(patch_size, patch_size, model_name=model_name)
    model.summary()
    model.load(weight_path)

    for input_path in input_path_list:

        Test_Path = os.path.dirname(input_path)
        Result_Path = os.path.dirname(input_path)
        imgname = input_path.split('/')[-1]
        imgename_time = weight_path.split('/')[-3]
        if not os.path.exists(Result_Path):
            os.makedirs(Result_Path)
        print(input_path)

        channel = 3

        print("loading ok!")
        img, proj, geotrans, width, height = readtifdata(input_path)
        img = img[:,:,0:3] / 255.0

        img[img < -1e30] = 0
        result = testoverlap(model, img, patch_size=patch_size, overlap_rate=0.01, isroad=True)
        outputRaster_binary = input_path.split('.tif')[0] + '_' + imgename_time + 'predict.tif'
        write_img(outputRaster_binary, proj, geotrans, result)  # 写数据
        print("输出完毕")
        del img
        del result