from tensorflow import keras as keras
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model256 import *
from data import *
import numpy as np
import glob
import os
import time
import random
import platform

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
## 将图像转成0-1范围内  归一化处理
import os
import glob
from osgeo import gdal
import numpy as np
import numpy as np
def readtifdata(fileName):
    # print('reading......')
    dataset_img = gdal.Open(fileName)
    if dataset_img == None:
        print(fileName + "文件无法打开")
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    # print('读取data')
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    # print('读取完毕正在归一化')
    return img, geotrans, proj
#写文件，以写成tif为例
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

def gdal_write(img, path):   ###0-1
    # img = np.transpose(img, [2, 0, 1])
    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    print(img.shape, datatype)
    [h, w, c] = img.shape
    # print(img.shape)
    driver = gdal.GetDriverByName('GTiff')
    ndvi_img = driver.Create(path, w, h, c, datatype)
    # print(ndvi_img.shape)

    for i in range(c):
        print(i)
        ndvi_img.GetRasterBand(i + 1).WriteArray(img[:, :, i])

    del ndvi_img


def mirror_transform(img, imgpca35, imgpca5, img_label):
    img_list, label_list, imgpca35_list, imgpca5_list = [], [], [], []
    img_k = img.shape[1]

    imgs_Result = np.zeros((5,) + img.shape, dtype=np.float32)
    imgspca35_Result = np.zeros((5,) + imgpca35.shape, dtype=np.float32)
    imgspca5_Result = np.zeros((5,) + imgpca5.shape, dtype=np.float32)
    label_Result = np.zeros((5,) + img_label.shape, dtype=np.float32)

    #    img_k = 13  #img_k = 9
    for ii in range(img_k):
        for jj in range(img_k):
            imgs_Result[0, :, ii, jj] = img[:, ii, jj]  # 原图
            imgs_Result[1, :, img_k - 1 - ii, jj] = img[:, ii, jj]  # 镜像1
            imgs_Result[2, :, ii, img_k - 1 - jj] = img[:, ii, jj]  # 镜像2
            imgs_Result[3, :, jj, ii] = img[:, ii, jj]  # 镜像3
            imgs_Result[4, :, img_k - 1 - jj, img_k - 1 - ii] = img[:, ii, jj]  # 镜像4
    for ii in range(img_k):
        for jj in range(img_k):
            imgspca35_Result[0, :, ii, jj] = imgpca35[:, ii, jj]  # 原图
            imgspca35_Result[1, :, img_k - 1 - ii, jj] = imgpca35[:, ii, jj]  # 镜像1
            imgspca35_Result[2, :, ii, img_k - 1 - jj] = imgpca35[:, ii, jj]  # 镜像2
            imgspca35_Result[3, :, jj, ii] = imgpca35[:, ii, jj]  # 镜像3
            imgspca35_Result[4, :, img_k - 1 - jj, img_k - 1 - ii] = imgpca35[:, ii, jj]  # 镜像4
    for ii in range(img_k):
        for jj in range(img_k):
            imgspca5_Result[0, :, ii, jj] = imgpca5[:, ii, jj]  # 原图
            imgspca5_Result[1, :, img_k - 1 - ii, jj] = imgpca5[:, ii, jj]  # 镜像1
            imgspca5_Result[2, :, ii, img_k - 1 - jj] = imgpca5[:, ii, jj]  # 镜像2
            imgspca5_Result[3, :, jj, ii] = imgpca5[:, ii, jj]  # 镜像3
            imgspca5_Result[4, :, img_k - 1 - jj, img_k - 1 - ii] = imgpca5[:, ii, jj]  # 镜像4

    for ii in range(img_k):
        for jj in range(img_k):
            label_Result[0, ii, jj] = img_label[ii, jj]  # 原图
            label_Result[1, img_k - 1 - ii, jj] = img_label[ii, jj]  # 镜像1
            label_Result[2, ii, img_k - 1 - jj] = img_label[ii, jj]  # 镜像2
            label_Result[3, jj, ii] = img_label[ii, jj]  # 镜像3
            label_Result[4, img_k - 1 - jj, img_k - 1 - ii] = img_label[ii, jj]  # 镜像4

    for i in range(5):
        img_list.append(imgs_Result[i])
        imgpca35_list.append(imgspca35_Result[i])
        imgpca5_list.append(imgspca5_Result[i])
        label_list.append(label_Result[i])

    return img_list, imgpca35_list, imgpca5_list, label_list


def mirror_transform_tif(img):
    img_list = []
    img_k = 256
    imgs_Result = np.zeros((5,) + img.shape, dtype=np.float32)
    if img.ndim == 3:
        for ii in range(img_k):
            for jj in range(img_k):
                imgs_Result[0, :, ii, jj] = img[:, ii, jj]  # 原图
                imgs_Result[1, :, img_k - 1 - ii, jj] = img[:, ii, jj]  # 镜像1
                imgs_Result[2, :, ii, img_k - 1 - jj] = img[:, ii, jj]  # 镜像2
                imgs_Result[3, :, jj, ii] = img[:, ii, jj]  # 镜像3
                imgs_Result[4, :, img_k - 1 - jj, img_k - 1 - ii] = img[:, ii, jj]  # 镜像4
    else:
        for ii in range(img_k):
            for jj in range(img_k):
                imgs_Result[0, ii, jj] = img[ii, jj]  # 原图
                imgs_Result[1, img_k - 1 - ii, jj] = img[ii, jj]  # 镜像1
                imgs_Result[2, ii, img_k - 1 - jj] = img[ii, jj]  # 镜像2
                imgs_Result[3, jj, ii] = img[ii, jj]  # 镜像3
                imgs_Result[4, img_k - 1 - jj, img_k - 1 - ii] = img[ii, jj]  # 镜像4

    for i in range(5):
        img_list.append(imgs_Result[i])

    return img_list

def generate_aug_imgs(img):
    """生成5个差异最大的增强图像（去掉4个相近变换）"""
    img_list = []
    # 1. 原图
    img_list.append(img)
    # 2. 水平镜像（左右翻转）
    img_list.append(np.fliplr(img))
    # 3. 垂直镜像（上下翻转）
    img_list.append(np.flipud(img))
    # 4. 转置镜像（对角线翻转）
    img_list.append(np.transpose(img))
    # 5. 顺时针90°旋转
    img_list.append(np.rot90(img, k=-1))

    return img_list

# 每5个数据的第val_index个数据为验证集
val_index = 0
upsample = True

BATCH_SIZE = 8

train_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_train_260422_original.txt'
val_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_val_260422_original.txt'
test_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_test_260422_original.txt'

# train_txt = './data_txt/list_train_sen.txt'
# val_txt = './data_txt/list_val_sen.txt'
# test_txt = './data_txt/list_test_sen.txt'

train_f, val_f, test_f = [], [], []

image_paths1 = glob.glob('/zhangx/data/UAV20260311/20260422_band8_class1/patch/TifImages_original_256/*.tif')
label_paths1 = glob.glob('/zhangx/data/UAV20260311/20260422_band8_class1/patch/SegmentationClass_original_256/*.tif')

image_paths = image_paths1
label_paths = label_paths1

train_f, val_f, test_f = split_train_val_random(image_paths, label_paths, val_index, upsample)

f1 = open(train_txt, 'w')
for line1 in train_f:
    f1.write(line1 + '\n')
f2 = open(val_txt, 'w')
for line2 in val_f:
    f2.write(line2 + '\n')
f3 = open(test_txt, 'w')
for line3 in test_f:
    f3.write(line3 + '\n')

f1.close()
f2.close()
f3.close()


random.shuffle(train_f)
random.shuffle(val_f)
random.shuffle(test_f)


Result_Path = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/train_augment/'

if not os.path.exists(Result_Path + 'TifImages_augment_256/'):
    os.makedirs(Result_Path + 'TifImages_augment_256/')
if not os.path.exists(Result_Path + 'SegmentationClass_augment_256/'):
    os.makedirs(Result_Path + 'SegmentationClass_augment_256/')

num = 0
for imgname in train_f:
    imgname = imgname.strip()
    img, geotrans, proj = readtifdata(imgname)

    imgnamelabel = imgname.replace('TifImages_original_256', 'SegmentationClass_original_256')
    img_label, geotrans3, proj3 = readtifdata(imgnamelabel)

    img_list = mirror_transform_tif(img)
    label_list = mirror_transform_tif(img_label)

    # img_list = generate_aug_imgs(img)
    # label_list = generate_aug_imgs(img_label)

    for j in range(len(img_list)):
        img_list_sub = img_list[j]
        label_list_sub = label_list[j]
        imgname_base = (imgname.split('/')[-1]).split('.tif')[0] + '_' + str(j) + '.tif'

        write_img((Result_Path + 'TifImages_augment_256/' + imgname_base), proj, geotrans, img_list_sub)  # 写数据
        write_img((Result_Path + 'SegmentationClass_augment_256/' + imgname_base), proj, geotrans,
                  label_list_sub)  # 写数据

    num = num + 1
    if num % 100 == 0:
        print("%d输出完毕" % num)



img_num = len(train_f)
val_num = len(val_f)
test_num = len(test_f)
print('training samples:', img_num)  # 1827
print('validation samples:', val_num)  # 522
print('test samples:', test_num)  # 239

