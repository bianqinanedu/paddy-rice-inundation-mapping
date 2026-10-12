# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import fnmatch
import platform
import scipy.io as sio

patch_size = 256
img_w = patch_size
img_h = patch_size
sys_info = platform.system()
if sys_info == 'Windows':
    image_sets = ['E:/landsat/LT05_L2SP_120038_19950420_123457_train.mat']
    label_sets = ['E:/landsat/LT05_L2SP_120038_19950420_123457_train_label.mat']
    filePath_sets = ['E:/landsat/patch/landsat05_1995/']
else:
    image_sets = [
                    '/zhangx/data/UAV20260311/src/ROI/changmocun2yanmo_clip.tif',
                  '/zhangx/data/UAV20260311/src/ROI/狮岭村3掩膜_clip.tif',
                  '/zhangx/data/UAV20260311/src/ROI/狮岭村1+2掩膜_clipnew.tif',
                  '/zhangx/data/UAV20260311/src/ROI/狮岭镇4+5掩膜_clip.tif',
                  '/zhangx/data/UAV20260311/src/ROI/xinhe1_clip.tif',

                  ]

    label_sets = [
                    '/zhangx/data/UAV20260311/src/ROI/changmo_clip_label2.tif',
                  '/zhangx/data/UAV20260311/src/ROI/shilingcun3_Clip_label.tif',
                  '/zhangx/data/UAV20260311/src/ROI/狮岭村1+2掩膜_clipnew_labelnew2347.tif',
                  '/zhangx/data/UAV20260311/src/ROI/shilingcun45_clip_label.tif',
                  '/zhangx/data/UAV20260311/src/ROI/xinhe_clip_label.tif',
                  ]

    filePath_sets = [
                    '/zhangx/data/UAV20260311/patch_roi/changmocun2/',
                         '/zhangx/data/UAV20260311/patch_roi/shilingcun3/',
                     '/zhangx/data/UAV20260311/patch_roi/shilingcun12/',
                     '/zhangx/data/UAV20260311/patch_roi/shilingcun45/',
                     '/zhangx/data/UAV20260311/patch_roi/xinhe/',
                     ]


    image_sets = [
                    '/zhangx/data/UAV20260311/20260422_band8_class1/lechangAsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/lechangBsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/lechangCsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/pingyuanAsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/pingyuanBsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanAsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanBsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanCsrc_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanDsrc_Clip.tif',
                  ]

    label_sets = [
                    '/zhangx/data/UAV20260311/20260422_band8_class1/lechangAlabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/lechangBlabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/lechangClabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/pingyuanAlabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/pingyuanBlabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanAlabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanBlabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanClabel_Clip.tif',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/taishanDlabel_Clip.tif',
                  ]

    filePath_sets = [
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/',
                    '/zhangx/data/UAV20260311/20260422_band8_class1/patch/'
                     ]

    image_sets = [
        '/zhangx/data/UAV20260311/20260422_band8_class1/img1_tile0003_y0_x60000_Clip.tif',
    ]

    label_sets = [
        '/zhangx/data/UAV20260311/20260422_band8_class1/img1_tile0003_y0_x60000_Clip_label.tif',
    ]

    filePath_sets = [
        '/zhangx/data/UAV20260311/20260422_band8_class1/patch/add_taishan/',
    ]

    image_sets = [
                    '/zhangx/data/UAV20260311/20260522/src/changmocun2yanmo.tif',
                  '/zhangx/data/UAV20260311/20260522/src/changmocun2yanmo_resample.tif',
                  '/zhangx/data/UAV20260311/20260522/src/shilingcun12.tif',
                  '/zhangx/data/UAV20260311/20260522/src/shilingcun3.tif',
                  '/zhangx/data/UAV20260311/20260522/src/shilingcun45.tif',
                 '/zhangx/data/UAV20260311/20260522/src/xinhe1.tif']

    label_sets = [
                      '/zhangx/data/UAV20260311/20260522/src/changmocun2yanmo_label.tif',
                  '/zhangx/data/UAV20260311/20260522/src/changmocun2yanmo_resample_label.tif',
                  '/zhangx/data/UAV20260311/20260522/src/shilingcun12_label.tif',
                  '/zhangx/data/UAV20260311/20260522/src/shilingcun3_label.tif',
                  '/zhangx/data/UAV20260311/20260522/src/shilingcun45_label.tif',
                 '/zhangx/data/UAV20260311/20260522/src/xinhe1_label.tif',

    ]

    filePath_sets = [
                    '/zhangx/data/UAV20260311/20260522/patch_roi2/changmocun2/',
                    '/zhangx/data/UAV20260311/20260522/patch_roi2/changmocun2_resample/',
                    '/zhangx/data/UAV20260311/20260522/patch_roi2/shilingcun12/',
                    '/zhangx/data/UAV20260311/20260522/patch_roi2/shilingcun3/',
                     '/zhangx/data/UAV20260311/20260522/patch_roi2/shilingcun45/',
                     '/zhangx/data/UAV20260311/20260522/patch_roi2/xinhe1/',
                     ]
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

def featureNormalize(X, type):
    # type==1 x = (x-mean)/std(x)
    # type==2 x = (x-max(x))/(max(x)-min(x))
    if type == 1:
        mu = np.mean(X, 0)
        X_norm = X - mu
        sigma = np.std(X_norm, 0)
        X_norm = X_norm / sigma
        return X_norm
    elif type == 2:
        minX = np.min(X, 0)
        maxX = np.max(X, 0)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX)
        return X_norm


def stretchImg(data, lower_percent=0.6, higher_percent=99.4):
    n = data.shape[2]
    out = np.zeros_like(data, dtype=np.uint8)
    for i in range(n):
        a = 0
        b = 255
        c = np.percentile(data[:, :, i], lower_percent)
        d = np.percentile(data[:, :, i], higher_percent)
        t = a + (data[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out


def mirror_transform(img, img_label, img_8bit):
    img_list, label_list, img_list_8bit = [], [], []
    img_k = patch_size  # img.shape[1]

    imgs_Result = np.zeros((5,) + img.shape, dtype=np.float32)
    label_Result = np.zeros((5,) + img_label.shape, dtype=np.float32)
    imgs_Result_8bit = np.zeros((5,) + img_8bit.shape, dtype=np.uint8)

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
            label_Result[0, ii, jj] = img_label[ii, jj]  # 原图
            label_Result[1, img_k - 1 - ii, jj] = img_label[ii, jj]  # 镜像1
            label_Result[2, ii, img_k - 1 - jj] = img_label[ii, jj]  # 镜像2
            label_Result[3, jj, ii] = img_label[ii, jj]  # 镜像3
            label_Result[4, img_k - 1 - jj, img_k - 1 - ii] = img_label[ii, jj]  # 镜像4

    for ii in range(img_k):
        for jj in range(img_k):
            imgs_Result_8bit[0, ii, jj, :] = img_8bit[ii, jj, :]  # 原图
            imgs_Result_8bit[1, img_k - 1 - ii, jj, :] = img_8bit[ii, jj, :]  # 镜像1
            imgs_Result_8bit[2, ii, img_k - 1 - jj, :] = img_8bit[ii, jj, :]  # 镜像2
            imgs_Result_8bit[3, jj, ii, :] = img_8bit[ii, jj, :]  # 镜像3
            imgs_Result_8bit[4, img_k - 1 - jj, img_k - 1 - ii, :] = img_8bit[ii, jj, :]  # 镜像4

    for i in range(5):
        img_list.append(imgs_Result[i])
        label_list.append(label_Result[i])
        img_list_8bit.append(imgs_Result_8bit[i])
    return img_list, label_list, img_list_8bit


def mirror_transform_mat(img, img_label, img_8bit):
    img_list, label_list, img_list_8bit = [], [], []
    img_k = patch_size  # img.shape[1]

    imgs_Result = np.zeros((5,) + img.shape, dtype=np.float32)
    label_Result = np.zeros((5,) + img_label.shape, dtype=np.float32)
    imgs_Result_8bit = np.zeros((5,) + img_8bit.shape, dtype=np.uint8)

    #    img_k = 13  #img_k = 9
    for ii in range(img_k):
        for jj in range(img_k):
            imgs_Result[0, ii, jj, :] = img[ii, jj, :]  # 原图
            imgs_Result[1, img_k - 1 - ii, jj, :] = img[ii, jj, :]  # 镜像1
            imgs_Result[2, ii, img_k - 1 - jj, :] = img[ii, jj, :]  # 镜像2
            imgs_Result[3, jj, ii, :] = img[ii, jj, :]  # 镜像3
            imgs_Result[4, img_k - 1 - jj, img_k - 1 - ii, :] = img[ii, jj, :]  # 镜像4
    for ii in range(img_k):
        for jj in range(img_k):
            label_Result[0, ii, jj] = img_label[ii, jj]  # 原图
            label_Result[1, img_k - 1 - ii, jj] = img_label[ii, jj]  # 镜像1
            label_Result[2, ii, img_k - 1 - jj] = img_label[ii, jj]  # 镜像2
            label_Result[3, jj, ii] = img_label[ii, jj]  # 镜像3
            label_Result[4, img_k - 1 - jj, img_k - 1 - ii] = img_label[ii, jj]  # 镜像4

    for ii in range(img_k):
        for jj in range(img_k):
            imgs_Result_8bit[0, ii, jj, :] = img_8bit[ii, jj, :]  # 原图
            imgs_Result_8bit[1, img_k - 1 - ii, jj, :] = img_8bit[ii, jj, :]  # 镜像1
            imgs_Result_8bit[2, ii, img_k - 1 - jj, :] = img_8bit[ii, jj, :]  # 镜像2
            imgs_Result_8bit[3, jj, ii, :] = img_8bit[ii, jj, :]  # 镜像3
            imgs_Result_8bit[4, img_k - 1 - jj, img_k - 1 - ii, :] = img_8bit[ii, jj, :]  # 镜像4

    for i in range(5):
        img_list.append(imgs_Result[i])
        label_list.append(label_Result[i])
        img_list_8bit.append(imgs_Result_8bit[i])
    return img_list, label_list, img_list_8bit


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


def data_augment_tif(xb, yb, zb):
    xb_list,yb_list, zb_list = mirror_transform(xb, yb, zb)
    return xb_list, yb_list, zb_list


def creat_dataset_tif(image_num=1000, mode='original',data_mode='tif'):

    image_each = [150, 150, 150, 150, 150]
    g_count = 0

    for i in tqdm(range(len(image_sets))):
        filePath_image = filePath_sets[i]
        print(filePath_image)
        if not os.path.exists(filePath_image + 'SegmentationClass_' + mode + '_' + str(patch_size) + '/png/'):
            os.makedirs(filePath_image + 'SegmentationClass_' + mode + '_' + str(patch_size) + '/png/')
        if data_mode == 'tif':
            if not os.path.exists(filePath_image + 'TifImages_' + mode + '_' + str(patch_size) + '/'):
                os.makedirs(filePath_image + 'TifImages_' + mode + '_'  + str(patch_size) + '/')

        fw1 = open(filePath_image + 'img_list.txt', 'w')
        fw2 = open(filePath_image + 'label_list.txt', 'w')

        # block_size = 6000
        print('creating dataset...')
        #image_each = image_num / len(image_sets)
        #image_each = 500#image_num

        if data_mode == 'tif':
            print(image_sets[i])
            data1 = gdal.Open(image_sets[i])
            im_width = data1.RasterXSize
            im_height = data1.RasterYSize  # 栅格矩阵的行数
            src_img = data1.ReadAsArray(xoff=0, yoff=0, xsize=im_width, ysize=im_height)  #(c, height, width)
            im_geotrans = data1.GetGeoTransform()  # 仿射矩阵
            im_proj = data1.GetProjection()  # 地图投影信息
            data2 = gdal.Open(label_sets[i])
            label_img = data2.ReadAsArray(xoff=0, yoff=0, xsize=data2.RasterXSize, ysize=data2.RasterYSize)  #(c, height, width)

            label_img[label_img >8] = 0
            label_img[label_img <1] = 0

            image_8bit = np.transpose(src_img, (1,2,0))

        # Print the size of the arrays
        print("Multispectral image shape: ", src_img.shape)
        print("0-1 image shape: ", label_img.shape)

        X_height = im_height
        X_width = im_width

        np.random.seed(134)
        huachuang = 2  #2样本不均衡  1样本不管是否均衡   0 随机
        if huachuang == 1:
            for i_height in range(0, X_height, patch_size):
                for j_width in range(0, X_width, patch_size):
                    if data_mode == 'tif':
                        src_roi = src_img[:, i_height: i_height + img_h, j_width: j_width + img_w]
                    elif data_mode == 'mat':
                        src_roi = src_img[i_height: i_height + img_h, j_width: j_width + img_w, :]
                    label_roi = label_img[i_height: i_height + img_h, j_width: j_width + img_w]
                    image_8bit_roi = image_8bit[i_height: i_height + img_h, j_width: j_width + img_w, :]
                    src_roi_list, label_roi_list, src_roi_list_8bit = [], [], []

                    # 跳过尺寸不完整的patch
                    if label_roi.shape[0] < img_h or label_roi.shape[1] < img_w:
                        continue

                    # ===================== 新增：标签有效占比判断（>30%才保留）=====================
                    total_pixel = label_roi.size  # 总像素数
                    positive_pixel = (label_roi > 0).sum()  # 标签>0的像素数
                    positive_ratio = positive_pixel / total_pixel  # 占比

                    # 占比≤30%，直接跳过，不保存
                    # if positive_ratio <= 0.1:
                    #     continue
                    if np.sum(image_8bit_roi) < 1e-6:
                        continue
                    # ============================================================================

                    if mode == 'augment':
                        src_roi_list, label_roi_list, src_roi_list_8bit = data_augment_tif(src_roi, label_roi,
                                                                                           image_8bit_roi)
                    else:
                        src_roi_list.append(src_roi)
                        label_roi_list.append(label_roi)
                        src_roi_list_8bit.append(image_8bit_roi)

                    for i in range(len(src_roi_list)):
                        src_roi_sub = src_roi_list[i]
                        label_roi_sub = label_roi_list[i]

                        # 保存图像和标签
                        write_img((filePath_image + 'TifImages_%s_%d/%d.tif' % (mode, patch_size, g_count)), im_proj,
                                  im_geotrans, src_roi_sub)
                        write_img((filePath_image + 'SegmentationClass_%s_%d/%d.tif' % (mode, patch_size, g_count)), im_proj,
                                  im_geotrans, label_roi_sub)
                        g_count += 1
        elif huachuang == 2:
            # 【先定义你的多分类类别】（根据实际标签修改）
            RARE_CLASSES = [1,2,4,8]  # 稀有类别ID（比如水体、小地物，可填多个）

            for i_height in range(0, X_height, patch_size):
                for j_width in range(0, X_width, patch_size):
                    if data_mode == 'tif':
                        src_roi = src_img[:, i_height: i_height + img_h, j_width: j_width + img_w]
                    label_roi = label_img[i_height: i_height + img_h, j_width: j_width + img_w]
                    image_8bit_roi = image_8bit[i_height: i_height + img_h, j_width: j_width + img_w, :]
                    src_roi_list, label_roi_list, src_roi_list_8bit = [], [], []

                    # 1. 跳过尺寸不完整的patch
                    if label_roi.shape[0] < img_h or label_roi.shape[1] < img_w:
                        continue
                    if len(np.shape(src_roi))<3:
                        break
                    # 2. 【多分类类别均衡筛选】核心逻辑
                    total_pixel = label_roi.size
                    # 统计稀有类别像素占比
                    rare_pixel = sum([(label_roi == cls).sum() for cls in RARE_CLASSES])
                    rare_ratio = rare_pixel / total_pixel
                    # 统计普通类别像素占比
                    # common_pixel = sum([(label_roi == cls).sum() for cls in COMMON_CLASSES])
                    # common_ratio = common_pixel / total_pixel

                    # 规则1：含稀有类别 → 直接保存（优先保障稀有样本）
                    # if rare_ratio > 0 and common_ratio > 0.1:
                    if rare_ratio > 0:
                        pass
                        print(np.max(label_roi))
                    # 规则2：普通类别 >30% → 保存
                    # elif common_ratio > 0.3:
                    #     pass
                    # 规则3：纯背景/占比不足 → 跳过（减少背景冗余）
                    else:
                        continue

                    # 3. 数据增强（仅对稀有类别patch做增强，翻倍稀有样本）
                    if mode == 'augment':
                        if rare_ratio > 0:
                            src_aug, label_aug, img8_aug = data_augment_tif(src_roi, label_roi, image_8bit_roi)
                            src_roi_list.append(src_aug)
                            label_roi_list.append(label_aug)
                            src_roi_list_8bit.append(img8_aug)

                            for i in range(len(src_roi_list[0])):

                                write_img(f'{filePath_image}TifImages_{mode}_{patch_size}/{g_count}_{i}.tif',
                                          im_proj, im_geotrans, src_roi_list[0][i])
                                write_img(f'{filePath_image}SegmentationClass_{mode}_{patch_size}/{g_count}_{i}.tif',
                                          im_proj, im_geotrans, label_roi_list[0][i])
                        else:
                            src_roi_list.append(src_roi)
                            label_roi_list.append(label_roi)
                            src_roi_list_8bit.append(image_8bit_roi)
                            for i in range(len(src_roi_list)):

                                write_img(f'{filePath_image}TifImages_{patch_size}/{g_count}_{i}.tif',
                                          im_proj, im_geotrans, src_roi_list[i])
                                write_img(f'{filePath_image}SegmentationClass_{patch_size}/{g_count}_{i}.tif',
                                          im_proj, im_geotrans, label_roi_list[i])
                        g_count += 1

        else:
            ####随机采样
            for count in range(int(image_each[i])):
                src_roi_list, label_roi_list, src_roi_list_8bit = [], [], []
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, X_height - img_h - 1)
                src_roi = src_img[:,random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
                image_8bit_roi = image_8bit[random_height: random_height + img_h, random_width: random_width + img_w, :]
                max_16bit = np.max(src_roi)
                min_16bit = np.min(src_roi)
                if max_16bit == 0:
                    continue
                if mode == 'augment':
                    src_roi_list, label_roi_list, src_roi_list_8bit = data_augment_tif(src_roi, label_roi,
                                                                                       image_8bit_roi)
                else:
                    src_roi_list.append(src_roi)
                    label_roi_list.append(label_roi)
                    src_roi_list_8bit.append(image_8bit_roi)

                for i in range(len(src_roi_list)):
                    src_roi_sub = src_roi_list[i]
                    label_roi_sub = label_roi_list[i]
                    src_roi_sub_8bit = src_roi_list_8bit[i]

                    cv2.imwrite((filePath_image + 'SegmentationClass_%s_%d/%d.png' % (mode, patch_size, g_count)),
                                label_roi_sub)
                    write_img((filePath_image + 'TifImages_%s_%d/%d.tif' % (mode, patch_size, g_count)), im_proj, im_geotrans,src_roi_sub)  # 写数据
                    g_count += 1

def write_list(filePath, list_name, subfile):
    fw = open(filePath + list_name, 'w')
    raw10_file_list = fnmatch.filter(os.listdir(filePath + subfile), '*')
    raw10_file_list.sort()
    for raw10_file in raw10_file_list:
        print(raw10_file)
        fw.write(filePath + subfile + raw10_file + '\n')
        # data_raw8 = self.data_parse.read_data_raw10(self.src_directory + image_directory + raw10_file)
        # fb = open(dst_folder + raw10_file, 'wb')
        # fb.write(data_raw8)
        # fb.close()
    fw.close()

if __name__ == '__main__':
    # creat_dataset(mode='augment')

    # ************************************************
    # mode='original'
    # mode = 'augment'

    data_mode = 'tif'
    #data_mode = 'mat'
    # creat_dataset_tif(mode='augment', data_mode='tif')
    creat_dataset_tif(mode='augment', data_mode='tif')

    #creat_dataset_mat(mode='augment')
    # ************************************************

    # creat_dataset_tif_single(mode='original')
    filePath = 'E:/ZX/Data/gaojing/xietang/'
    # create_tif_to_hdf5(filePath)
