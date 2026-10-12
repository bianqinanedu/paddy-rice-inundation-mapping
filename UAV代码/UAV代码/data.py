from tensorflow.keras.preprocessing.image import img_to_array
from osgeo import gdal
from pylab import *
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

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

def imgread(fileName, addNDVI=False):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    # 如果是image的话,因为label是单通道
    if(len(data.shape) == 3):
        # 添加归一化植被指数NDVI特征
        if(addNDVI):
            nir, r = data[3], data[0]
            ndvi = (nir - r) / (nir + r + 0.00001) * 1.0
            # 和其他波段保持统一,归到0-255,后面的totensor会/255统一归一化
            # 统计了所有训练集ndvi的值，最小值为0，最大值很大但是数目很少，所以我们取了98%处的25
            ndvi = (ndvi - 0) / (25 - 0) * 255
            ndvi = np.clip(ndvi, 0, 255)
            data_add_ndvi = np.zeros((5, 256, 256), np.uint8)
            data_add_ndvi[0:4] = data
            data_add_ndvi[4] = np.uint8(ndvi)
            data = data_add_ndvi
        # (C,H,W)->(H,W,C)
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    # else:
    #     data = label_class(data)
    return data


def split_train_val_random(train_image_paths, label_paths, val_index=0, upsample = True, valrate = 0.2, testrate= 0.1):
    train_set, val_set, test_set = [],[],[]
    np.random.seed(7)
    import random
    random.shuffle(train_image_paths)
    total_num = len(train_image_paths)
    val_num = int(valrate * total_num)
    test_num = int(testrate * total_num)
    for i in range(len(train_image_paths)):
        if i < val_num:
            val_set.append(train_image_paths[i])
        elif i >= val_num and i < val_num + test_num:
            test_set.append(train_image_paths[i])
        else:
            train_set.append(train_image_paths[i])
    print("Number of train images after upsample: ", len(train_set))
    print("Number of val images after upsample: ", len(val_set))
    print("Number of test images after upsample: ", len(test_set))
    return train_set, val_set, test_set

def mirror_transform_tif(img):
    img_list = []
    img_k = 512
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

def readtifdata(fileName):
    dataset_img = gdal.Open(fileName.strip())
    if dataset_img == None:
        print(fileName + "文件无法打开")
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize

    img_src = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    if len(img_src.shape) == 3:
        img_src =np.moveaxis(img_src, 0, 2)
    return img_src

def readtifdata_multioutput(fileName):
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

def generatedata_0riginal_1class(path,batchsize):
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in path:
            imgname = imgname.split('\n')[0]
            # print(imgname)
            img = readtifdata(imgname)
            img[img < 0] = 0
            img[img > 255] = 0
            if np.all(img == 0):  # if判断全0
                continue  # 跳过无效样本
            if img.ndim <3:
                print(imgname)
                break
            # labelname = imgname.replace('TifImages_1024','SegmentationClass_1024')
            labelname = imgname.replace('TifImages_augment_256','SegmentationClass_augment_256')
            labelname = labelname.replace('TifImages_256','SegmentationClass_256')
            labelname = labelname.replace('TifImages_original_256','SegmentationClass_original_256')

            label = readtifdata(labelname)

            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img = np.array(img[:,:,0:3], dtype="float") / 255.0  #####8bitarray

            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []

def generatedata_0riginal_multiclass(path,batchsize):
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in path:
            imgname = imgname.split('\n')[0]
            # print(imgname)
            img = readtifdata(imgname)
            img[img < 0] = 0
            img[img > 255] = 0
            if np.all(img == 0):  # if判断全0
                continue  # 跳过无效样本
            if img.ndim <3:
                print(imgname)
                break
            # labelname = imgname.replace('TifImages_1024','SegmentationClass_1024')
            labelname = imgname.replace('TifImages_augment_256','SegmentationClass_augment_256')
            labelname = labelname.replace('TifImages_256','SegmentationClass_256')
            labelname = labelname.replace('TifImages_original_256','SegmentationClass_original_256')

            label = readtifdata(labelname)

            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img = np.array(img, dtype="float") / 255.0  #####8bitarray

            label = np.round(label).astype(np.int32)
            label_mapping = {
                0:0,
                1:1,
                2:2,
                4:3,
                8:4
            }
            new_label = np.zeros_like(label)

            for old_val, new_val in label_mapping.items():
                new_label[label == old_val] = new_val

            # 3. 清理异常值（防止出现不在映射表里的标签）
            new_label[~np.isin(new_label, list(label_mapping.values()))] = 0  # 不在0-4里的都设为背景

            # 4. 对映射后的连续标签做 one-hot 编码（5个类别，所以num_classes=5）
            label = to_categorical(new_label, num_classes=5)
            # 此时 label 的 shape 会从 (H, W, 1) 变成 (H, W, 5)，对应5个类别的one-hot
            #******************多类别

            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []

def generatedata_0riginal_multiclass0(path,batchsize):
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in path:
            imgname = imgname.split('\n')[0]
            # print(imgname)
            img = readtifdata(imgname)
            img[img < 1] = 0
            img[img > 8] = 0
            if np.all(img == 0):  # if判断全0
                continue  # 跳过无效样本
            if img.ndim <3:
                print(imgname)
                break
            # labelname = imgname.replace('TifImages_1024','SegmentationClass_1024')
            labelname = imgname.replace('TifImages_augment_256','SegmentationClass_augment_256')
            labelname = labelname.replace('TifImages_256','SegmentationClass_256')
            labelname = labelname.replace('TifImages_original_256','SegmentationClass_original_256')

            label = readtifdata(labelname)
            img_normalized = img[:, :, 0:3] / 255.0
            # img_normalized[:, :, 3:8] = img[:, :, 3:8]

            img = img_to_array(img_normalized).astype('float32')
            label = img_to_array(label).astype('float32')
            img = np.array(img, dtype="float") / 255.0  #####8bitarray
            # img = np.array(img, dtype="float") / 65535.0  #####8bitarray

            label = np.round(label).astype(np.int32)

            #******************多类别
            # label[label > 6] = 0
            # label = to_categorical(label,7)

            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []
