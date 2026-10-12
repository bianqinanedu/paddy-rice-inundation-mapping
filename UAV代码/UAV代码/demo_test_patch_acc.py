# coding:utf-8
import os
import cv2
from osgeo import gdal
import numpy as np
from model256 import *
import glob
from data import *
from metrics import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, jaccard_score,cohen_kappa_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 每5个数据的第val_index个数据为验证集
val_index = 0
upsample = True


BATCH_SIZE = 8
print('batchsize', BATCH_SIZE)
#model_name = 'seg_hrnet_ce'
#model_name = 'unet'
test_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_test_260422_original.txt'

test_f=[]
if os.path.exists(test_txt):
    f3 = open(test_txt, 'r')
    for line3 in f3.readlines():
        test_f.append(line3)
    f3.close()

if __name__ == '__main__':
    channel = 3
    patch_size = 256

    # weight_path = './models/2026_05_21_07h_build_innovative_unet_cce_260413/bestweight/checkpoint-51e-val_loss_0.2251--val_accuracy_0.9183.h5'
    # weight_path = './models/2026_05_24_00h_build_innovative_unet_FFT/bestweight/checkpoint-22e-val_loss_0.1992--val_accuracy_0.9216.h5'
    # weight_path = './models/2026_05_22_09h_build_innovative_unet_TFF/bestweight/checkpoint-08e-val_loss_0.2574--val_accuracy_0.9028.h5'
    # weight_path = './models/2026_05_24_00h_build_innovative_unet_FTF/bestweight/checkpoint-175e-val_loss_0.2242--val_accuracy_0.9119.h5'
    # weight_path = './models/2026_05_24_00h_build_innovative_unet_FFT/bestweight/checkpoint-22e-val_loss_0.1992--val_accuracy_0.9216.h5'
    # weight_path = './models/2026_05_24_01h_build_innovative_unet_TTF/bestweight/checkpoint-156e-val_loss_0.2243--val_accuracy_0.9124.h5'
    # weight_path = './models/2026_05_24_01h_build_innovative_unet_TFT/bestweight/checkpoint-17e-val_loss_0.2263--val_accuracy_0.9089.h5'
    # weight_path = './models/2026_06_20_03h_build_innovative_unet_FTT/bestweight/checkpoint-63e-val_loss_0.2044--val_accuracy_0.9215.h5'
    # model_name = 'build_innovative_unet'

    # weight_path = './models/2026_05_21_09h_hrnet_keras_cce_260413/bestweight/checkpoint-27e-val_loss_0.2199--val_accuracy_0.9147.h5'
    # model_name = 'hrnet_keras'
    #
    # weight_path = './models/2026_05_21_09h_unet_cce_260413/bestweight/checkpoint-07e-val_loss_0.3049--val_accuracy_0.8821.h5'
    # model_name = 'unet'
    #
    # weight_path = './models/2026_05_21_09h_deeplabv3+_cce_260413/bestweight/checkpoint-33e-val_loss_0.2632--val_accuracy_0.9099.h5'
    # model_name = 'deeplabv3+'
    #
    weight_path = './models/2026_05_22_02h_seg_hrnet_cce_260413/bestweight/checkpoint-34e-val_loss_0.2474--val_accuracy_0.9012.h5'
    model_name = 'seg_hrnet'

    model = ONet(patch_size, patch_size, model_name=model_name)
    model.summary()
    model.load(weight_path)
    print("loading ok!")
    OA_list,IoU_list, FWIOU_list,mIOU_list, f1score0_list,f1score1_list,kappa_list = [],[],[],[],[],[],[]

    model_name_data = weight_path.split('/')[-3]
    fw1 = open('./test_acc_txt/accurcy_' + model_name_data + '.txt', 'w')
    # 循环外提前定义（原有列表下方添加两行）
    OA_list = []
    IoU_list = []
    FWIOU_list = []
    mIOU_list = []
    f1score_list = []
    kappa_list = []
    # 新增
    precision_list = []
    recall_list = []

    for i in range(len(test_f)):
        print("[%d/%d] %s" % (i + 1, len(test_f), test_f[i]))

        imgname = test_f[i].split('\n')[0]
        img, proj, geotrans, width, height = readtifdata_multioutput(imgname)
        img[img < -1e30] = 0
        if np.all(img == 0):  # if判断全0
            continue  # 跳过无效样本
        if img.ndim < 3:
            print(imgname)
            break
        img_normalized = img[:, :, 0:3] / 255.0
        img_list = []
        img_list.append(img_normalized)
        img_array = np.stack(img_list).astype('float32')
        data_predict = model.predict(img_array)
        result = (data_predict[0, :, :, 0] * 255).astype('uint8')  # 256 256
        ret, predict_binary = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        outputRaster_binary = test_f[i].split('.tif')[0] + '_' + model_name + '_FTT_predict_binary.tif'
        modelnametime = './' + test_f[i].split('/')[-3] + '/'
        if not os.path.exists(modelnametime):
            os.makedirs(modelnametime)

        savepathout = modelnametime + outputRaster_binary.split('/')[-1]
        write_img(savepathout, proj, geotrans, predict_binary)  # 写数据

        labelname = imgname.replace('TifImages_augment_256', 'SegmentationClass_augment_256')
        labelname = labelname.replace('TifImages_256', 'SegmentationClass_256')
        labelname = labelname.replace('TifImages_original_256', 'SegmentationClass_original_256')
        label = readtifdata(labelname)

        predict_binary = (predict_binary == 255).astype(int)
        #  拉直成一维
        label_all_full = label.flatten()
        predict_all_full = predict_binary.flatten()

        label_all = label_all_full
        predict_all = predict_all_full

        #  计算混淆矩阵及各精度参数
        confusionMatrix = ConfusionMatrix(2, predict_all, label_all)
        precision = precision_score(label_all, predict_all, pos_label=1)
        recall = recall_score(label_all, predict_all, pos_label=1)
        oa = accuracy_score(label_all, predict_all)
        iou = jaccard_score(label_all, predict_all, pos_label=1)
        f1score = f1_score(label_all, predict_all, pos_label=1)
        kappa = Kappa(confusionMatrix)

        print(oa)
        OA_list.append(oa)
        precision_list.append(precision)
        recall_list.append(recall)
        IoU_list.append(iou)
        f1score_list.append(f1score)
        kappa_list.append(kappa)
        # 新增两行，保存Precision、Recall

        fw1.write(('OA：%f, Precision：%f, Recall：%f, IOU：%f, f1score：%f, kappa：%f' % (oa, precision, recall, iou,f1score,kappa) + '\n'))
    OA_average = sum(OA_list) / len(OA_list)
    IoU_average = sum(IoU_list) / len(IoU_list)
    precision_average = sum(precision_list) / len(precision_list)
    recall_average = sum(recall_list) / len(recall_list)
    iou_average = sum(IoU_list) / len(IoU_list)
    f1score_average = sum(f1score_list) / len(f1score_list)
    kappa_average = sum(kappa_list) / len(kappa_list)

    print('OA_average: ' + str(OA_average))
    fw1.write('OA_average：%f\n' % (OA_average))
    fw1.write('Precision_average：%f\n' % (precision_average))
    fw1.write('recall_average：%f\n' % (recall_average))
    fw1.write('IoU_average：%f\n' % (iou_average))
    fw1.write('f1score_average：%f\n' % (f1score_average))
    fw1.write('kappa_average：%f\n' % (kappa_average))
    fw1.close()
