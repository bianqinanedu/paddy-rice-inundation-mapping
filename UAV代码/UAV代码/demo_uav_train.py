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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 每5个数据的第val_index个数据为验证集
val_index = 0
upsample = True


def split_train_val_random_path(train_f, val_f, test_f, image_paths_add, label_paths_add):
    image_paths_add = glob.glob(image_paths_add)
    label_paths_add = glob.glob(label_paths_add)
    train_f_add, val_f_add, test_f_add = split_train_val_random(image_paths_add, label_paths_add, val_index, upsample)

    train_f = train_f + train_f_add
    val_f = val_f + val_f_add
    test_f = test_f + test_f_add
    return train_f, val_f, test_f


class TrainingLogger(keras.callbacks.Callback):
    """记录每个epoch的训练和验证指标到CSV文件"""

    def __init__(self, log_file_path):
        super(TrainingLogger, self).__init__()
        self.log_file_path = log_file_path
        # 创建表头
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write('epoch,accuracy,loss,val_accuracy,val_loss\n')

    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时调用"""
        if logs is None:
            logs = {}

        # 获取指标
        accuracy = logs.get('accuracy', np.nan)
        loss = logs.get('loss', np.nan)
        val_accuracy = logs.get('val_accuracy', np.nan)
        val_loss = logs.get('val_loss', np.nan)

        # 写入CSV
        with open(self.log_file_path, 'a') as f:
            f.write(f'{epoch + 1},{accuracy:.6f},{loss:.6f},{val_accuracy:.6f},{val_loss:.6f}\n')


BATCH_SIZE = 8
print('batchsize', BATCH_SIZE)

# model_name = 'deeplabv3+'
# model_name = 'deeplabv3_iou'
# model_name = 'unet'
model_name = 'deeplabv3plus_improve'
# lossmode = 'TFF'
lossmode = 'TTT'


train_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_train_260422_original.txt'
val_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_val_260422_original.txt'
test_txt = '/zhangx/data/UAV20260311/20260422_band8_class1/patch/list_test_260422_original.txt'

train_f, val_f, test_f = [], [], []
if os.path.exists(train_txt):
    # f1 = open(train_txt, 'r')
    # for line1 in f1.readlines():
    #     train_f.append(line1)
    f2 = open(val_txt, 'r')
    for line2 in f2.readlines():
        val_f.append(line2)
    f3 = open(test_txt, 'r')
    for line3 in f3.readlines():
        test_f.append(line3)
    # f1.close()
    f2.close()
    f3.close()
else:

    image_paths2 = glob.glob('/zhangx/data/UAV20260311/second_yangben/Sen20260407/TifImages_augment_256/*.tif')
    label_paths2 = glob.glob('/zhangx/data/UAV20260311/second_yangben/Sen20260407/SegmentationClass_augment_256/*.tif')

    image_paths = image_paths2
    label_paths = label_paths2

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


train_f = glob.glob('/zhangx/data/UAV20260311/20260422_band8_class1/patch/train_augment/TifImages_augment_256/*.tif')
add_f1 = glob.glob('/zhangx/data/UAV20260311/20260422_band8_class1/patch/add_taishan/TifImages_augment_256/*.tif')
add_f2 = glob.glob('/zhangx/data/UAV20260311/20260422_band8_class1/patch/add_taishan/TifImages_original_256/*.tif')

train_f = train_f + add_f1 + add_f2

random.shuffle(train_f)
random.shuffle(val_f)
random.shuffle(test_f)

img_num = len(train_f)
val_num = len(val_f)
test_num = len(test_f)
print('training samples:', img_num)  # 1827
print('validation samples:', val_num)  # 522
print('test samples:', test_num)  # 239

train_generator = generatedata_0riginal_1class(train_f, BATCH_SIZE)
val_generator = generatedata_0riginal_1class(val_f, BATCH_SIZE)
test_generator = generatedata_0riginal_1class(test_f, BATCH_SIZE)

test = next(test_generator)
(masked, label) = test
# convert class labels to on-hot encoding
print(masked.shape)
print(label.shape)


dataTime = time.strftime("%Y_%m_%d_%Hh", time.localtime())
model_path = './models/' + dataTime + '_' + model_name + '_' + lossmode + '/'
weight_path = model_path + 'weight/'
best_weight_path = model_path + 'bestweight/'

if not os.path.exists(weight_path):
    os.makedirs(weight_path)
if not os.path.exists(best_weight_path):
    os.makedirs(best_weight_path)

(b, h, w, c) = masked.shape
model = ONet(weight_filepath=None, model_name=model_name)

checkpoint_fn = os.path.join(
    best_weight_path + 'checkpoint-{epoch:02d}e-val_loss_{val_loss:.4f}--val_accuracy_{val_accuracy:.4f}.h5')
checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', mode='min', save_best_only='True')

log_path = os.path.join(best_weight_path + 'training_log.csv')

callbacks_list = [checkpoint, TrainingLogger(log_file_path=log_path)]

H = model.fit(
    train_generator,
    steps_per_epoch=img_num // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_num // BATCH_SIZE,
    epochs=200,
    callbacks=callbacks_list
)

plt.style.use("ggplot")
plt.figure()
N = 200
sys_info = platform.system()
if sys_info == 'Windows':
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
else:
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on %s Satellite Seg")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
