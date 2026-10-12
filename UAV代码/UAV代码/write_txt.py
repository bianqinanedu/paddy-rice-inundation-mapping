import os
import numpy as np
import random
filepath_list = ['/zhangx/data/GF2/patch/wheat_ROI2_20220308/TifImages_augment_256/']
list = []
fw1 = open('./dataset/list_wheat_ROI2_20220308.txt', 'w')
fw2 = open('./dataset/list_wheat_ROI2_20220308_ramdom.txt', 'w')
for filepath_sub in filepath_list:
    for pic in os.listdir(filepath_sub):
        if pic == 'png' or pic.find('enp') >= 0:  # 排除掉png文件夹
            continue
        list.append(pic)
        fw1.write(filepath_sub + pic + '\n')
n = len(list)
np.random.seed(7)
random.shuffle(list)

for i in list:
    fw2.write(filepath_sub + i + '\n')

fw1.close()
fw2.close()
print('finish!!')