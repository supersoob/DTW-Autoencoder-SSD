from math import ceil
import matplotlib.pyplot as plt
import tables
import numpy as np
import cv2 as cv
import tables, imageio
#from random import shuffle
import sys

hdf5_path = 'test_hand.hdf5'
subtract_mean = False

# open the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='r')
#data_num = hdf5_file.root.train_img.shape[0]
test_num = hdf5_file.root.test_img.shape[0]
#images = hdf5_file.root.train_img
images_test = hdf5_file.root.test_img



hop = int(sys.argv[1])
hdf5_path2 = '{}deg_test_hand.hdf5'.format(hop)

data_shape = (0, 120, 160)
img_dtype = tables.UInt8Atom()

hdf5_file2 = tables.open_file(hdf5_path2, mode='w')
#train_storage = hdf5_file2.create_earray(hdf5_file2.root, 'train_img', img_dtype, shape=data_shape)
test_storage = hdf5_file2.create_earray(hdf5_file2.root, 'test_img', img_dtype, shape=data_shape)
#storage = hdf5_file2.create_array(hdf5_file2,'imgs',img_dtype,shape=data_shape)



# loop over batches
#for i in range(data_num):
#    h, w = images[i].shape[:2]
#    M = np.float32([[1, 0, hop], [0, 1, 0]])  # 이미지를 오른쪽으로 100, 아래로 25 이동시킵니다.
#    img_translation = cv.warpAffine(images[i], M, (w, h))
#    if i % 1000 == 0 and i > 1:
#        print(i)
#    train_storage.append(img_translation[None])

for i in range(test_num):
    h, w = images_test[i].shape[:2]
    print(h,w)
    #M = np.float32([[1, 0, 0], [0, 1, hop]])  # 이미지를 오른쪽으로 100, 아래로 25 이동시킵니다.
    #img_translation_test = cv.warpAffine(images_test[i], M, (w, h))
    M = cv.getRotationMatrix2D((w/2, h/2), hop, 1)
    img_translation_test = cv.warpAffine(images_test[i], M, (w, h))
    if i % 1000 == 0:
        print(i)
    test_storage.append(img_translation_test[None])

hdf5_file.close()
hdf5_file2.close()