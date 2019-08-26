from math import ceil
import matplotlib.pyplot as plt
import tables
import numpy as np
import cv2 as cv
import tables, imageio
from random import shuffle
import sys

hdf5_path = '45deg_test_hand.hdf5'
subtract_mean = False
# open the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='r')
# subtract the training mean
# Total number of samples
data_num = hdf5_file.root.test_img.shape[0]
images = hdf5_file.root.test_img

# loop over batches
for i in range(data_num):
    #h, w = images[i].shape[:2]
    #M = np.float32([[1, 0, 25], [0, 1, 0]])  # 이미지를 오른쪽으로 100, 아래로 25 이동시킵니다.
    #img_translation = cv.warpAffine(images[i], M, (w, h))
    #cv.imshow("translation", img_translation)

    cv.imshow("original",images[i])
    cv.waitKey(0)
    #cv.destroyAllWindows()

cv.destroyAllWindows()

hdf5_file.close()