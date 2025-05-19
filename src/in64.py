import os
import pickle
import imageio
import numpy as np


def UnPKL(dir):
    with open(dir, 'rb') as fo:
        return pickle.load(fo)


def GetImgLabelPair(dirPkl, imgSize=64):
    d = UnPKL(dirPkl)
    img = d['data']
    imgSize2 = imgSize * imgSize
    img = np.dstack((
        img[:,           :imgSize2],
        img[:,   imgSize2:2*imgSize2],
        img[:, 2*imgSize2:]))
    img = img.reshape((img.shape[0], imgSize, imgSize, 3))
    print(img.shape)
    return img, d['labels']


DIR_RAW = '/data/imagenet-64'
os.mkdir(f'{DIR_RAW}/val')
os.mkdir(f'{DIR_RAW}/train')
for i in range(1000):
    os.mkdir(f'{DIR_RAW}/val/%05d' % (i+1))
    os.mkdir(f'{DIR_RAW}/train/%05d' % (i+1))

vaImg, label = GetImgLabelPair(os.path.join(DIR_RAW, 'val_data'), 64)
for i, img in enumerate(vaImg):
    imageio.imwrite(f'{DIR_RAW}/val/%05d/%08d.png' % (label[i], i), img)

idx = 1
for i in range(1, 11):
    trImg, label = GetImgLabelPair(os.path.join(DIR_RAW, f'train_data_batch_{i}'), 64)
    for j, img in enumerate(trImg):
        imageio.imwrite(f'{DIR_RAW}/train/%05d/%08d.png' % (label[j], idx), img)
        idx += 1
