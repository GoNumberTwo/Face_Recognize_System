import cv2
import numpy as np
import random
import sys
import os

size = 64

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)
    
    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def read_path(path, imgs, labels):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            #扩充图片边缘
            img= cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=[0,0,0])
            img = cv2.resize(img,(size, size))
            imgs.append(img)
            labels.append(path)
        

def load_dataset(user_path, others_path):
    #加载用户图片数据集
    imgs = []
    labels = []
    read_path(user_path, imgs, labels)
    read_path(others_path, imgs, labels)
    imgs = np.array(imgs)
    labels = np.array([0 if label == user_path else 1 for label in labels])

    return imgs,labels