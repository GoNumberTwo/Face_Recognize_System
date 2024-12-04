import cv2
import numpy as np
import random
import sys
import os
# 建立卷积神经网络
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from load_data import getPaddingSize, read_path, load_dataset, size

user_path = './user_faces'
others_path = './others_faces'

    # train_x, test_x, train_y, test_y = train_test_split(imgs, labels, test_size=0.1, random_state=random.randint(0,100))

    # train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    # test_x = test_x.reshape(test_x.shape[0], size, size, 3)
    # #归一化
    # train_x = train_x.astype('float32')/255.0
    # test_x = test_x.astype('float32')/255.0
class Dataset:
    def __init__(self):
        #训练集
        self.train_imgs = None
        self.train_labels = None
        #验证集
        self.val_imgs = None
        self.val_labels = None
        #测试集
        self.test_imgs = None
        self.test_labels = None

        #数据集加载路径
        self.user_path = './user_face'
        self.others_path = './others_face'

        #当前库采用的维度顺序
        self.input_shape = None

    def load(self, img_rows = size, img_cols = size, img_channels = 3, nb_classes = 2):
        imgs, labels = load_dataset(self.user_path, self.others_path)

        train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=0.2, random_state=random.randint(0,100))
        _, test_imgs, _, test_labels = train_test_split(imgs, labels, test_size=0.5, random_state=random.randint(0,100))

        #当前维度顺序如果为‘th’，则输入图片数据时的顺序为：channels， rows， cols， 否则：rows， cols， channels
        #根据keras要求的为欸都顺序重组训练数据集
        if K.image_dim_ordering() == 'th':
            train_imgs = train_imgs.reshape(train_imgs.shape[0], img_channels, img_rows, img_cols)
            val_imgs = val_imgs.reshape(val_imgs.shape[0], img_channels, img_rows, img_cols)
            test_imgs = test_imgs.reshape(test_imgs.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_imgs = train_imgs.reshape(train_imgs.shape[0], img_rows, img_cols, img_channels)
            val_imgs = val_imgs.reshape(val_imgs.shape[0], img_rows, img_cols, img_channels)
            test_imgs = test_imgs.reshape(test_imgs.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        #one-hot编码将标签向量化
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        val_labels = np_utils.to_categorical(val_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
        #浮点化+归一化
        train_imgs = train_imgs.astype('float32')/255.0
        val_imgs = val_imgs.astype('float32')/255.0
        test_imgs = test_imgs.astype('float32')/255.0

        self.train_imgs = train_imgs
        self.val_imgs = val_imgs
        self.test_imgs = test_imgs
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
    
#CNN模型
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 2):
        self.model = Sequential()
        
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = dataset.input_shape))#2维卷积层
        self.model.add(Activation('relu'))                                                            #激活函数

        self.model.add(Convolution2D(32, 3, 3))                                                        #卷积层
        self.model.add(Activation('relu'))                                                             #激活函数

        self.model.add(MaxPooling2D(pool_size= (2, 2)))#池化层
        self.model.add(Dropout(0.25))#dropout层

        self.model.add(Convolution2D(64,3,3, border_mode = 'same'))#卷积层
        self.model.add(Activation('relu'))#激活函数

        self.model.add(Convolution2D(64,3,3))#卷积层
        self.model.add(Activation('relu'))#激活函数

        self.model.add(MaxPooling2D(pool_size=(2,2)))#池化层
        self.model.add(Dropout(0.25))#Dropout层

        self.model.add(Flatten())#flatten层
        self.model.add(Dense(512))#全连接层
        self.model.add(Activation('relu'))#激活函数
        self.model.add(Dropout(0.5))#Dropout层
        self.model.add(Dense(nb_classes))#全连接层
        self.model.add(Activation('softmax'))#softmax函数分类输出

        self.model.summary()

        
    def Train(self, dataset, batch_size = 20, nb_epoch = 10, data_augmentation = True):
        sgd = SGD(learning_rate=0.01, decay = 1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        if not data_augmentation:
            self.model.fit(dataset.train_imgs, dataset.train_labels, batch_size=batch_size, nb_epoch = nb_epoch, validation_data=(dataset.val_imgs, dataset.val_labels), shuffle=True)

        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center= False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False
            )
            datagen.fit(dataset.train_imgs)

        self.model.fit_generator(datagen.flow(dataset.train_imgs, dataset.train_labels, batch_size=batch_size), samples_per_epoch = dataset.train_imgs.shape[0],
                                 nb_epoch=nb_epoch, validation_data=(dataset.val_imgs, dataset.val_labels,))

    def save_model(self, file_path):
            self.model.save(file_path)
    def load_model(self, file_path):
            self.model = load_model(file_path)
        
    def evaluate(self, dataset):
            score = self.model.evaluate(dataset.test_imgs, dataset.test_labels, verbose = 1)    
            print("%s: %.2f%%" %(self.model.metrics_names[1], score[1]*100))

    def face_predict(self,img):
            if K.image_dim_ordering() == 'th' and img.shape !=(1,3, size, size):
                img = getPaddingSize(img)
                img = img.reshape((1,3,size,size))
            elif K.image_dim_ordering() != 'th' and img.shape != (1, size, size, 3):
                img = getPaddingSize(img)
                img = img.reshape((1,size, size, 3))
            img = img.astype('float32')/255.0

            result = self.model.predict_proba(img)
            print(f'[info]     Result:{result}')
            return result

def find_model():
    suffix = '.h5'
    dir = './model'
    for filename in os.listdir(dir):
        if filename.endswith(suffix):
            return 1
    return 0

def trainCNN():
    dataset = Dataset()
    model = Model()
    model.build_model(dataset)
    model.Train(dataset)
    model.save_model(file_path = './model/model.h5')
    return 1