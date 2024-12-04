import cv2
import dlib
import os
import sys
import random
import numpy as np

#改变图片亮度和对比度（三重）
def relight(img, light = 1,bias = 0):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(0,height):
        for j in range(0,width):
            for p in range(3):
                tmp = int(img[i,j,p]*light + bias)
                if tmp < 0:
                    tmp = 0
                elif tmp > 255:
                    tmp = 255
                img[i,j,p] = tmp
    return img
def get_user_faces():
    #变量定义
    output_dir = './user_faces'
    size = 64
    MAX_ITERATION = 100
    iteration = 0

    # 创建存储用户图片目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #调用dlib的frontal_face_detector函数作为特征提取器
    detector = dlib.get_frontal_face_detector()

    #初始化摄像头
    cv2.namedWindow('user_face')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[info]     打开摄像头失败")
        return 0
    else:
        while cap.isOpened():
            if (iteration <= MAX_ITERATION):
                print(f'[info]     Getting picture {iteration}%')
                # 读取照片
                success, img = cap.read()
                if not success:
                    break
                # 获取灰度照片
                gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # 开始进行人脸检测
                # 1，将图片放大一倍，提高识别效果，dectector函数返回图片中的人脸区域
                dets= detector(gray_img, 1)
                
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0                  
                    #提取保存人脸部分
                    face = img[x1:y1, x2:y2]
                    face = relight(face, random.uniform(0.5,1.5), random.randint(-50, 50))
                    face = cv2.resize(face, (size,size))

                    cv2.imwrite(output_dir+'/'+str(iteration)+'.jpg',face)
                    iteration += 1
                    #展示提取到的脸部部位
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, x2),(y1, y2), color, 1) 
                # 30ms内等待输入，若按esc则取消展示
                cv2.imshow('user_face', img) 
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    print('[info]     You click ESC to stop.')
                    return 1
            else:
                print("[info]     Grapping Finished!")
                cap.release()
                cv2.destroyAllWindows()
        return 1