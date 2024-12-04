import sys
import os
import cv2
import dlib
def set_others_faces():
    input_dir = './input_img'
    output_dir = './others_faces'
    size = 64

    # 创建存储用户图片目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 生成特征提取器
    cv2.namedWindow('others_face')
    detector = dlib.get_frontal_face_detector()
    # 开始循环读取图片
    iteration = 1
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg') and iteration < 100:
                print(f"[info]     Start processing picture {iteration}th")
                img = cv2.imread(path+'/'+filename)
                gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_img, 1)
                # 获取图片中的人脸部分
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    x2 = d.bottom() if d.bottom() >0 else 0
                    y1 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    #提取保存人脸
                    face = img[x1:x2, y1:y2]
                    face = cv2.resize(face,(size,size))
                # 保存图片
                cv2.imwrite(output_dir+'/'+str(iteration)+'.jpg',face)
                iteration += 1
                
                cv2.imshow('others_face', img)
            else:
                return 1
        # 按esc停止获取他人人脸       
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            print('[info]     Stop getting others\' face!')
            return 1
    return 1