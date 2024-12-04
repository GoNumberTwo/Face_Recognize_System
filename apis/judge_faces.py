import tensorflow as tf
import sys
import dlib
import cv2
import random
from .train_cnn import Model

def judge_faces_camera():
    model =Model()
    model.load_model(file_path='./model/model.h5')
    color = (0,255,0)
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    while True:
        success, img = cap.read()
        if success:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets= detector(gray_img, 1)
            for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0                  
                    #提取保存人脸部分
                    face = img[x1:y1, x2:y2]
                    face = cv2.resize(face, (64,64))
                    # faces.append(face)
                    faceID = model.face_predict(face)
                    if faceID[0][0] > faceID[0][1]:
                        cv2.rectangle(img, (x1, x2), (y1, y2), color, 1)
                        cv2.putText(img, 'Me',(x1+30, x2+30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                    else:
                        cv2.rectangle(img, (x1-10, x2-10), (y1+10, y2+10), (255,0,0), 2)
                        cv2.putText(img, 'Not me',(x1+30, x2+30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2)                    
            cv2.imshow("Recognise muself", img)
            k = cv2.waitKey(10)
            if k& 0xff == 27:
                cap.release()
                cv2.destroyAllWindows()
                return 1
    return 1
        