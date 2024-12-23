# Face_Recognize_System
## 项目介绍
### 个人小型项目：人脸识别系统，初衷为实践了解CNN卷积神经网络的运作原理。
目前项目实现了根据读取用户图片或预训练模型来进行对摄像机内人脸进行实时识别，其他样本采用LFW人脸数据集。功能单一，无法进行多用户的识别分类。
## 环境配置
### 项目运行需安装以下第三方库
flask，
cv2，
dlib，
numpy，
tensorflow

可通过如下指令安装
```
pip install ***
```
### 项目需要打开摄像头设备

## 程序运行
### 运行文件夹下app.py文件或终端目录文件夹下键入
```
python app.py
```
程序运行后，在编程软件的终端或电脑终端，将会出现
```
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 554-089-377
```
按住CTRL键点击http://127.0.0.1:5000打开网页，或复制该网址到浏览器输入网址跳转到模型的交互网页。

## 程序使用
若第一次使用，需要按照步骤依次点击按钮，可以观察到采集用户脸部照片时出现记录脸部图片的弹窗，获取其他脸部照片时同样出现弹窗。在网页提示成功后开始训练，最终弹出摄像机获取的当前场景，其中人脸部分根据分类结果出现Yes或No字样，代表分类结果。 

当第一次训练结束后，项目会保存第一次训练模型的相关参数，再次开始识别时可以直接进行脸部识别操作。当然您也可以重新训练来更新用户脸部信息。