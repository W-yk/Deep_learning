#生成数据集
from PIL  import Image,ImageDraw,ImageFont
import math

characters = "abcdefghizklmnopqrstuvwxyz"

def generate_img(letter,position,size):
    #图片宽度
    width = 40
    #图片高度
    height = 60
    #背景颜色白
    bgcolor = (255,255,255)
    #生成背景图片
    image = Image.new('RGB',(width,height),bgcolor)
    #加载字体
    font = ImageFont.truetype('C:/windows/fonts/Arial.ttf', size)
    #字体颜色黑
    fontcolor = (0,0,0)
    #产生draw对象，draw是一些算法的集合
    draw = ImageDraw.Draw(image)
    #画字体，(0，0)是起始位置
    draw.text(position,letter,font=font,fill=fontcolor)
    #释放draw
    del draw
    image.save('D:/Git/Deep_Learning/SJTU_captcha_crack/training_data/'+letter+'%d%d%d.jpg'%(position[0],position[1],size))
    return

#生成类似网站验证码字母训练集，由于采用图片分割故只使用单个字母
for letter in characters:
    for i in range(0,10):
        for j in range(8):
            for size in range(30,50,5):
                generate_img(letter,(i,j),size)




'''
处理并加载数据集
'''

import os
import cv2
import numpy as np

#获得图片名python列表
imgs=os.listdir("D:/Git/Deep_Learning/SJTU_captcha_crack/training_data")
num=len(imgs)
#创建空tensor
data=np.empty((num,40,40),dtype="float32")
label=np.empty((num,26),dtype="float32")
#构造独热编码
one_hot_code=np.eye(26)
characters = "abcdefghizklmnopqrstuvwxyz"
one_hot_dict=dict.fromkeys(characters)
for i in range(26):
    one_hot_dict[characters[i]]=one_hot_code[i]
#加载数据集
for i in range(num):
  
    img = np.array(cv2.imread('D:/Git/Deep_Learning/SJTU_captcha_crack/training_data/' + imgs[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rs=cv2.resize(img,(40,40))
    data[i, :, :] = img_rs/255
    label[i] = one_hot_dict[imgs[i][0]]
print(data.shape,label.shape)
X_train=np.reshape(data,(data.shape[0],40,40,1))


'''
构建模型（采用类似LeNet方案）
'''

from keras.models import Model
from keras.layers import Conv2D,Dense,MaxPooling2D,Input,Flatten,Dropout
import keras

X_input=Input(shape=(40,40,1))
X=Conv2D(20, (5, 5), padding="same", activation="relu")(X_input)
X=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

X=  Conv2D(20, (5, 5), padding="same", activation="relu")(X)
X=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
 
X=Flatten()(X)
X=Dense(500, activation="relu")(X)
 
X=Dense(26, activation="softmax")(X)
model = Model(inputs=X_input, outputs=X)
checkpoint=keras.callbacks.ModelCheckpoint("D:/Git/Deep_Learning/SJTU_captcha_crack/weights/weights.hdf5", verbose=1, save_weights_only=False)
try:
    model.load_weights("D:/Git/Deep_Learning/SJTU_captcha_crack/weights/weights.hdf5")
    print("load weights succeed")
except OSError:
    print("no weights found")
model.compile(optimizer=keras.optimizers.Adam(),loss="categorical_crossentropy",metrics=["accuracy"])

#使用Adam算法训练模型，并保存权重和调用tensorboard
model.fit(x=X_train, y=label, batch_size=32, epochs=200, shuffle=True,callbacks=[checkpoint,keras.callbacks.TensorBoard(log_dir='D://Git//Deep_Learning//SJTU_captcha_crack//log')])


#测试模型效果函数
import matplotlib.pyplot as plt
def im_predict():
    img = np.array(cv2.imread('C:/Users/Wyk/Desktop/a.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rs=np.reshape(cv2.resize(img,(40,40)),(1,40,40,1))
    plt.imshow(cv2.resize(img,(40,40)))
    plt.show()
    print(model.predict(img_rs))
im_predict()