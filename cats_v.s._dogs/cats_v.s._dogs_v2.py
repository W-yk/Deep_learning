import cv2
import numpy as np
from keras.utils import normalize
import os

def norm(img):
    
    img_rs=np.resize(img,(224*224*3))
    img_norm=normalize(img_rs,axis=0)
    img_f=np.resize(img,(224,224,3))

    return img_f
def load_data(path,start):
    """
    load the dataset
    args:
        path: the path to the pictures
    return:
        X_train,Y_train
    """

    data=np.empty((5000,224,224,3),dtype="float32")
    label=np.empty((5000,1))
    imgs=os.listdir(path)
    num=len(imgs)
    cats=0
    dogs=0
    
    #从编号start开始加载数据，由于笔记本内存限制问题，每次只加载5000张图片
    for i in range(num):
        if imgs[i].split('.')[0] == 'cat' and int(imgs[i].split('.')[1])>= start :
            if cats ==2500:
                continue
            img = np.array(cv2.imread(path + imgs[i]))
            img_rs=cv2.resize(img,(224,224))
            data[cats, :, :, :] = norm(img_rs)
            label[cats] = 0
            cats +=1


        elif int(imgs[i].split('.')[1])>= start:

            img = np.array(cv2.imread(path + imgs[i]))
            img_rs=cv2.resize(img,(224,224))
            data[cats+dogs, :, :, :] = norm(img_rs)
            label[cats+dogs] = 1
            dogs +=1
            if dogs == 2500:
                break

    return data,label


#build the model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
import keras
from keras.layers.core import Flatten

base_model=ResNet50(include_top=False, weights='imagenet',input_tensor=None, input_shape=None,pooling='max')

X=base_model.output
X=Dense(1024,activation='relu')(X)
X=Dense(64,activation='relu')(X)
X=Dropout(0.6)(X)
X=Dense(1,activation='sigmoid')(X)


model = Model(inputs=base_model.input, outputs=X)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])
checkpoint=keras.callbacks.ModelCheckpoint("D:/Git/Deep_Learning/cats_v.s._dogs/weights/weights.hdf5", verbose=1, save_weights_only=False)
try:
    model.load_weights("D:/Git/Deep_Learning/cats_v.s._dogs/weights/weights.hdf5")
    print("load weights succeed")
except OSError:
    print("no weights found")

    
# model  training
starts=[0,2500,5000,7500]
for start in starts:
    X_train,Y_train=load_data("D:/Git/Deep_Learning/cats_v.s._dogs/train/",start)
    print("train data starting form %d"%(start))

    model.fit(x=X_train, y=Y_train, batch_size=32, epochs=20, shuffle=True,callbacks=[checkpoint,keras.callbacks.TensorBoard(log_dir='D://Git//Deep_Learning//cats_v.s._dogs//log')])



#predict single picture output
def predict(path):
    img = np.array(cv2.imread(path))
    img_rs=cv2.resize(img,(224,224))
    img_rs=norm(img_rs)
    img_rs=np.resize(img_rs,(1,224,224,3))
    model_prediction=model.predict(img_rs)
    print("model output:%f"%(model_prediction))
    if np.squeeze(model_prediction) < 0.5:
        print("Picture:"+path+"   is a cat.")
    else:
        print("Picture:"+path+"   is a dog.")
    return
predict("C:/Git/Deep_learning/cats_and_dogs/test4.jpg")
predict("C:/Git/Deep_learning/cats_and_dogs/test5.jpg")


#predict the test set

import pandas as pd

df=pd.read_csv("D:/Git/Deep_Learning/cats_v.s._dogs/output.csv",index_col=0)


def predict(path):
    
    img_r=np.array(cv2.imread(path)).astype("float32")
    img_rs=cv2.resize(img_r,(224,224))
    img_f=np.resize(norm(img_rs),(1,224,224,3))
    prediction=model.predict(img_f)
    if prediction>0.5:
        output=1
    else:
        output=0
    return output

test_imgs=os.listdir("D:/Git/Deep_Learning/cats_v.s._dogs/test")
num=len(test_imgs)

for i in range(num):
    label=predict("D:/Git/Deep_Learning/cats_v.s._dogs/test/"+test_imgs[i])
    df.set_value(int(test_imgs[i].split('.')[0]),'label',label)
    if i%500==0:
        df.to_csv("D:/Git/Deep_Learning/cats_v.s._dogs/output.csv")
        print("%d label writed"%(i))
