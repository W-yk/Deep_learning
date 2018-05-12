import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    """
    load the dataset
    args:
        path: the path to the pictures
    return:
        X_train,Y_train
    """

    data=np.empty((2000,224,224,3),dtype="float32")
    label=np.empty((2000,))
    imgs=os.listdir(path)
    num=len(imgs)
    times=0
    time=0
    for i in range(num):

        if imgs[i].split('.')[0] == 'cat':
            if times ==1000:
                continue
            img = np.array(plt.imread(path + imgs[i]))
            img_rs=misc.imresize(img,(224,224,3))
            data[i, :, :, :] = img_rs
            label[i] = 0
            times +=1


        else:

            img = np.array(plt.imread(path + imgs[i]))
            img_rs=misc.imresize(img,(224,224,3))
            data[1000+time, :, :, :] = img_rs
            label[1000+time] = 1
            time +=1
            if time == 1000:
                break

    return data,label

from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    """
    load the dataset
    args:
        path: the path to the pictures
    return:
        X_train,Y_train
    """

    #the code is running by my laptop thus I will just use 2000 data
    data=np.empty((2000,224,224,3),dtype="float32")
    label=np.empty((2000,1))
    imgs=os.listdir(path)
    num=len(imgs)
    times=0
    time=0
    for i in range(num):

        if imgs[i].split('.')[0] == 'cat':
            if times ==1000:
                continue
            img = np.array(plt.imread(path + imgs[i]))
            img_rs=resize(img,(224,224))
            data[i, :, :, :] = img_rs
            label[i] = 0
            times +=1


        else:

            img = np.array(plt.imread(path + imgs[i]))
            img_rs=resize(img,(224,224))
            data[1000+time, :, :, :] = img_rs
            label[1000+time] = 1
            time +=1
            if time == 1000:
                break

    return data,label

    
X_train,Y_train=load_data("C:/Git/Deep_learning/cats_and_dogs/train/")
print(X_train.shape,Y_train.shape)


'''
cats and dogs recognition with transfer learning
using ResNet50 pre_train weights
'''

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers.core import Flatten

base_model=ResNet50(include_top=False, weights='imagenet',input_tensor=None, input_shape=None,pooling='max')

X=base_model.output
X=Dense(1024,activation='relu')(X)
X=Dense(64,activation='relu')(X)
X=Dense(1,activation='sigmoid')(X)


model = Model(inputs=base_model.input, outputs=X)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x=X_train, y=Y_train, batch_size=32, epochs=10, shuffle=True )   



def predict(path):
    '''
    predict a image
    args: 
        path: The path of the image
    return:
        prediction
    '''
    img = np.array(plt.imread(path))
    img_rs=resize(img,(224,224))
    img_rs=np.resize(img_rs,(1,224,224,3))
    model_prediction=model.predict(img_rs)
    print(model_prediction)
    if np.squeeze(model_prediction) < 0.5:
        print("Picture:"+path+"   is a cat.")
    else:
        print("Picture:"+path+"   is a dog.")
    return


predict("C:/Users/wyk/Desktop/test.jpg")