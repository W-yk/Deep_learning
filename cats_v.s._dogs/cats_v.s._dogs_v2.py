import cv2
import numpy as np
from keras.utils import normalize

def norm(img):
    
    img_rs=np.resize(img,(224*224*3))
    img_norm=normalize(img_rs,axis=0)
    img_f=np.resize(img,(224,224,3))

    return img_f
def load_data(path):
    """
    load the dataset
    args:
        path: the path to the pictures
    return:
        X_train,Y_train
    """

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
            img = np.array(cv2.imread(path + imgs[i]))
            img_rs=cv2.resize(img,(224,224))
            data[i, :, :, :] = norm(img_rs)
            label[i] = 0
            times +=1


        else:

            img = np.array(cv2.imread(path + imgs[i]))
            img_rs=cv2.resize(img,(224,224))
            data[1000+time, :, :, :] = norm(img_rs)
            label[1000+time] = 1
            time +=1
            if time == 1000:
                break

    return data,label

    

# model building and training
X_train,Y_train=load_data("C:/Git/Deep_learning/cats_and_dogs/train/")
print(X_train.shape,Y_train.shape)


from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import keras
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
checkpoint=keras.callbacks.ModelCheckpoint("C:/Git/Deep_learning/cats_and_dogs/weights/weights.hdf5", verbose=1, save_weights_only=False)
try:
    model.load_weights("C:/Git/Deep_learning/cats_and_dogs/weights/weights.hdf5")
    print("load weights succeed")
except OSError:
    print("no weights found")
model.fit(x=X_train, y=Y_train, batch_size=32, epochs=10, shuffle=True,callbacks=[checkpoint])


#predict output
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
