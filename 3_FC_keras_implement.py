#First try


import keras
from keras import backend as K
from keras.layers import Input,Dense,Flatten
from keras.models import Model
#import visualization tools
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

#import the MNIST dataset
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
Y_train = keras.utils.np_utils.to_categorical(Y_train, 10)
Y_test = keras.utils.np_utils.to_categorical(Y_test, 10)
def Three_layer_model(input_shape):

    X_input=Input(input_shape)
    
    X=Flatten()(X_input)
    X=Dense(64,activation='relu')(X)
    X=Dense(64,activation='relu')(X)
    X=Dense(10,activation='softmax')(X)

    model=Model(inputs=X_input,outputs=X)
    return model

TL_model=Three_layer_model(X_train.shape[1:])
TL_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
TL_model.fit(x=X_train,y=Y_train,epochs=20,batch_size=64)

preds = TL_model.evaluate(x = X_test, y = Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

TL_model.summary()
plot_model(TL_model, to_file='TL_model.png')
SVG(model_to_dot(TL_model).create(prog='dot', format='svg'))


