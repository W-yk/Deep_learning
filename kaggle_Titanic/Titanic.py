#建立数据集

import numpy as np
import pandas as pd

def load_data(path):
    data=pd.read_csv(path)

    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0) #把性别从字符串类型转换为0或1数值型数据  
    data = data.fillna(0) #缺失字段填0  


    X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()  
    #字段说明：性别，年龄，客舱等级，兄弟姐妹和配偶在船数量，父母孩子在船的数量，船票价格  
  
    # 建立标签数据集  
    Y= data['Survived'].as_matrix() 

    print(X.shape,Y.shape)
    return X,Y
X_train,Y_train=load_data("D:/Git/Deep_Learning/titanic/train.csv")

#构建模型
from keras.models import Model
from keras.layers import Dense, Dropout,Input
import keras

X_input= Input(shape=(6,))
X = Dense(16)(X_input)
X = Dense(16)(X)
X = Dense(1,activation='sigmoid')(X)

model = Model(inputs=X_input, outputs=X)
model.compile(optimizer=keras.optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])
checkpoint=keras.callbacks.ModelCheckpoint("D:/Git/Deep_Learning/titanic/weights/weights.hdf5", verbose=1, save_weights_only=False)
try:
    model.load_weights("D:/Git/Deep_Learning/titanic/weights/weights.hdf5")
    print("load weights succeed")
except OSError:
    print("no weights found")
    
    
model.fit(x=X_train, y=Y_train, batch_size=32, epochs=2000,validation_split=0.1,callbacks=[checkpoint,keras.callbacks.TensorBoard(log_dir='D://Git//Deep_Learning//titanic//log')] )

#建立测试集，并生成模型预测结果

data=pd.read_csv("D:/Git/Deep_Learning/titanic/test.csv")
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0) 
data = data.fillna(0)
X_test = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()

output=pd.read_csv("D:/Git/Deep_Learning/titanic/gender_submission.csv")
for i in range(X_test.shape[0]):
    Y_test=np.reshape(X_test[i],(1,6))
    label=model.predict(Y_test)
    if label >0.5 :
        label=1
    else:
        label=0
    output.set_value(i,'Survived',label)
    if i%100==0:
        print(X_test[i],label)
output.to_csv("D:/Git/Deep_Learning/titanic/gender_submission.csv")