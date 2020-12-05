# -*- coding: utf-8 -*-

import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt 
from keras.models import Sequential ,Model
from keras.optimizers import Adam
from keras.layers.core import Dense , Flatten
from keras.layers.convolutional import Conv2D , MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.applications import MobileNet

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
data_train = train_datagen.flow_from_directory(
"DATOS/train",
    target_size=(128, 128),
    batch_size=128,
    class_mode='binary')

test_datagen = ImageDataGenerator(
    rescale=1./255)
data_test = test_datagen.flow_from_directory(  
    "DATOS/test",
    target_size=(128, 128),
    class_mode='binary')

model = MobileNet(input_shape=(128, 128, 3),include_top=False,pooling='avg')
x=model.output
x=Dense(512,activation="relu")(x)
predict=Dense(1,activation="sigmoid")(x)
model=Model(inputs=model.input,outputs=predict)

adam=Adam(lr=0.0001)
model.compile(optimizer=adam,loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

history=model.fit(data_train,epochs=10,validation_data=data_test,verbose=1)

model.save("/Clasificador.h5")

def visualizar_img_test(path,modelo):
  img=cv2.imread(path)
  img_1=cv2.resize(img,(128,128))
  img=(img_1/255)
  img=np.expand_dims(img,axis=0)
  predict=modelo.predict(img)
  if predict[0][0]<0.7:
    plt.title(str(100-predict[0][0]*100)+"% MASCARILLA")
    plt.imshow(img_1.astype("uint8"))
    plt.show()
  else:
    plt.title(str(predict[0][0]*100)+"% NO TIENE MASCARILLA")
    plt.imshow(img_1.astype("uint8"))
    plt.show()

visualizar_img_test("10.jpg",model)
visualizar_img_test("20200516_211423.jpg",model)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train","test"])
plt.xlabel("EPOCA")
plt.ylabel("PORCENTAJE DE ERROR")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["train","test"])
plt.xlabel("EPOCA")
plt.ylabel("PORCENTAJE DE PREDICCION")

plt.show()
