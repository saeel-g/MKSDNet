
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, activations
from tensorflow.keras.applications import ResNet152, VGG19, Xception, InceptionResNetV2, EfficientNetB7, DenseNet121
import pandas as pd
from tqdm import tqdm
import os
from tensorboard.plugins.hparams import api as hp
import glob
import matplotlib.pyplot as plt
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from DenseNet import DenseNet, DenseNet2





SIZE=256


train = pd.read_csv('../dataset/RFMiD2_0/RFMiD_2_Training_labels.csv')    # reading the csv file
df_train = pd.DataFrame(train)
train_IDs=df_train['id'].values

print(train_IDs)
train_image = []
for i in tqdm(train_IDs):
    img = image.load_img('../dataset/RFMiD2_0/Training_set/'+str(i)+'.jpg',target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X_train = np.array(train_image)

val= pd.read_csv('../dataset/RFMiD2_0/RFMiD_2_Validation_labels.csv')    # reading the csv file
df_val = pd.DataFrame(val)
val_IDs=df_val['id'].values
val_image = []
for i  in tqdm(val_IDs):
    img = image.load_img('../dataset/RFMiD2_0/Validation_set/'+str(i)+'.jpg',target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
    img = img/255
    val_image.append(img)
X_val = np.array(val_image)

y_train = np.array(train.drop(['id'],axis=1))
y_val = np.array(val.drop(['id'],axis=1))






from tensorflow.keras.applications import ResNet152, VGG19, Xception, InceptionResNetV2, DenseNet121
# MN=[ResNet152, VGG19, Xception, InceptionResNetV2]
# results=[]
# for model_fun in MN:    
#     model_name = model_fun.__name__
#     print(f'*******************{model_name}************************')
#     model=model_fun(input_shape=(SIZE,SIZE,3), weights=None, include_top=True, classes=2)
#     # model= DenseNet(blocks=[6, 12, 24, 16],input_shape=(SIZE,SIZE,3))
#     # model.summary()


    

#     model.compile(optimizer=Adam(learning_rate=0.00001),
#                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                     metrics=['accuracy'])

#     filepath = f'./models/RFMiD2/{model_name}_BCE_bestmodel.hdf5'

#     checkpoint = ModelCheckpoint(filepath=filepath, 
#                                 monitor='val_accuracy',
#                                 verbose=1, 
#                                 save_best_only=True,
#                                 mode='max')

    
#     callbacks = [checkpoint]

#     history = model.fit(X_train, y_train,
#                     batch_size=32,
#                     epochs=50,
#                     validation_data=(X_val, y_val),
#                     callbacks=callbacks)
    
#     print(f'*******************{model_name}_KLloss************************')
#     model1=model_fun(input_shape=(SIZE,SIZE,3), weights=None, include_top=True, classes=2)
#     # model= DenseNet(blocks=[6, 12, 24, 16],input_shape=(SIZE,SIZE,3))
#     # model.summary()


    

#     model1.compile(optimizer=Adam(learning_rate=0.00001),
#                     loss=tf.keras.losses.KLDivergence(),
#                     metrics=['accuracy'])

#     filepath = f'./models/RFMiD2/{model_name}_KLloss_bestmodel.hdf5'

#     checkpoint = ModelCheckpoint(filepath=filepath, 
#                                 monitor='val_accuracy',
#                                 verbose=1, 
#                                 save_best_only=True,
#                                 mode='max')

    
#     callbacks = [checkpoint]

#     history1 = model1.fit(X_train, y_train,
#                     batch_size=32,
#                     epochs=50,
#                     validation_data=(X_val, y_val),
#                     callbacks=callbacks)

# ############################### Densenet ###################################
# model2= DenseNet(blocks=[6, 12, 24, 16],input_shape=(SIZE,SIZE,3))
# # model.summary()

# model2.compile(optimizer=Adam(learning_rate=0.00001),
#                     loss=tf.keras.losses.KLDivergence(),
#                     metrics=['accuracy'])

# filepath = f'./models/RFMiD2/Densenet_KLloss_bestmodel.hdf5'

# checkpoint = ModelCheckpoint(filepath=filepath, 
#                             monitor='val_accuracy',
#                             verbose=1, 
#                             save_best_only=True,
#                             mode='max')


# callbacks = [checkpoint]

# history2 = model2.fit(X_train, y_train,
#                 batch_size=32,
#                 epochs=50,
#                 validation_data=(X_val, y_val),
#                 callbacks=callbacks)



# model2.compile(optimizer=Adam(learning_rate=0.00001),
#                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                     metrics=['accuracy'])

# filepath = f'./models/RFMiD2/Densenet_BCE_bestmodel.hdf5'

# checkpoint = ModelCheckpoint(filepath=filepath, 
#                             monitor='val_accuracy',
#                             verbose=1, 
#                             save_best_only=True,
#                             mode='max')


# callbacks = [checkpoint]

# history2 = model2.fit(X_train, y_train,
#                 batch_size=32,
#                 epochs=50,
#                 validation_data=(X_val, y_val),
#                 callbacks=callbacks)


############################### Densenet2 ###################################
model3= DenseNet2(blocks=[6, 12, 24, 16],input_shape=(SIZE,SIZE,3))
# model.summary()

model3.compile(optimizer=Adam(learning_rate=0.00001),
                    loss=tf.keras.losses.KLDivergence(),
                    metrics=['accuracy'])

filepath = f'./models/RFMiD2/Densenet2_KLloss_bestmodel_2.hdf5'

checkpoint = ModelCheckpoint(filepath=filepath, 
                            monitor='val_accuracy',
                            verbose=1, 
                            save_best_only=True,
                            mode='max')


callbacks = [checkpoint]

history3 = model3.fit(X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=callbacks)



model3.compile(optimizer=Adam(learning_rate=0.00001),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

filepath = f'./models/RFMiD2/Densenet2_BCE_bestmodel_2.hdf5'

checkpoint = ModelCheckpoint(filepath=filepath, 
                            monitor='val_accuracy',
                            verbose=1, 
                            save_best_only=True,
                            mode='max')


callbacks = [checkpoint]

history3 = model3.fit(X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=callbacks)
