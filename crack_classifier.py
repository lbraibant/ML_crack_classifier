from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

#main_dir = "/home/lorraine/PycharmProjects/ML_crack_detection/"


def build_data_set(self,data_set='data1'):
    """
    Create "paths" and "labels" vectors that contains relative paths to images
    from the "data" repository and the corresponding labels=["uncracked","cracked"]
    :param data_set: 'data1' (Ozgenel data set), 'data2' (SDNET2018) or 'both'
    """
    data_paths = []
    data_labels = []
    if data_set in ['data1','both']:
        paths,labels = self.__load_data1()
        data_paths += paths
        data_labels += labels
    if data_set in ['data2','both']:
            paths,labels = self.__load_data2()
            data_paths += paths
            data_labels += labels
    return paths, labels

def __load_data1(self):
    """
    Provide the absolute paths and labels of the data set from Ozgenel (2018)
    :return: paths and labels (0=uncracked, 1=cracked)
    """
    data_dir = os.path.abspath("data")
    temp_dir = os.path.join("Concrete Crack Images for Classification","U")
    data_paths = [os.path.join(temp_dir,filename)
                  for filename in os.listdir(os.path.join(data_dir,temp_dir))]
    ndata = len(data_paths)
    data_labels = ["uncracked"]*ndata
    temp_dir = os.path.join("Concrete Crack Images for Classification","C")
    data_paths += [os.path.join(temp_dir,filename)
                   for filename in os.listdir(os.path.join(data_dir,temp_dir))]
    data_labels += ["cracked"]*(len(data_paths)-ndata)
    ndata = len(data_paths)
    print("Data set 1 contains %i images"%ndata)
    return data_paths, data_labels

def __load_data2(self):
    """
    Provide the absolute paths and labels of the data set SDNET2018
    :return: paths and labels (0=uncracked, 1=cracked)
    """
    data_dir = os.path.abspath("data")
    data_sub_dir = "SDNET2018"
    data_paths = []
    data_labels = []
    ndata = 0
    for sub_dir in ["D","P","W"]:
        for iscrack in ["U","C"]:
            temp_dir = os.path.join(data_sub_dir,"%s/%s%s"%(sub_dir,iscrack,sub_dir))
            data_paths += [os.path.join(temp_dir,filename)
                           for filename in os.listdir(os.path.join(data_dir,temp_dir))]
            if iscrack=="U":
                data_labels += ["uncracked"]*(len(data_paths)-ndata)
            else:
                data_labels += ["cracked"]*(len(data_paths)-ndata)
            ndata = len(data_paths)
    print("Data set 2 contains %i images"%ndata)
    return data_paths, data_labels







batch_size=20
ftrain = 0.8
fvalid = 0
ftest = 0.2
datagen = crackDataSet(data_set='data1',target_size=(227,227),color_mode="rgb")
datagen.split_dataset(ftrain,fvalid,ftest)
datagen.save_data_set('data1_proportional_seed10')
train_generator = datagen.get_training_set(batch_size=batch_size)
#print(train_generator
valid_generator = datagen.get_test_set(batch_size=batch_size)

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet',
                   input_shape=(227,227,3)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(datagen.classes), activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint
filepath = "log/weihgts-{epoch:02d}-{val_acc:2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc')
callbacks_list = [checkpoint]
nsteps = int(ftrain*(datagen.ndata)/batch_size)+1

model.fit_generator(train_generator, steps_per_epoch=nsteps, epochs=5,
                    validation_data=valid_generator, validation_freq=1,
                    callbacks=callbacks_list)
