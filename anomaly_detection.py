"""
https://github.com/keras-team/keras-io/blob/master/examples/vision/video_classification.py
https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
""" 

import tensorflow.keras as keras
import cv2
import numpy as np
keras.backend.clear_session()

avenue_dataset = []
subway_dataset = []
anomaly_dataset = []


#cap = cv2.VideoCapture(0)
 
import os

"""Preprocessing """
videos_buffer = []
basedir = "data/Avenue Dataset/training_videos"
videos = os.listdir(basedir)
newsize = (227, 227)
for video in videos:
    single_video = []
    cap = cv2.VideoCapture("data/Avenue Dataset/training_videos/{}".format(video)) 
    ret, frame = cap.read()
    while ret:
        image = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        image = np.resize(image, newsize)
        image = image * (255.0/image.max())
        single_video.append(image)
        ret, frame = cap.read()
    videos_buffer.append(single_video)
    cap.release()

"""
    counter = 0
    strideCounter = 0
    for x in videoBuffer:

        if(strideCounter == 10):
            buffer1stride = videoBuffer[counter-10, counter].copy()
            strideCounter == 0
            #stride1.append(videoBuffer[0], videoBuffer[1], videoBuffer[2], videoBuffer[3], videoBuffer[4],videoBuffer[5],videoBuffer[6],videoBuffer[7],videoBuffer[8],videoBuffer[9],videoBuffer[10])
        #if(counter == 20):
        counter=counter+1
        strideCounter= strideCounter+1
"""


from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.layers import BatchNormalization, TimeDistributed, Activation
from tensorflow.keras.models import Sequential

model = Sequential()

model.add(Input(shape=(10, 227 , 227, 1)))

# pierwsza warstwa
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(11,11), strides=(4,4))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# druga warstwa
model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))


model.add(ConvLSTM2D(filters=64, padding='same', return_sequences=True, kernel_size=(3,3)))
model.add(ConvLSTM2D(filters=32, padding='same', return_sequences=True, kernel_size=(3,3)))
model.add(ConvLSTM2D(filters=64, padding='same', return_sequences=True, kernel_size=(3,3)))

model.add(TimeDistributed(Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(2,2))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(11,11), strides=(4,4))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(227,227), strides=(4,4))))


model.summary()


model.compile(
    optimizer='Adam',
    loss='mse',
    metrics=["accuracy"],
)

from tensorflow.keras.callbacks import ModelCheckpoint
#checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True, mode='max')
callback_list = [checkpoint]


"""
    ZMIENNE POMOCNICZE

"""
epochs = 100


"""
    TODO zmienić dane trenujące
"""
model.fit(
    videos_buffer,
    steps_per_epoch=492,
    epochs=epochs,
    callbacks=callback_list
)

model.save("model")