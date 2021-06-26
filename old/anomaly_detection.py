"""
https://github.com/keras-team/keras-io/blob/master/examples/vision/video_classification.py
https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
""" 

from numpy.linalg.linalg import norm
import tensorflow.keras as keras
import cv2
import numpy as np
from tensorflow.python.keras.backend import reshape
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
    cap = cv2.VideoCapture("data/Avenue Dataset/training_videos/{}".format(video)) 
    ret, frame = cap.read()
    while ret:
        image = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = image.reshape(227, 227, 1)
        videos_buffer.append(image)
        ret, frame = cap.read()
    cap.release()
    break
videos_buffer = np.clip(videos_buffer, 0, 1)

data = []
num = len(videos_buffer)
for idx in range(num-10):
    frame_batch = videos_buffer[idx:idx+10]
    if len(frame_batch) == 10:
        frame_batch = np.array(frame_batch)
        data.append(frame_batch)

for idx in range(num//2):
    frame_batch = videos_buffer[idx:2*idx:2]
    if len(frame_batch) == 10:
        frame_batch = np.array(frame_batch)
        data.append(frame_batch)

for idx in range(num//3):
    frame_batch = videos_buffer[idx:3*idx:3]
    if len(frame_batch) == 10:
        frame_batch = np.array(frame_batch)
        data.append(frame_batch)


data = np.array(data)

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
from tensorflow.keras.models import Model

inputs = Input(shape=(10, 227, 227, 1))
#        norm_prediction = model.predict_on_batch(X[i].reshape(1, 160, 240, 1))
# pierwsza warstwa

conv_11 = TimeDistributed(Conv2D(filters=128, kernel_size=(11,11), padding='same', strides=(4,4)))(inputs)
conv_11 = TimeDistributed(BatchNormalization())(conv_11)
conv_11 = TimeDistributed(Activation('relu'))(conv_11)

# druga warstwa
conv_5 = TimeDistributed(Conv2D(filters=64, kernel_size=(5,5), padding='same', strides=(2,2)))(conv_11)
conv_5 = TimeDistributed(BatchNormalization())(conv_5)
conv_5 = TimeDistributed(Activation('relu'))(conv_5)

autoencoder = ConvLSTM2D(filters=64, padding='same', return_sequences=True, kernel_size=(3,3))(conv_5)
autoencoder = ConvLSTM2D(filters=32, padding='same', return_sequences=True, kernel_size=(3,3))(autoencoder)
autoencoder = ConvLSTM2D(filters=64, padding='same', return_sequences=True, kernel_size=(3,3))(autoencoder)

deconv_5 = TimeDistributed(Conv2DTranspose(filters=128, kernel_size=(5,5), padding='same', strides=(2,2)))(autoencoder)
deconv_5 = TimeDistributed(BatchNormalization())(deconv_5)
deconv_5 = TimeDistributed(Activation('relu'))(deconv_5)

deconv_11 = TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(11,11), padding='same', strides=(4,4)))(deconv_5)
deconv_11 = TimeDistributed(BatchNormalization())(deconv_11)
deconv_11 = TimeDistributed(Activation('relu'))(deconv_11)

outputs = TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(227,227), padding='same', strides=(4,4)))(deconv_11)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='Adam',
    loss='mse',
    metrics=["accuracy"],
)

from tensorflow.keras.callbacks import ModelCheckpoint
#checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

"""
    ZMIENNE POMOCNICZE

"""
epochs = 10

"""
    TODO zmienić dane trenujące
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model.summary()

model.fit(
    data, data,
    batch_size=32,
    epochs=epochs,
    callbacks=callback_list
)

model.save("model")