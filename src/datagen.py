
import glob
from os import read
import numpy as np
import cv2

def avenue_datagenerator(batch_size, image_size, setToTrueIfTraining):
    src = None
    if setToTrueIfTraining is True:
        src = "data/Avenue Dataset/training_videos/*"
    else:
        src = "data/Avenue Dataset/testing_videos/*"
        
    files = glob.glob(src)
    T = 10
    for file in files:
        video = cv2.VideoCapture(file)
        video_frames = []
        ret, frame = video.read()
        while ret:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            resized = cv2.resize(grayscale, image_size, interpolation=cv2.INTER_AREA)
            image = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            reshaped = image.reshape(image_size + (1,))
            video_frames.append(np.array(reshaped))
            ret, frame = video.read()

        video.release()

        frame_batches = []
        frame_amount = len(video_frames)

        # generate stride comibnations

        # strides of +1
        for idx in range(frame_amount-T):
            frame_batch = np.array(video_frames[idx:idx+T])
            if len(frame_batch) != T:
                break
            frame_batches.append(frame_batch)

        #strides of +2
        for idx in range(frame_amount//2):
            frame_batch = np.array(video_frames[idx:idx+2*T:2])
            if len(frame_batch) != T:
                break
            frame_batches.append(frame_batch)

        #strides of +3
        for idx in range(frame_amount//3):
            frame_batch = np.array(video_frames[idx:idx+3*T:3])
            if len(frame_batch) == T:
                break
            frame_batches.append(frame_batch)

        #align to strides to have only full batches
        number_of_batches = len(frame_batches)
        amount_of_batches_to_drop = number_of_batches % batch_size
        frame_batches = np.array(frame_batches[:number_of_batches-amount_of_batches_to_drop])

        import random
        random.shuffle(frame_batches)

        counter = 0
        return_array = []
        while counter + batch_size <= len(frame_batches)-1:
            for i in range(batch_size):
                return_array.append(frame_batches[counter])
                counter += 1
            array = np.array(return_array)
            yield (array, array)
