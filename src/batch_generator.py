
import numpy as np
import cv2
import os
import random

def generate_from_dir(batch_size, dir, trueIfTrain):
    if trueIfTrain:
        dir = dir + '/train'
    else: 
        dir = dir + '/test'

    frame_batches = os.listdir(dir)
    random.shuffle(frame_batches)
    amount = len(frame_batches)
    max_idx = (amount // batch_size) * batch_size

    return_val = []
    counter = 0

    frame_batches = frame_batches[:max_idx]
    for batch_file in frame_batches:
        frame_set = np.load(dir + '/' + batch_file)
        return_val.append(frame_set)
        counter += 1
        if counter == batch_size:
            counter = 0
            converted = np.array(return_val)
            return_val = []
            yield (converted, converted)

def generate_from_file(video_file, image_size):
    video = cv2.VideoCapture(video_file)
    video_frames = []
    ret, frame = video.read()
    while ret:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        resized = cv2.resize(grayscale, image_size, interpolation=cv2.INTER_AREA)
        image = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        reshaped = image.reshape(image_size + (1,))
        video_frames.append(np.array(reshaped))
        if len(video_frames) == 10:
            yield video_frames
            video_frames = video_frames[1:]
        ret, frame = video.read()

    video.release()
