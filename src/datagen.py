
import glob
import numpy as np
import cv2

def avenue_datagenerator(image_size, setToTrueIfTraining):
    src = None
    if setToTrueIfTraining is True:
        src = "data/Avenue Dataset/training_videos/*"
    else:
        src = "data/Avenue Dataset/testing_videos/*"
        
    import os
    generated_prefix = ''
    cwd = os.getcwd()
    if setToTrueIfTraining:
        generated_prefix = cwd + '/generated'
        if not os.path.exists(generated_prefix):
            os.mkdir(generated_prefix)

        generated_prefix = cwd + '/generated/avenue'
        if not os.path.exists(generated_prefix):
            os.mkdir(generated_prefix)

        generated_prefix = cwd + '/generated/avenue/train'
        if not os.path.exists(generated_prefix):
            os.mkdir(generated_prefix)

    else:
        generated_prefix = cwd + '/generated'
        if not os.path.exists(generated_prefix):
            os.mkdir(generated_prefix)
        
        generated_prefix = cwd + '/generated/avenue'
        if not os.path.exists(generated_prefix):
            os.mkdir(generated_prefix)
        
        generated_prefix = cwd + '/generated/avenue/test'
        if not os.path.exists(generated_prefix):
            os.mkdir(generated_prefix)
    
    files = glob.glob(src)


    T = 10
    counter = 0
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

        frame_amount = len(video_frames)

        # generate stride comibnations            

        # strides of +1
        for idx in range(frame_amount-T):
            frame_batch = np.array(video_frames[idx:idx+T])
            if len(frame_batch) != T:
                break
            np.save(generated_prefix + '/stride_1_{}'.format(counter), frame_batch)
            counter += 1

        #strides of +2
        for idx in range(frame_amount//2):
            frame_batch = np.array(video_frames[idx:idx+2*T:2])
            if len(frame_batch) != T:
                break
            np.save(generated_prefix + '/stride_2_{0}'.format(counter), frame_batch)
            counter += 1


        #strides of +3
        for idx in range(frame_amount//3):
            frame_batch = np.array(video_frames[idx:idx+3*T:3])
            if len(frame_batch) != T:
                break
            np.save(generated_prefix + '/stride_3_{0}'.format(counter), frame_batch)
            counter += 1

