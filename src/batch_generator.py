
import numpy as np
import os
import random

def generate_from_dir(batch_size, dir, trueIfTrain):
    if trueIfTrain:
        dir = dir + 'train'
    else: 
        dir = dir + 'test'

    frame_batches = os.listdir(dir)
    random.shuffle(frame_batches)
    amount = len(frame_batches)
    max_idx = (amount // batch_size) * batch_size

    return_val = []

    counter = 0
    frame_batches = frame_batches[:max_idx]
    for batch_file in frame_batches:
        return_val.append(np.load(dir + '/' + batch_file))
        counter += 1
        if counter == 10:
            counter = 0
            converted = np.array(return_val)
            yield (converted, converted)
