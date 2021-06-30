import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from tensorflow.keras.models import load_model
import numpy as np

import src.batch_generator as datagenerator


batch_size = 16

trained_model = load_model("best_model.hdf5")

datagen = datagenerator.generate_from_dir(batch_size, 'generated/avenue/', False)

filesize = batch_size * 16

X_test = []
#X_test = HDF5Matrix(filepath, 'data')
for i in range(filesize):
    frame_set = next(datagen)
    X_test.append(frame_set[0])


# main function
result = trained_model.predict(X_test, batch_size)

costs = np.zeros(len(X_test))

for j in range(filesize):
    costs[j] = np.linalg.norm(np.squeeze(result[j])-np.squeeze(X_test[j]))

score_vid = costs - min(costs)
score_vid = 1 - (score_vid / max(score_vid))


np.save(os.path.join(os.getcwd(), 'score_vid.npy'), score_vid)