import os


from tensorflow.keras.models import load_model
import numpy as np

import src.batch_generator as datagenerator


test_data_amount = 2

trained_model = load_model("trained_model")

datagen = datagenerator.generate_from_dir(test_data_amount, 'generated/avenue/', False)

X_test = []
frame_set = next(datagen)
X_test = [*frame_set[0][:]]

X_test = np.array(X_test)
X_test.reshape((1,-1))

# main function
result = trained_model.predict(X_test, test_data_amount)

costs = np.zeros(test_data_amount)

for j in range(test_data_amount):
    costs[j] = np.linalg.norm(np.squeeze(result[j])-np.squeeze(X_test[j]))

score_vid = costs - min(costs)
score_vid = 1 - (score_vid / max(score_vid))


np.save(os.path.join(os.getcwd(), 'score_vid.npy'), score_vid)