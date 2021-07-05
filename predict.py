import os
from tensorflow.keras.models import load_model
import numpy as np
import src.batch_generator as datagenerator

def predict_fun(modelFile, dataFile):

    trained_model = load_model(modelFile)

    datagen = datagenerator.generate_from_file(dataFile, (224,224))

    costs = []

    # main function
    for single_batch in datagen:
        test_data = []
        test_data.append(np.array(single_batch))
        test_data = np.array(test_data)

        result = trained_model.predict(test_data)
        costs.append(np.linalg.norm(np.squeeze(result[0])-np.squeeze(test_data)))

    score_vid = costs - min(costs)
    score_vid = 1 - (score_vid / max(score_vid))

    np.save(os.path.join(os.getcwd(), 'score_vid.npy'), score_vid)
