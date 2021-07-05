import src.lstm_net as lstm_net
import src.batch_generator as datagenerator
import numpy as np
import os


batch_size = 16
epochs = 25

image_size = (224,224)

model = lstm_net.get_model(image_size)

datagen = datagenerator.generate_from_dir(batch_size, 'generated/avenue/')

from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]
# https://github.com/drsagitn/anomaly-detection-and-localization/

model.summary()

from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import optimizers

sgd = optimizers.SGD(nesterov=True)

model.compile(
    loss=mean_squared_error,
    optimizer=sgd
)

history = model.fit(
    datagen,
    epochs=epochs,
    steps_per_epoch=64,
    callbacks=callback_list
)

np.save(os.path.join(os.getcwd(), 'train_profile.npy'), history.history)

model.save("trained_model")
