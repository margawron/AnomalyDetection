from matplotlib.pyplot import imshow
import src.lstm_net as lstm_net
import src.datagen as datagenerator


batch_size = 1
epochs = 10

image_size = (224,224)

model = lstm_net.get_model(image_size)

datagen = datagenerator.avenue_datagenerator(batch_size, image_size, True)


from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]


model.summary()

from keras import backend as K

def perceptual_distance(y_true, y_pred):
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.metrics import Accuracy

model.compile(optimizer='adam', 
    loss=mean_squared_error, 
    metrics=[
        "accuracy",
        perceptual_distance
    ]
)


model.fit(
    datagen,
    epochs=epochs,
    callbacks=callback_list
)

model.save("model")
