import tensorflow.keras as keras

keras.backend.clear_session()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=0.2,
)

BATCH_SIZE = 32

training_data = image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory="dataset",
    shuffle=True,
    target_size=(64, 64),
    subset="training",
    class_mode="categorical",
)
validation_data = image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory="dataset",
    shuffle=True,
    target_size=(64, 64),
    subset="validation",
    class_mode="categorical",
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import BatchNormalization

num_classes = 40
epochs = 100
# Wymiary wejścia - bitmapa 64 x 64 x trzy kolory
input_shape = (64, 64, 3)
model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

from tensorflow.keras.optimizers import SGD

sgd = SGD(learning_rate=0.005, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(
    loss=keras.losses.categorical_crossentropy,
    # optimizer=keras.optimizers.Adadelta(),
    optimizer=sgd,
    metrics=["accuracy"],
)

from tensorflow.keras.callbacks import ModelCheckpoint
#checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True, mode='max')
callback_list = [checkpoint]

model.fit(
    training_data,
    steps_per_epoch=492,
    epochs=epochs,
    validation_data=validation_data,
    validation_steps=122,
    callbacks=callback_list
)

model.save("model")