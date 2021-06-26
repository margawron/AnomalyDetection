
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.layers import BatchNormalization, TimeDistributed, Activation
from tensorflow.keras.models import Model


def get_model(input_shape):
    inputs = Input(shape=(10,) + input_shape + (1,))
    # pierwsza warstwa

    conv_11 = TimeDistributed(Conv2D(filters=128, kernel_size=(11,11), padding='same', strides=(4,4)))(inputs)
    conv_11 = TimeDistributed(BatchNormalization())(conv_11)
    conv_11 = TimeDistributed(Activation('relu'))(conv_11)

    # druga warstwa
    conv_5 = TimeDistributed(Conv2D(filters=64, kernel_size=(5,5), padding='same', strides=(2,2)))(conv_11)
    conv_5 = TimeDistributed(BatchNormalization())(conv_5)
    conv_5 = TimeDistributed(Activation('relu'))(conv_5)

    autoencoder = ConvLSTM2D(filters=64, padding='same', return_sequences=True, kernel_size=(3,3))(conv_5)
    autoencoder = ConvLSTM2D(filters=32, padding='same', return_sequences=True, kernel_size=(3,3))(autoencoder)
    autoencoder = ConvLSTM2D(filters=64, padding='same', return_sequences=True, kernel_size=(3,3))(autoencoder)

    deconv_5 = TimeDistributed(Conv2DTranspose(filters=128, kernel_size=(5,5), padding='same', strides=(2,2)))(autoencoder)
    deconv_5 = TimeDistributed(BatchNormalization())(deconv_5)
    deconv_5 = TimeDistributed(Activation('relu'))(deconv_5)

    deconv_11 = TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(11,11), padding='same', strides=(4,4)))(deconv_5)
    deconv_11 = TimeDistributed(BatchNormalization())(deconv_11)
    outputs = TimeDistributed(Activation('relu'))(deconv_11)

    model = Model(inputs=inputs, outputs=outputs)
    return model