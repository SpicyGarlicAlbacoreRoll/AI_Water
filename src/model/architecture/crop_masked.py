"""
    Contains the architecture for creating a cropland data layer within SAR images.
"""
from tensorflow.python.framework.ops import disable_eager_execution
from keras.layers import Activation, BatchNormalization, Dropout, Input, Layer, TimeDistributed, LSTM, Flatten, Dense, ConvLSTM2D, Reshape, AveragePooling3D, Conv3D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from src.config import NETWORK_DEMS as dems
from keras.losses import SparseCategoricalCrossentropy

def conv2d_block_time_dist(
    input_tensor: Input,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    # first layer
    x = ConvLSTM2D(
            filters=num_filters,
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            padding='same',
            activation='relu', 
            return_sequences=True,
    )(input_tensor)
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same', return_sequences=True)(input_tensor)

    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    # second layer
    x = ConvLSTM2D(
            filters=num_filters,
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            padding='same',
            activation='relu', 
            return_sequences=True,
    )(input_tensor)

    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same', return_sequences=True)(input_tensor)
    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    return x


""" Cropland Data Time Series version of U-net model used in masked.py """


def create_cdl_model_masked(
    model_name: str,
    num_filters: int = 16,
    time_steps: int = 5,
    dropout: float = 0.1,
    batchnorm: bool = True
) -> Model:
    """ Function to define the Time Distributed UNET Model """

    """Requires stack of Sequential SAR data (with vh vv channels stacked), where each image is a different timestep"""
    inputs = Input(shape=(None, dems, dems, 2), batch_size=None)
    c1 = conv2d_block_time_dist(
        inputs, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )

    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
    p1 = TimeDistributed(Dropout(dropout))(p1)

    c2 = conv2d_block_time_dist(p1, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
    p2 = TimeDistributed(Dropout(dropout))(p2)

    c3 = conv2d_block_time_dist(p2, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 =TimeDistributed( MaxPooling2D((2, 2)))(c3)
    p3 = TimeDistributed(Dropout(dropout))(p3)

    c7 = conv2d_block_time_dist(
        p3, num_filters=num_filters * 8, kernel_size=3, batchnorm=batchnorm
    )



    u11 = TimeDistributed(Conv2DTranspose(
        num_filters * 4, (3, 3), strides=(2, 2), padding='same'
    ))(c7)

    u11 = concatenate([u11, c3])
    u11 = TimeDistributed(Dropout(dropout))(u11)
    c11 = conv2d_block_time_dist(
        u11, num_filters * 2, kernel_size=3, batchnorm=batchnorm
    )

    u12 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c11)
    u12 = concatenate([u12, c2])
    u12 = TimeDistributed(Dropout(dropout))(u12)
    c12 = conv2d_block_time_dist(
        u12, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )

    u13 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c12)
    u13 = concatenate([u13, c1])
    u13 = TimeDistributed(Dropout(dropout))(u13)
    c13 = conv2d_block_time_dist(
        u13, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )

    #V1.1.2
    # outputs = TimeDistributed(Conv2D(2, 1, activation='softmax', name='last_layer'))(c13)

    #V1.1.5
    lstm_layer_0 = ConvLSTM2D(2, (1, 1), return_sequences=True)(c13)
    normalized = BatchNormalization()(lstm_layer_0)
    lstm_layer_1 = ConvLSTM2D(1, 1, name='last_layer', activation="sigmoid")(normalized)
    # outputs = BatchNormalization()(lstm_layer_1)
    outputs_reshaped = Reshape((1, dems, dems, 1))(lstm_layer_1)
    
    # output = Conv3D( filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(outputs)
    # normalized_output = BatchNormalization()(outputs)
    # # averaged = AveragePooling3D()
    # LSTM_to_conv_dims = (-1, dems, dems, 1)
    # reshaped = Reshape(LSTM_to_conv_dims)(normalized_output)
    # output = Conv2D(3, (3, 3), activation='relu', padding='same')(reshaped)
    # outputs = TimeDistributed(Flatten())(outputs)
    # lstm = LSTM(2)(outputs)
    # final = Dense(1, activation='sigmoid')(lstm)

    model = Model(inputs=inputs, outputs=[outputs_reshaped])

    model.__asf_model_name = model_name
    
    model.compile(
        loss='mean_squared_error', optimizer=Adam(), metrics=["accuracy"]
    )

    return model
