"""
    Contains the architecture for creating a cropland data layer within SAR images.
"""
from tensorflow.python.framework.ops import disable_eager_execution
from keras.layers import Activation, BatchNormalization, Dropout, Input, Layer, TimeDistributed, LSTM, Flatten, Dense, ConvLSTM2D, Reshape, AveragePooling3D, Conv3D, Bidirectional
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from src.config import NETWORK_DEMS as dems


def conv2d_block_time_dist(
    input_tensor: Input,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    # first layer
    x = TimeDistributed(
        Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer='he_normal',
            padding='same'
        )
    )(input_tensor)
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same', return_sequences=True)(input_tensor)

    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    # second layer
    x = TimeDistributed(
        Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer='he_normal',
            padding='same'
        )
    )(input_tensor)

    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same', return_sequences=True)(input_tensor)
    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """


def create_cdl_model_masked(
    model_name: str,
    num_filters: int = 32,
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

    clstmForwards = ConvLSTM2D(num_filters * 4, kernel_size=3, padding='same', return_sequences=True)
    clstmBlock = Bidirectional(clstmForwards, merge_mode="sum")(p3)
    # Expanding dims
    u11 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(clstmBlock)

    u11 = concatenate([u11, c3])
    u11 = TimeDistributed(Dropout(dropout))(u11)
    c11 = conv2d_block_time_dist(u11, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    u12 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c11)
    u12 = concatenate([u12, c2])
    u12 = TimeDistributed(Dropout(dropout))(u12)
    c12 = conv2d_block_time_dist(u12, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    u13 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c12)
    u13 = concatenate([u13, c1])
    u13 = TimeDistributed(Dropout(dropout))(u13)
    c13 = conv2d_block_time_dist(
        u13, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    clstmForwards_2 = ConvLSTM2D(num_filters, kernel_size=3, padding='same', return_sequences=False)
    clstmBlock_2 = Bidirectional(clstmForwards_2, merge_mode="sum")(c13)
    final_layer = BatchNormalization()(clstmBlock_2)
    final_conv = Conv2D(1, 1, activation='sigmoid')(final_layer)
    
    model = Model(inputs=inputs, outputs=[final_conv])

    model.__asf_model_name = model_name

    # Adam(lr=1e-3)
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"]
    )

    return model
