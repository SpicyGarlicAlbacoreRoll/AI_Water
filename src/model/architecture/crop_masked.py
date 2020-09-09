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

def convlstm2d_block(
    input_tensor: Input,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    # first layer
    # x = TimeDistributed(
    #     Conv2D(
    #         filters=num_filters,
    #         kernel_size=(kernel_size, kernel_size),
    #         kernel_initializer='he_normal',
    #         padding='same'
    #     )
    # )(input_tensor)
    x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), activation = 'relu', padding='same', return_sequences=True)(input_tensor)

    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    # x = TimeDistributed(Activation('relu'))(x)
    # second layer
    # x = TimeDistributed(
    #     Conv2D(
    #         filters=num_filters,
    #         kernel_size=(kernel_size, kernel_size),
    #         kernel_initializer='he_normal',
    #         padding='same'
    #     )
    # )(input_tensor)

    x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), activation = 'relu', padding='same', return_sequences=True)(input_tensor)
    
    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    # x = TimeDistributed(Activation('relu'))(x)

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

    c4 = conv2d_block_time_dist(p3, num_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = TimeDistributed(MaxPooling2D((2, 2)))(c4)
    p4 = TimeDistributed(Dropout(dropout))(p4)

    c5 = conv2d_block_time_dist(p4, num_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p5 = TimeDistributed(MaxPooling2D((2, 2)))(c5)
    p5 = TimeDistributed(Dropout(dropout))(p5)

    c6 = conv2d_block_time_dist(
        p5, num_filters * 8, kernel_size=3, batchnorm=batchnorm
    )
    p6 = TimeDistributed(MaxPooling2D((2, 2)))(c6)
    p6 = TimeDistributed(Dropout(dropout))(p6)

    # c7 = conv2d_block_time_dist(
    #     p6, num_filters=num_filters * 16, kernel_size=3, batchnorm=batchnorm
    # )

    clstmForwards = ConvLSTM2D(num_filters * 8, kernel_size=3, padding='same', return_sequences=True)
    # clstmBackwards = ConvLSTM2D(num_filters * 16, kernel_size=3, padding='same', return_sequences=True, go_backwards=True)
    clstmBlock = Bidirectional(clstmForwards, merge_mode="sum")(p6)
    # Expanding to 64 x 64 x 1
    u8 = TimeDistributed(Conv2DTranspose(
        num_filters * 4, (3, 3), strides=(2, 2), padding='same'
    ))(clstmBlock)
    u8 = concatenate([u8, c6])
    u8 = TimeDistributed(Dropout(dropout))(u8)
    c8 = conv2d_block_time_dist(u8, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    # c8 = convlstm2d_block(u8, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    u9 = TimeDistributed(Conv2DTranspose(
        num_filters * 2, (3, 3), strides=(2, 2), padding='same'
    ))(c8)
    u9 = concatenate([u9, c5])
    u9 = TimeDistributed(Dropout(dropout))(u9)
    c9 = conv2d_block_time_dist(u9, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    # c9 = convlstm2d_block(u9, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    u10 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c9)

    u10 = concatenate([u10, c4])
    u10 = TimeDistributed(Dropout(dropout))(u10)
    c10 = conv2d_block_time_dist(u10, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # c10 = convlstm2d_block(u10, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    u11 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c10)

    u11 = concatenate([u11, c3])
    u11 = TimeDistributed(Dropout(dropout))(u11)
    c11 = conv2d_block_time_dist(u11, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # c11 = convlstm2d_block(u11, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    u12 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c11)
    u12 = concatenate([u12, c2])
    u12 = TimeDistributed(Dropout(dropout))(u12)
    c12 = conv2d_block_time_dist(u12, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # c12 = convlstm2d_block(u12, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    u13 = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(c12)
    u13 = concatenate([u13, c1])
    u13 = TimeDistributed(Dropout(dropout))(u13)
    c13 = conv2d_block_time_dist(
        u13, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    # c13 = convlstm2d_block(u13, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    #V1.1.2
    # outputs = TimeDistributed(Conv2D(2, 1, activation='softmax', name='last_layer'))(c13)
    clstmForwards_2 = ConvLSTM2D(num_filters, kernel_size=3, padding='same', return_sequences=False)
    # clstmBackwards_2 = ConvLSTM2D(num_filters * 16, kernel_size=3, padding='same', return_sequences=False, go_backwards=True)
    clstmBlock_2 = Bidirectional(clstmForwards_2, merge_mode="sum")(c13)
    final_conv = Conv2D(1, 1, activation='sigmoid')(clstmBlock_2)
    # output = Reshape((1, dems, dems, 1))(final_conv)
    #V1.1.5
    # lstm_layer_0 = ConvLSTM2D(2, 1, return_sequences=False, activation='softmax')(c13)
    # normalized = BatchNormalization()(lstm_layer_0)
    # lstm_layer_1 = ConvLSTM2D(2, 1, name='last_layer',  activation='softmax')(normalized)
    # outputs = BatchNormalization()(lstm_layer_1)
    # outputs_reshaped = Reshape((1, dems, dems, 2))(normalized)
    
    # output = Conv3D( filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(outputs)
    # normalized_output = BatchNormalization()(outputs)
    # # averaged = AveragePooling3D()
    # LSTM_to_conv_dims = (-1, dems, dems, 1)
    # reshaped = Reshape(LSTM_to_conv_dims)(normalized_output)
    # output = Conv2D(3, (3, 3), activation='relu', padding='same')(reshaped)
    # outputs = TimeDistributed(Flatten())(outputs)
    # lstm = LSTM(2)(outputs)
    # final = Dense(1, activation='sigmoid')(lstm)

    model = Model(inputs=inputs, outputs=[final_conv])

    model.__asf_model_name = model_name

    # Adam(lr=1e-3)
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"]
    )

    return model
