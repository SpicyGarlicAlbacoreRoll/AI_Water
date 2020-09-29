"""
    Contains the architecture for creating a cropland data layer within SAR images.
"""
from tensorflow.python.framework.ops import disable_eager_execution
from keras.layers import Activation, BatchNormalization, Dropout, Input, Layer, TimeDistributed, LSTM, Flatten, Dense, ConvLSTM2D, Reshape, AveragePooling3D, Conv3D, Bidirectional, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling3D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from src.config import NETWORK_DEMS as dems
from src.model.architecture.dice_loss import dice_coefficient, dice_coefficient_loss

def conv2d_block_time_dist(
    input_tensor: Input,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    # first layer
    x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)(input_tensor)

    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    # second layer

    x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)(x)
    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def deconv2d_block_time_dist(
    input_tensor: Input,
    concat_layer: Input,
    dropout: int,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True,
    
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    x = TimeDistributed(UpSampling2D(
        size=(2, 2)
    ))(input_tensor)

    x = concatenate([x, concat_layer])
    x = TimeDistributed(Dropout(dropout))(x)
    x = conv2d_block_time_dist(x, num_filters, kernel_size=3, batchnorm=batchnorm)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def create_cdl_model_masked(
    model_name: str,
    num_filters: int = 8,
    time_steps: int = 5,
    dropout: float = 0.5,
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

    # c4 = conv2d_block_time_dist(p3, num_filters * 8, kernel_size=3, batchnorm=batchnorm)
    # p4 =TimeDistributed( MaxPooling2D((2, 2)))(c4)
    # p4 = TimeDistributed(Dropout(dropout))(p4)

    # c5 = conv2d_block_time_dist(p4, num_filters * 16, kernel_size=3, batchnorm=batchnorm)
    # p5 =TimeDistributed( MaxPooling2D((2, 2)))(c5)
    # p5 = TimeDistributed(Dropout(dropout))(p5)

    # clstmForwards = ConvLSTM2D(num_filters * 16, kernel_size=3, padding='same', return_sequences=True)
    # clstmBlock = Bidirectional(clstmForwards, merge_mode="sum")(p5)
    # clstmNormalized = TimeDistributed(BatchNormalization())(clstmBlock)

    # Expanding dims
    # c6 = deconv2d_block_time_dist(p5, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c5)
    # c7 = deconv2d_block_time_dist(p4, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c4)
    c8 = deconv2d_block_time_dist(p3, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c3)
    c9 = deconv2d_block_time_dist(c8, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c2)
    c10 = deconv2d_block_time_dist(c9, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1)
    # u6 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(p5)

    # u6 = concatenate([u6, c5])
    # u6 = TimeDistributed(Dropout(dropout))(u6)
    # c6 = conv2d_block_time_dist(u6, num_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # u7 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(c6)

    # u7 = concatenate([u7, c4])
    # u7 = TimeDistributed(Dropout(dropout))(u7)
    # c7 = conv2d_block_time_dist(u7, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
   
    
    # u11 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(c7)

    # u11 = concatenate([u11, c3])
    # u11 = TimeDistributed(Dropout(dropout))(u11)
    # c11 = conv2d_block_time_dist(u11, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # u12 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(c11)
    # u12 = concatenate([u12, c2])
    # u12 = TimeDistributed(Dropout(dropout))(u12)
    # c12 = conv2d_block_time_dist(u12, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # u13 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(c12)
    # u13 = concatenate([u13, c1])
    # u13 = TimeDistributed(Dropout(dropout))(u13)
    # c13 = conv2d_block_time_dist(
    #     u13, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    # )
    # final_max_pool = GlobalAveragePooling3D()(c13)
    # reshaped = Reshape((64, 64, 1))(final_max_pool)
    # final_layer = Conv2D(1, 1, activation='sigmoid')(reshaped)
    # final_conv = TimeDistributed(Conv2D(1, 1))(c13)
    clstmForwards_2 = ConvLSTM2D(num_filters, kernel_size=3, padding='same', kernel_initializer = 'he_normal', return_sequences=False)(c10)
    final_conv_1 = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = 'he_normal')(clstmForwards_2)
    final_conv_2 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(final_conv_1)
    final_conv = Conv2D(1, 1, activation='sigmoid',)(final_conv_2)
    # clstmBlock_2 = Bidirectional(clstmForwards_2, merge_mode="sum")(c13)
    # final_layer = BatchNormalization()(clstmForwards_2)
    # # final_conv = Conv2D(1, 1, activation='sigmoid')(final_layer)
    # final_dense = Dense(1)(final_conv)
    model = Model(inputs=inputs, outputs=[final_conv])

    model.__asf_model_name = model_name

    # Adam(lr=1e-3)
    # dice_coefficient_loss
    model.compile(
        loss=dice_coefficient_loss, optimizer=Adam(), metrics=[MeanIoU(num_classes=2)]
    )

    return model
