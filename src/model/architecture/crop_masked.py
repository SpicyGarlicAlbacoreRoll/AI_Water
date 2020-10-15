"""
    Contains the architecture for creating a cropland data layer within SAR images.
"""
from tensorflow.python.framework.ops import disable_eager_execution
from keras.layers import Activation, BatchNormalization, Dropout, Input, Layer, TimeDistributed, LSTM, Flatten, Dense, ConvLSTM2D, Reshape, AveragePooling3D, Conv3D, Bidirectional, UpSampling2D, UpSampling3D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.recurrent import 
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, GlobalAveragePooling3D
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.metrics import MeanIoU
import tensorflow as tf
from src.config import NETWORK_DEMS as dems
from src.config import TIME_STEPS, CROP_CLASSES, N_CHANNELS
from src.model.architecture.dice_loss import dice_coefficient, dice_coefficient_loss, jaccard_distance_loss

def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.8, 0.2]
    return -tf.reduce_sum(target * weights * tf.math.log(output))

def conv2d_block_time_dist(
    input_tensor: Input,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True,
    
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    # first layer
    convLSTM_layer = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)
    x = Bidirectional(convLSTM_layer)(input_tensor)
    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    # second layer

    x = Bidirectional(convLSTM_layer)(x)
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)(x)
    if batchnorm:
        x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def conv2d_block(
    input_tensor: Input,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True,
    depth: int = 3,
    activation: bool = True,
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    # first layer
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)(input_tensor)
    x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(input_tensor)
    # x = Conv3D(filters=num_filters, kernel_size=(depth, kernel_size, kernel_size), padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)
    # second layer

    # x = Conv3D(filters=num_filters, kernel_size=(depth, kernel_size, kernel_size), padding='same')(x)
    x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(x)
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def deconv2d_block_time_dist(
    input_tensor: Input,
    concat_layer: Input,
    dropout: int,
    num_filters: int,
    kernel_size: int = 3,
    batchnorm: bool = True,
    return_last_sequence: bool = True,
    activation=True,
    
) -> Layer:
    """ Function to add 2 convolutional layers with the parameters
    passed to it """
    x = TimeDistributed(Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    ))(input_tensor)
    # x = TimeDistributed(UpSampling2D(
    #     size=(2, 2)
    # ))(input_tensor)
    # x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(input_tensor)
    x = concatenate([x, concat_layer], axis=-1)
    # x = Dropout(dropout)(x)
    # x = Conv3D(filters=num_filters, kernel_size=(3, kernel_size, kernel_size), padding='same')(x)
    # x = Conv3D(filters=num_filters, kernel_size=(3, kernel_size, kernel_size), padding='same')(x)
    # x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(input_tensor)
    # x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(input_tensor)
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=True)(x)
    # x = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', return_sequences=return_last_sequence)(x)
    x = conv2d_block(x, num_filters, kernel_size=3, batchnorm=batchnorm, activation=activation)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def create_cdl_model_masked(
    model_name: str,
    num_filters: int = 8,
    time_steps: int = TIME_STEPS,
    dropout: float = 0.7,
    batchnorm: bool = True
) -> Model:
    """ Function to define the Time Distributed UNET Model """

    """Requires stack of Sequential SAR data (with vh vv channels stacked), where each image is a different timestep"""
    inputs = Input(shape=(time_steps, dems, dems, N_CHANNELS), batch_size=None)
    c1 = conv2d_block(
        inputs, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )

    # p1 = MaxPooling3D((2, 2, 2))(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
    # p2 = MaxPooling3D((2, 2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)
    # p3 = MaxPooling3D((1, 2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    # c4 = conv2d_block(p3, num_filters * 8, kernel_size=3, depth=1, batchnorm=batchnorm)
    # c4 = conv2d_block(p3, num_filters * 8, kernel_size=3, batchnorm=batchnorm)
    # p4 = TimeDistributed(MaxPooling2D((2, 2)))(c4)
    # # p3 = MaxPooling3D((1, 2, 2))(c3)
    # p4 = Dropout(dropout)(p4)

    # c5 = conv2d_block_time_dist(p3, num_filters * 16, kernel_size=3)
    c5 = conv2d_block(p3, num_filters * 8, kernel_size=3)
    # c4 = conv2d_block_time_dist(p3, num_filters * 8, kernel_size=3, batchnorm=batchnorm)
    # p4 =TimeDistributed( MaxPooling2D((2, 2)))(c4)
    # p4 = Dropout(dropout)(p4)

    # c5 = conv2d_block_time_dist(p4, num_filters * 16, kernel_size=3, batchnorm=batchnorm)
    # p5 =TimeDistributed( MaxPooling2D((2, 2)))(c5)
    # p5 = Dropout(dropout)(p5)

    # clstmForwards = ConvLSTM2D(num_filters * 16, kernel_size=3, padding='same', return_sequences=True)
    # clstmBlock = Bidirectional(clstmForwards, merge_mode="sum")(p5)
    # clstmNormalized = TimeDistributed(BatchNormalization())(clstmBlock)

    # Expanding dims
    u1 = deconv2d_block_time_dist(c5, num_filters=num_filters * 8, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c3, activation=True)
    u2 = deconv2d_block_time_dist(u1, num_filters=num_filters * 4, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c2, activation=True)
    u3 = deconv2d_block_time_dist(u2, num_filters=num_filters * 1, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1, activation=True)
    # u4 = deconv2d_block_time_dist(u3, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1, activation=True)
    # c9 = deconv2d_block_time_dist(c8, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c2)
    # c10 = deconv2d_block_time_dist(c9, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1, return_last_sequence=False)
    
    
    
    
    # u6 = UpSampling3D(
    #     size=(1, 2, 2)
    # )(c4)

    # u6 = concatenate([u6, c3])
    # u6 = Dropout(dropout)(u6)
    # c6 = conv2d_block(u6, num_filters * 8, kernel_size=3, depth=1, activation=False, batchnorm=batchnorm)

    # u7 = UpSampling3D(
    #     size=(2, 2, 2)
    # )(c6)

    # u7 = concatenate([u7, c2])
    # u7 = Dropout(dropout)(u7)
    # c7 = conv2d_block(u7, num_filters * 4, kernel_size=3, depth=1, activation=False, batchnorm=batchnorm)


    # u8 = UpSampling3D(
    #     size=(2, 2, 2)
    # )(c7)

    # u8 = concatenate([u8, c1])
    # u8 = Dropout(dropout)(u8)
    # c8 = conv2d_block(u8, num_filters * 2, kernel_size=3, depth=1, activation=False, batchnorm=batchnorm)  
    




    # final_upsample = TimeDistributed(UpSampling2D(
    #     size=(2, 2)
    # ))(c8)
    # fdropout = Dropout(dropout)(final_upsample)
    # fconv = conv2d_block(fdropout, num_filters * 2, kernel_size=3, batchnorm=batchnorm)  
    # u11 = concatenate([u11, c3])
    # u11 = Dropout(dropout)(u11)
    # c11 = conv2d_block_time_dist(u11, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # u12 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(c11)
    # u12 = concatenate([u12, c2])
    # u12 = Dropout(dropout)(u12)
    # c12 = conv2d_block_time_dist(u12, num_filters * 1, kernel_size=3, batchnorm=batchnorm)
    # u13 = TimeDistributed(Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # ))(c12)
    # u13 = concatenate([u13, c1])
    # u13 = Dropout(dropout)(u13)
    # c13 = conv2d_block_time_dist(
    #     u13, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    # )
    # final_max_pool = GlobalAveragePooling3D()(c13)
    # reshaped = Reshape((64, 64, 1))(final_max_pool)
    # final_layer = Conv2D(1, 1, activation='sigmoid')(reshaped)
    # final_conv = TimeDistributed(Conv2D(1, 1))(c13)
    # clstmForwards_2 = ConvLSTM2D(num_filters, kernel_size=3, padding='same', kernel_initializer = 'he_normal', return_sequences=False, name="clstmForwards_2")(c10)
    # final_conv_1 = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = 'he_normal')(clstmForwards_2)
    # final_conv_2 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(final_conv_1)
    # final_max_pool = MaxPooling3D((4, 1, 1))(c8)
    # final_layer = Reshape((1, 64, 64, 1))(final_max_pool)

    # flattened = TimeDistributed(Flatten())(c4)
    # final_lstm = LSTM(2*64*64*1)(flattened)
    # final_conv = Conv3D(1, 1, padding='same', activation="sigmoid")(c8)
    # final_conv = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(c8)
    final_conv = ConvLSTM2D(filters=1, kernel_size=1, activation="sigmoid", padding='same', return_sequences=False)(u3)
    # output_shape=28*128*4
    # final_layer = Reshape((-1))(final_conv)
    # clstmBlock_2 = Bidirectional(clstmForwards_2, merge_mode="sum")(c13)
    # final_layer = BatchNormalization()(clstmForwards_2)
    # # final_conv = Conv2D(1, 1, activation='sigmoid')(final_layer)
    # final_dense = Dense(1)(final_conv)
    model = Model(inputs=inputs, outputs=[final_conv])

    model.__asf_model_name = model_name

    # Adam(lr=1e-3)
    # dice_coefficient_loss
    model.compile(
        loss='mean_squared_error', optimizer=Adam(lr=1e-3), metrics=['accuracy' ]
    )

    return model
    
