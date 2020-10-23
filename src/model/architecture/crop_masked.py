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
from keras.optimizers.schedules import ExponentialDecay
from keras.metrics import MeanIoU
import tensorflow as tf
from src.config import NETWORK_DEMS as dems
from src.config import TIME_STEPS, CROP_CLASSES, N_CHANNELS
from src.model.architecture.dice_loss import dice_coefficient, dice_coefficient_loss, jaccard_distance_loss

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
    
    # first conv layer
    
    x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    # second conv layer

    x = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',))(x)
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
    x = conv2d_block(x, num_filters, kernel_size=3, batchnorm=batchnorm, activation=activation)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def create_cdl_model_masked(
    model_name: str,
    num_filters: int = 8,
    time_steps: int = TIME_STEPS,
    dropout: float = 0.5,
    batchnorm: bool = True
) -> Model:
    """ Function to define the Time Distributed UNET Model """

    """Requires stack of Sequential SAR data (with vh vv channels stacked), where each image is a different timestep"""
    inputs = Input(shape=(time_steps, dems, dems, N_CHANNELS), batch_size=None)
    c1 = conv2d_block(
        inputs, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )

    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)
    p3 = Dropout(dropout)(p3)

    c5 = conv2d_block(p3, num_filters * 8, kernel_size=3)

    # Expanding dims
    u1 = deconv2d_block_time_dist(c5, num_filters=num_filters * 8, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c3, activation=True)
    u2 = deconv2d_block_time_dist(u1, num_filters=num_filters * 4, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c2, activation=True)
    u3 = deconv2d_block_time_dist(u2, num_filters=num_filters * 4, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1, activation=True)
    
    # classifier (forward-backwards convlstm)
    final_conv = ConvLSTM2D(filters=1, kernel_size=1, activation="sigmoid", padding='same', return_sequences=False)
    forward_and_back = Bidirectional(final_conv)(u3)

    model = Model(inputs=inputs, outputs=[forward_and_back])

    model.__asf_model_name = model_name

    lr_schedule = ExponentialDecay(1e-3, decay_steps=100000, decay_rate=0.96, staircase=True)
    # Adam(lr=1e-3)
    # dice_coefficient_loss
    model.compile(
        loss=jaccard_distance_loss, optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy' ]
    )

    return model
