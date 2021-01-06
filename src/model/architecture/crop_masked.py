"""
    Contains the model architecture for predicting a cropland data layer within a time series of SAR images.
"""
from keras.layers import (
    Activation, BatchNormalization, Bidirectional, ConvLSTM2D, Dropout, Input,
    Layer, Reshape, TimeDistributed, UpSampling2D
)
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, LSTM
from keras.metrics import MeanIoU
from keras.losses import (
    BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
)
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.framework.ops import disable_eager_execution

from src.config import CROP_CLASSES, N_CHANNELS, NUM_FILTERS
from src.config import NETWORK_DEMS as dems
from src.config import TIME_STEPS
from src.model.architecture.dice_loss import jaccard_distance_loss, dice_coefficient_loss, cosh_dice_coefficient_loss

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
    
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    # second conv layer

    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',)(x)
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
    # x = Conv2DTranspose(
    #     num_filters * 1, (3, 3), strides=(2, 2), padding='same'
    # )(input_tensor)
    x = UpSampling2D(
        size=(2, 2)
    )(input_tensor)
    # x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same',)(input_tensor)
    x = concatenate([x, concat_layer], axis=-1)
    x = conv2d_block(x, num_filters, kernel_size=3, batchnorm=batchnorm, activation=activation)

    return x

""" Cropland Data Time Series version of U-net model used in masked.py """

def create_cdl_model_masked(
    model_name: str,
    num_filters: int = NUM_FILTERS,
    time_steps: int = TIME_STEPS,
    dropout: float = 0.5,
    batchnorm: bool = True
) -> Model:
    """ Function to define the Time Distributed UNET Model """

    """Requires stack of Sequential SAR data (with vh vv channels stacked), where each image is a different timestep"""
    inputs = Input(shape=(None, None, TIME_STEPS*N_CHANNELS), batch_size=None)
    c1 = conv2d_block(
        inputs, num_filters * 1, kernel_size=3, batchnorm=batchnorm
    )

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)


    c4 = conv2d_block(p3, num_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, num_filters * 16, kernel_size=3, batchnorm=batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)

    c6 = conv2d_block(p5, num_filters * 32, kernel_size=3, batchnorm=batchnorm)
    p6 = MaxPooling2D((2, 2))(c6)
    p6 = Dropout(dropout)(p6)

    # c7 = conv2d_block(p6, num_filters * 64, kernel_size=3, batchnorm=batchnorm)
    # p7 = MaxPooling2D((2, 2))(c7)
    # p7 = Dropout(dropout)(p7)

    # c8 = conv2d_block(p7, num_filters * 128, kernel_size=3, batchnorm=batchnorm)
    # p8 = MaxPooling2D((2, 2))(c8)
    # p8 = Dropout(dropout)(p8)        
    # middle_clstm = ConvLSTM2D(filters=num_filters * 4, kernel_size=3, activation="tanh", padding='same', return_sequences=True)
    # middle_bidirection = Bidirectional(middle_clstm)(p3)
    middle = conv2d_block(p6, num_filters * 32, kernel_size=3)

        # Expanding dims
    # uv = deconv2d_block_time_dist(middle, num_filters=num_filters*128, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c8, activation=True)
    # uw = deconv2d_block_time_dist(uv, num_filters=num_filters*64, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c7, activation=True)
    uy = deconv2d_block_time_dist(middle, num_filters=num_filters*32, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c6, activation=True)
    uz = deconv2d_block_time_dist(uy, num_filters=num_filters*16, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c5, activation=True)
    u = deconv2d_block_time_dist(uz, num_filters=num_filters*8, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c4, activation=True)
    u1 = deconv2d_block_time_dist(u, num_filters=num_filters*4, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c3, activation=True)
    u2 = deconv2d_block_time_dist(u1, num_filters=num_filters*2, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c2, activation=True)
    u3 = deconv2d_block_time_dist(u2, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1, activation=True)
    
    # classifier (forward-backwards convlstm)
    # final_conv_forward = ConvLSTM2D(filters=num_filters, kernel_size=3, activation="tanh", padding='same', return_sequences=False)(u3)
    # final_conv_backwards = ConvLSTM2D(filters=num_filters, kernel_size=3, activation="tanh", padding='same', return_sequences=False)
    # final_bidirectional = Bidirectional(final_conv_forward)(u3)

    final = Conv2D(filters=1, kernel_size=1,  activation="sigmoid", padding='same')(u3)
    # final = ConvLSTM2D(filters=1, kernel_size=1, activation="sigmoid", padding='same', return_sequences=False)(final_bidirecitonal)
    # final_conv_locality = feature_locality(inputs, final, num_filters, batchnorm, dropout)
    model = Model(inputs=inputs, outputs=[final])

    model.__asf_model_name = model_name

    lr_schedule = ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.96, staircase=True)
    # Adam(lr=1e-3)
    # dice_coefficient_loss
    #[BinaryCrossentropy(from_logits=False), cosh_dice_coefficient_loss]
    model.compile(
        loss=jaccard_distance_loss, optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy', MeanIoU(num_classes=2) ]
    )

    return model

# get first, middle, and last layer, and compare with the output of the first model
def feature_locality(original_input, raw_ouput, num_filters, batchnorm, dropout):
    # inputs = Input(shape=(None, None, N_CHANNELS*time_steps), batch_size=None)
    beg = Lambda(lambda x: x [:, :, :2])(original_input)
    # mid = Lambda(lambda x: x [:, :, x.shape[-1]//2: 2+ (x.shape[-1]//2)])(original_input)
    last = Lambda(lambda x: x [:, :, -2:])(original_input)

    beg_mid = []
    beg_mid.append(Lambda(lambda x: x [:, :, :2])(original_input))
    beg_mid.append(Lambda(lambda x: x [:, :, -2:])(original_input))
    # beg_mid = concatenate([beg, last], axis=-1)
    # beg_mid_last = concatenate([beg_mid, last], axis=-1)

    # lstm_layer = LSTM()
    c1 = conv2d_block(
        beg_mid, num_filters, kernel_size=3, batchnorm=batchnorm
    )

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, num_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, num_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    middle = conv2d_block(p3, num_filters * 4, kernel_size=3)

    u1 = deconv2d_block_time_dist(middle, num_filters=num_filters*4, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c3, activation=True)
    u2 = deconv2d_block_time_dist(u1, num_filters=num_filters*2, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c2, activation=True)
    u3 = deconv2d_block_time_dist(u2, num_filters=num_filters, dropout=dropout, kernel_size=3, batchnorm=batchnorm, concat_layer=c1, activation=True)

    final_conv_local = Conv2D(filters=1, kernel_size=1,  activation="softmax", padding='same')(u3)

    final_concat = concatenate([raw_ouput, final_conv_local])

    final_conv = Conv2D(filters=1, kernel_size=1,  activation="sigmoid", padding='same')(final_concat)

    return final_conv
