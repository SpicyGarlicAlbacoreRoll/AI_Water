from keras import backend as K
from keras.metrics import MeanIoU
import tensorflow as tf
import numpy as np

"""Dice Coefficient Loss function based on Sørensen–Dice coefficient method.
Performs well when there is an imbalance between classes in image segmnetation
https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient 
A more readable explanation is available near the end of this article 
https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2#:~:text=3.-,Dice%20Coefficient%20(F1%20Score),of%20union%20in%20section%202).
The dice_coefficient is the 2 * the intersection of our prediction and ground truth label divided by the sum of the two"""
def dice_coefficient(y_true, y_pred, smooth=1):

    # flatten prediction and mask along single axis
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # calculate the intersection between the prediction and ground truth images
    intersection = K.sum(K.abs(y_true_flat * y_pred_flat), axis=-1)
    truth_pred_sum = K.sum(K.square(y_true_flat), axis=-1) + K.sum(K.square(y_pred_flat), axis=-1)

    return (2. * intersection + smooth) / (truth_pred_sum + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

# implementation borrowed from here:
'''   https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py'''
def cosh_dice_coefficient_loss(y_true, y_pred):
    x = dice_coefficient_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

# def soft_dice_coefficient(y_true, y_pred, smooth=1):

#     # flatten prediction and mask along single axis
#     y_true_flat = K.flatten(y_true)
#     y_pred_flat = K.flatten(y_pred)

#     # calculate the intersection between the prediction and ground truth images
#     intersection = K.sum(K.abs(y_true_flat * y_pred_flat), axis=-1)
#     truth_pred_sum = K.sum(K.square(y_true_flat), axis=-1) + K.sum(K.square(y_pred_flat), axis=-1)

#     return (2. * intersection + smooth) / (truth_pred_sum + smooth)

# def soft_dice_coefficient_loss(y_true, y_pred):
#     return 1-dice_coefficient(y_true, y_pred)


"""A slightly better loss function than dice for unbalanced datasets"""
def Jaccard_coefficient(y_true, y_pred, smooth=100):
    # flatten prediction and mask along single axis
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # calculate the intersection between the prediction and ground truth images
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection

    smooth = 1.0
    jac = (intersection + smooth) / (union + smooth)
    return (1 - jac) * smooth



def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth




# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         x = y_pred[y_pred > t]
#         y_pred_ = tf.cast(x, tf.int32)
#         score, up_opt = MeanIoU(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)