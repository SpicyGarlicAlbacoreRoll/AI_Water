from keras import backend as K

"""Dice Coefficient Loss function based on Sørensen–Dice coefficient method.
Performs well when there is an imbalance between classes in image segmnetation
https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient """
def dice_coefficient(y_true, y_pred):

    # flatten prediction and mask along single axis
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # calculate the intersection between the prediciton and ground truth images
    intersection = K.sum(y_true_flat * y_pred_flat)

    smooth = 1.0
    return (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)