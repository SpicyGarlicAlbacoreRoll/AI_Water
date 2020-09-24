from keras import backend as K

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


"""A slightly better loss function for unbalanced datasets"""
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