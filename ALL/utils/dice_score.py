import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# anchor = [
#     "Brain Stem.nrrd",  # 2
#     "Eye-L.nrrd",  # 6
#     "Eye-R.nrrd",  # 7
#     "Mandible.nrrd",  # 12
#     "Spinal Cord.nrrd",  # 21
#     # "TMJL.nrrd",  # 26
#     # "TMJR.nrrd",  # 27
#     "Trachea.nrrd",  # 28
# ]

# Loss Function and coefficients to be used during training:
def dice_coefficient(predicted_image, true_image):
    assert predicted_image.size() == true_image.size()

    input = predicted_image.reshape(-1)
    input = (input > 0.5).float()
    target = true_image.reshape(-1)

    inter = torch.dot(input, target)
    sets_sum = torch.sum(input) + torch.sum(target)

    smoothing_factor = 1

    dice_result = (2 * inter) / (sets_sum)

    return dice_result.item()


def dice_coefficient_loss(predicted_image, true_image):
    print("loss")

    prediction = predicted_image[0][1]
    mask = true_image[0]
    # exit()
    dice_score = dice_coefficient(prediction, mask)

    loss = 1 - dice_score
    # print(loss)
    return loss


def dice_coefficient_score(predicted_image, true_image):
    print("score")

    prediction = predicted_image[0][1]
    mask = true_image[0]

    dice_score = dice_coefficient(prediction, mask)

    return dice_score


def multi_class_loss(predicted_image, true_image):

    dice_scores = [0] * 29

    classes = np.unique(true_image.cpu())
    classes = classes.astype(int)

    true_image = (
        F.one_hot(true_image.to(torch.int64), 29).permute(0, 4, 1, 2, 3).float()
    )

    num_classes = len(predicted_image[0])

    dice_score = 0
    for i in classes:
        prediction = predicted_image[0][i]
        mask = true_image[0][i]

        score = dice_coefficient(prediction, mask)
        dice_scores[i] = score

    for unique_class in classes:
        if unique_class != 0:
            dice_score += dice_scores[unique_class]

    dice_score = dice_score / len(classes)

    loss = 1 - dice_score

    return loss


def multi_class_score(predicted_image, true_image):

    dice_scores = [0] * 29

    classes = np.unique(true_image.cpu())
    classes = classes.astype(int)

    true_image = (
        F.one_hot(true_image.to(torch.int64), 29).permute(0, 4, 1, 2, 3).float()
    )

    num_classes = len(predicted_image[0])

    dice_score = 0
    for i in classes:
        prediction = predicted_image[0][i]
        mask = true_image[0][i]

        score = dice_coefficient(prediction, mask)
        dice_scores[i] = score

    for i in classes:
        # exclude background in value
        if i != 0:
            dice_score += dice_scores[i]

    dice_score = dice_score / (len(classes) - 1)

    return dice_score, dice_scores, classes
