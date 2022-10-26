import torch
import math
from utils.dice_score import dice_coefficient, dice_coefficient_loss
import numpy as np
import torch.nn.functional as F

x = torch.tensor(
    [
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        ],
    ]
)

y = torch.tensor(
    [
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    ]
)

print(x.shape)
print(y.shape)
print(dice_coefficient_loss(x, y))

# (unique, counts) = np.unique(x, return_counts=True)
# frequencies = np.asarray((unique, counts)).T

# print(frequencies)

# (unique, counts) = np.unique(y, return_counts=True)
# frequencies = np.asarray((unique, counts)).T

# print(frequencies)
