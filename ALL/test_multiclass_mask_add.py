import torch
import math
from utils.dice_score import dice_coefficient, dice_coefficient_loss
import numpy as np
import torch.nn.functional as F

x = torch.tensor(
    [
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
    ]
)

y = torch.tensor(
    [
        [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        [[2.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]],
    ]
)

out_arr = np.add(x, y)

print(y.shape)
print(out_arr)
