import torch
import torch.nn.functional as F
from tqdm import tqdm
import csv

from utils.dice_score import (
    dice_coefficient,
    dice_coefficient_loss,
    dice_coefficient_score,
    multi_class_score,
)
import torchio as tio

# anchor = [
#     "Background",  # 0
#     "BrainStem.nrrd",  # 1
#     "Mandible.nrrd",  # 3
# ]

label_index = [
    # anchor
    "Backgounrd",  #: 0,
    "Brain Stem.nrrd",  #: 1,
    "Eye-L.nrrd",  #: 2,
    "Eye-R.nrrd",  #: 3,
    "Mandible.nrrd",  #: 4,
    "Spinal Cord.nrrd",  #: 5,  # 21
    "Trachea.nrrd",  #: 6,  # 28
    # mid
    "Brachial Plexus.nrrd",  #: 7,  # 1
    "ConstrictorNaris.nrrd",  #: 8,  # 3
    "Larynx.nrrd",  #: 9,  # 9
    "Oral Cavity.nrrd",  #: 10,  # 16
    "Parotid L.nrrd",  #: 11,  # 17
    "Parotid R.nrrd",  #: 12,  # 18
    "SmgL.nrrd",  #: 13,  # 19
    "SmgR.nrrd",  #: 14,  # 20
    "Temporal Lobe L.nrrd",  #: 15,  # 23
    "Temporal Lobe R.nrrd",  #: 16,  # 24
    "Thyroid.nrrd",  #: 17,  # 25
    "TMJL.nrrd",  #: 18,  # 26
    "TMJR.nrrd",  #: 19,  # 27
    "Lens L.nrrd",  #: 20,  # 10
    "Lens R.nrrd",  #: 21,  # 11
    "Sublingual Gland.nrrd",  # 22,  # 22
    # low
    "Ear-L.nrrd",  # 23,  # 4
    "Ear-R.nrrd",  # 24,  # 5
    "Hypophysis.nrrd",  #: 25,  # 8
    "Optical Chiasm.nrrd",  #: 26,  # 13
    "Optical Nerve L.nrrd",  #: 27,  # 14
    "Optical Nerve R.nrrd",  #: 28,  # 15
]

csv_file = "all.csv"


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # print(num_val_batches)
    # iterate over the validation set
    dice_scores = [0] * len(label_index)
    class_counter = [0] * len(label_index)

    for batch in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        images = batch["image"][tio.DATA]
        true_masks = batch["mask"][tio.DATA]

        # true_masks = true_masks.squeeze(0)
        # image = torch.unsqueeze(image, 0)
        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        # true_masks = true_masks.float()

        with torch.no_grad():

            mask_pred = net(images)

            true_masks = torch.squeeze(true_masks, 0)

            # repeat for loss
            all_dice_score, list_scores, classes = multi_class_score(
                mask_pred, true_masks.float()
            )

            dice_score += all_dice_score

            for i, class_score in enumerate(list_scores):
                dice_scores[i] += class_score

            for i in classes:
                class_counter[i] += 1

            # import pdb

            # pdb.set_trace()

    print(class_counter)
    for i, class_instances in enumerate(class_counter):
        print(class_instances)
        dice_scores[i] = dice_scores[i] / class_instances

    total = 0
    for i, score in enumerate(dice_scores):
        print(label_index[i] + " has dice " + str(score))
        if i > 0:
            total += score

    with open(csv_file, "a+", newline="") as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(dice_scores)

    total = total / 28
    net.train()
    return total
