import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
from functools import total_ordering
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import nrrd


from utils.data_loading import CT_data
from utils.evaluate import evaluate

from utils.dice_score import dice_coefficient, dice_coefficient_loss, multi_class_loss

# from evaluate import evaluate
from model.ynet3d import UNet3D
import os
import numpy as np
import csv
from volumentations import *
import torchio as tio
import torchvision
from IPython import display
from unet import UNet

import enum
import time
import random
import multiprocessing
from pathlib import Path

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
# from unet import UNet
from scipy import stats
import matplotlib.pyplot as plt

from IPython import display
from tqdm.auto import tqdm

dir_scans = Path("./dataset/part3/")

dir_checkpoint = Path("./checkpoints/")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )

    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    logging.info(f"Using device {device}")

    net = UNet3D(n_class=3)

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f"Model loaded from {args.load}")

    net.to(device=device)

    all_image_urls = []
    all_mask_urls = []

    with open("anchor.csv", "w") as f:
        writer = csv.writer(f)

    for root, d_names, files in os.walk(dir_scans):
        folder_names = d_names
        for folder in folder_names:
            new_path = os.path.join(dir_scans, folder)
            img_path = os.path.join(new_path, "img.nrrd")
            all_image_urls.append(img_path)

            mask_path = os.path.join(new_path, "anchor.nrrd")
            all_mask_urls.append(mask_path)

        break

    assert len(all_image_urls) == len(all_mask_urls)

    HOUNSFIELD_MIN, HOUNSFIELD_MAX = -500, 1000

    training_transform = tio.Compose(
        [
            tio.CopyAffine("image"),
            tio.Clamp(out_min=HOUNSFIELD_MIN, out_max=HOUNSFIELD_MAX),
        ]
    )

    validation_transform = tio.Compose([tio.CopyAffine("image")])

    subjects = []
    for (image_path, label_path) in zip(all_image_urls, all_mask_urls):

        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            mask=tio.LabelMap(label_path),
        )
        subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)
    print("Dataset size:", len(dataset), "subjects")

    num_subjects = len(dataset)
    num_training_subjects = int(0.7 * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(
        subjects, num_split_subjects
    )

    training_set = tio.SubjectsDataset(training_subjects, training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform
    )

    print("Training set:", len(training_set), "subjects")
    print("Validation set:", len(validation_set), "subjects")

    print(len(training_set))
    training_instance = training_set[6]  # transform is applied inside SubjectsDataset

    training_batch_size = 1
    validation_batch_size = 1
    max_queue_length = 50
    sampler_label = tio.data.LabelSampler(patch_size=(128, 128, 64))
    # sampler_random = tio.data.UniformSampler(patch_size=(128, 128, 64))

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=10,
        sampler=sampler_label,
        num_workers=1,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=2,
        sampler=sampler_label,
        num_workers=1,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size
    )

    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size
    )

    #######################################################################
    # Params
    #######################################################################

    batch_size = 1
    learning_rate = 0.00001

    amp = True
    net.to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    epochs = 30

    # print(training_loader_patches[0]

    #######################################################################

    #######################################################################
    # Training
    #######################################################################

    for epoch in range(epochs):
        # print("new_epoch")
        net.train()
        epoch_loss = 0
        with tqdm(
            total=len(patches_training_set),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="img",
        ) as pbar:
            i = 0
            for batch in training_loader_patches:
                # print(i)
                i += 1
                images = batch["image"][tio.DATA]
                true_masks = batch["mask"][tio.DATA]

                true_masks = true_masks.squeeze(0)
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                print(images.shape)
                print(true_masks.shape)
                exit()
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    loss = criterion(masks_pred, true_masks) + multi_class_loss(
                        masks_pred,
                        true_masks.float(),
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss
                # experiment.log(
                #     {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                # )
                pbar.set_postfix(**{"loss (batch)": loss})

            val_score = evaluate(
                net,
                validation_loader_patches,
                device,
            )
            # scheduler.step(val_score)

            logging.info("Validation Dice score: {}".format(val_score))

    #######################################################################

    print("val score")
    print(val_score)

    # I added this
    torch.save(net.state_dict(), "anchor_128_128_64.pth")
    logging.info("Saved Model")
    sys.exit(0)

    print("val score")
    print(val_score)
