import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
from functools import total_ordering
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from utils.data_loading import CT_data
from utils.evaluate import evaluate

from utils.dice_score import dice_coefficient, dice_coefficient_loss, multi_class_loss

# from evaluate import evaluate
from model.ynet3d import UNet3D
import numpy as np
import csv
import torchio as tio
import torchvision

# from IPython import display
# from unet import UNet


from tqdm.auto import tqdm
from img_viewer import list_images

import nrrd

# print(torch.cuda.is_available())
# print("here")
# exit()
# dir_scans = Path("./dataset/part3/")
dir_scans = Path("./data/raw/")

dir_checkpoint = Path("./checkpoints/")

no_classes = 29


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
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")
    logging.info(f"Using device {device}")

    net = UNet3D(n_class=no_classes)
    # print(torch.cuda.is_available())
    # print("here")
    # exit()
    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    # net = nn.DataParallel(net, device_ids=["cuda:0", "cuda:1",'cuda:2', 'cuda:3'], output_device=device)

    all_image_urls = []
    all_mask_urls = []

    csv_file = "all.csv"

    with open(csv_file, "w") as f:
        writer = csv.writer(f)

    for root, d_names, files in os.walk(dir_scans):
        folder_names = d_names
        for folder in folder_names:
            new_path = os.path.join(dir_scans, folder)
            img_path = os.path.join(new_path, "data.nii.gz")
            all_image_urls.append(img_path)

            mask_path = os.path.join(new_path, "label.nii.gz")
            all_mask_urls.append(mask_path)

        break

    assert len(all_image_urls) == len(all_mask_urls)
    # print(all_image_urls)
    HOUNSFIELD_MIN, HOUNSFIELD_MAX = -500, 1000

    training_transform = tio.Compose(
        [
            # tio.CopyAffine("image"),
            tio.Clamp(out_min=HOUNSFIELD_MIN, out_max=HOUNSFIELD_MAX),
            # new ones
            # tio.ToCanonical(),
            # tio.Resample((1, 1, 1)),
            # tio.RandomNoise(p=0.2),
            # tio.RandomAffine(scales=(0.8, 1.2), p=0.3),
            # tio.RandomElasticDeformation(p=0.1),
            # tio.RandomMotion(p=0.2),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RandomNoise(p=0.25),
            # tio.OneOf(
            #     {
            #         tio.RandomAffine(): 0.8,
            #         tio.RandomElasticDeformation(): 0.2,
            #     }
            # ),
        ]
    )

    validation_transform = tio.Compose(
        [
            # tio.CopyAffine("image"),
            tio.Clamp(out_min=HOUNSFIELD_MIN, out_max=HOUNSFIELD_MAX),
            # tio.ToCanonical(),
            # tio.Resample((1, 1, 1)),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ]
    )

    subjects = []
    for (image_path, label_path) in zip(all_image_urls, all_mask_urls):
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            mask=tio.LabelMap(label_path),
        )
        subjects.append(subject)
        # print(image_path)
        # print(subject["image"][tio.DATA].shape)

        # (unique, counts) = np.unique(subject["mask"][tio.DATA], return_counts=True)
        # frequencies = np.asarray((unique, counts)).T

        # print(frequencies)
        # exit()
        # print("------")
        # print("------")
        # print("------")

    # exit()
    dataset = tio.SubjectsDataset(subjects)
    print("Dataset size:", len(dataset), "subjects")

    # for i in dataset:
    #     print(i.shape)

    # print(len(dataset))
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

    # for i in training_set:
    #     print(i["image"][tio.DATA].shape)

    # print("validation")
    # for i in validation_set:
    #     print(i["image"][tio.DATA].shape)
    print("Training set:", len(training_set), "subjects")
    print("Validation set:", len(validation_set), "subjects")

    print(len(training_set))
    # print("hello")
    # training_instance = training_set[6]  # transform is applied inside SubjectsDataset

    probabilities = {
        0: 20,
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 2,
        18: 2,
        19: 2,
        20: 5,
        21: 7,
        22: 7,
        23: 5,
        24: 5,
        25: 5,
        26: 5,
        27: 5,
        28: 5,
    }

    training_batch_size = 1
    validation_batch_size = 1
    max_queue_length = 25
    sampler_label = tio.data.LabelSampler(patch_size=(128, 128, 64))
    # sampler_label = tio.data.LabelSampler(
    #     patch_size=(128, 128, 64), label_probabilities=probabilities
    # )
    # sampler_random = tio.data.UniformSampler(patch_size=(128, 128, 64))

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=5,
        sampler=sampler_label,
        num_workers=1,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size
    )

    # print("--------")
    print(len(patches_training_set))
    # print(patches_training_set)
    # for batch in training_loader_patches:
    #     images = batch["image"][tio.DATA].squeeze(0).squeeze(0)
    #     true_masks = batch["mask"][tio.DATA].squeeze(0).squeeze(0)
    #     print("@@")
    #     print(true_masks.shape)
    #     list_images(images, true_masks)
    # print("i know")
    # exit()

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=2,
        sampler=sampler_label,
        num_workers=1,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size
    )

    #######################################################################
    # Params
    #######################################################################

    learning_rate = 0.00001

    amp = True
    net.to(device=device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # optimizer = optim.RMSprop(
    #     net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
    # )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "max", patience=2
    # )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    epochs = 100
    best_val_score = 0
    #######################################################################

    #######################################################################
    # Training
    #######################################################################
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=len(patches_training_set),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="img",
        ) as pbar:
            i = 0
            for batch in training_loader_patches:
                i += 1

                images = batch["image"][tio.DATA]
                true_masks = batch["mask"][tio.DATA]
                # print(images.shape)
                # print(true_masks.shape)
                # list_images(
                #     images.cpu().squeeze(0).squeeze(0),
                #     true_masks.cpu().squeeze(0).squeeze(0),
                # )

                # exit()
                true_masks = true_masks.squeeze(0)

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

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

                # division_step = len(patches_training_set) // (10 * training_batch_size)
                # if division_step > 0:
        val_score = evaluate(
            net,
            validation_loader_patches,
            device,
        )

        logging.info("Validation Dice score: {}".format(val_score))

        if val_score > best_val_score:
            best_val_score = val_score
            print("new best model")
            torch.save(net.state_dict(), "BEST_all_model.pth")
    #######################################################################

    print("val score")
    print(val_score)

    # I added this
    torch.save(net.state_dict(), "all_model.pth")
    logging.info("Saved Model")
    sys.exit(0)

    print("val score")
    print(val_score)
