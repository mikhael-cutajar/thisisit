import argparse
import logging
import os
from black import out

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import nrrd
import matplotlib.pyplot as plt
import torchio as tio

from model.ynet3d import UNet3D


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        nargs="+",
        help="Filenames of input images",
        required=True,
    )
    parser.add_argument(
        "--mask",
        "-mk",
        metavar="MASK",
        nargs="+",
        help="Filenames of mask images",
        required=True,
    )

    parser.add_argument(
        "--output", "-o", metavar="INPUT", nargs="+", help="Filenames of output images"
    )
    parser.add_argument(
        "--viz",
        "-v",
        action="store_true",
        help="Visualize the images as they are processed",
    )
    parser.add_argument(
        "--no-save", "-n", action="store_true", help="Do not save the output masks"
    )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Scale factor for the input images",
    )

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f"{split[0]}_OUT{split[1]}"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray(
            (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
        )


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    mask = args.mask
    out_files = get_output_filenames(args)

    net = UNet3D(n_class=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded!")

    for i, filename in enumerate(in_files):
        logging.info(f"\nPredicting image {filename} ...")

        subject = tio.Subject(
            image=tio.ScalarImage(filename),
            mask=tio.LabelMap(mask),
        )
        # subject.plot()
        print(subject["image"].shape)
        full_img = subject["image"][tio.DATA][0]
        print(full_img.shape)
        true_mask = subject["mask"][tio.DATA][0]
        print(true_mask.shape)

        HOUNSFIELD_MIN, HOUNSFIELD_MAX = -500, 1000
        training_transform = tio.Compose(
            [
                tio.CopyAffine("image"),
                tio.Clamp(out_min=HOUNSFIELD_MIN, out_max=HOUNSFIELD_MAX),
            ]
        )

        subject = [subject]
        training_set = tio.SubjectsDataset(subject, training_transform)

        sampler_label = tio.data.LabelSampler(patch_size=(64, 64, 64))

        patches_training_set = tio.Queue(
            subjects_dataset=training_set,
            max_length=1,
            samples_per_volume=1,
            sampler=sampler_label,
            num_workers=1,
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        print(len(patches_training_set))
        training_loader_patches = torch.utils.data.DataLoader(
            patches_training_set, batch_size=1
        )

        image_mask = []
        net.eval()
        with torch.no_grad():
            for batch in training_loader_patches:
                images = batch["image"][tio.DATA]
                true_masks = batch["mask"][tio.DATA]
                image_mask.append(images)
                image_mask.append(true_masks)

                print(images.shape)
                images = images.cuda()
                # inputs = patches_batch["image"][tio.DATA].to(device)
                # locations = patches_batch[tio.LOCATION]
                output = net(images)
                output = (output > 0.5).float()

        print(output.shape)

        fig, ax = plt.subplots(3, 20)

        # single_class = output[0][3].cpu()
        print(image_mask[0].shape)
        print(image_mask[1].shape)
        image = image_mask[0][0][0]
        true_mask = image_mask[1][0][0]
        print(output.shape)
        labels = torch.argmax(output, dim=1).cpu()
        print(labels.shape)
        for i in range(20):
            ax[0, i].imshow(image[..., 30 + i])
            ax[1, i].imshow(true_mask[..., 30 + i])
            ax[2, i].imshow(labels[0][..., 30 + i])

        plt.show()
