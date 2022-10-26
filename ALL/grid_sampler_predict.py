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


def predict_img(net, full_img, device, true_mask, scale_factor=1, out_threshold=0.5):
    net.eval()
    # img = torch.from_numpy(
    #     BasicDataset.preprocess(full_img, scale_factor, is_mask=False)
    # )
    # img = full_img[0]
    # img = img.unsqueeze(0)

    with torch.no_grad():

        # print(full_img.shape)
        output_list = []
        # for i in range(4):
        model_input = full_img
        model_input = model_input.to(device=device, dtype=torch.float32)
        model_input = torch.unsqueeze(model_input, 0)
        # model_input = torch.unsqueeze(model_input, 0)

        output = net(model_input)

        output = (output > 0.5).float()

        output = output[0][1].cpu()

        full_img = torch.squeeze(full_img, 0)
        true_mask = torch.squeeze(true_mask, 0)

        print(full_img.shape)
        print(true_mask.shape)
        print(output.shape)
        print("----------")
        fig, ax = plt.subplots(3, 20)
        for i in range(20):
            ax[0, i].imshow(full_img[..., 30 + i])
            ax[1, i].imshow(true_mask[..., 30 + i])
            ax[2, i].imshow(output[..., 30 + i])
        # plt.imshow(img[0][0][75].cpu())
        plt.show()

    #     for i in range(0, 10):
    #         full_mask = tf(probs[i].cpu()).squeeze()

    #         img = cv2.imread("output.jpg", 0)
    #         img = np.asarray(full_mask)

    #         cv2.imshow("image", img)
    #         cv2.waitKey(0)

    # if net.n_classes == 1:
    #     return (full_mask > out_threshold).numpy()
    # else:
    #     return (
    #         F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    #     )


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

    # parser.add_argument(
    #     "--image_mask",
    #     "-x",
    #     metavar="MASK",
    #     help="True mask of image",
    # )

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

    net = UNet3D(n_class=10)
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

        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size=(128, 128, 64),
            patch_overlap=(4, 4, 4),
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        # subjects_list = [subject]
        # subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

        # print(subjects_dataset[0]["image"].shape)
        # subjects_dataset[0].plot()
        net.eval()
        # img = torch.from_numpy(
        #     BasicDataset.preprocess(full_img, scale_factor, is_mask=False)
        # )
        # img = full_img[0]
        # img = img.unsqueeze(0)

        with torch.no_grad():
            for i, patches_batch in enumerate(patch_loader):
                print(i)
                inputs = patches_batch["image"][tio.DATA].to(device)
                locations = patches_batch[tio.LOCATION]
                print(inputs.shape)
                output = net(inputs)
                # probabilities = net(inputs).softmax(dim=CHANNELS_DIMENSION)
                aggregator.add_batch(output, locations)

        output_tensor = aggregator.get_output_tensor()
        print(output_tensor.shape)

        fig, ax = plt.subplots(3, 20)
        testin = output_tensor[7]
        testin = (testin > 0.5).float()

        for i in range(20):
            ax[0, i].imshow(full_img[..., 65 + i])
            ax[1, i].imshow(true_mask[..., 65 + i])
            ax[2, i].imshow(testin[..., 65 + i])
        # plt.imshow(img[0][0][75].cpu())
        plt.show()
        # print(locations)
        # mask = predict_img(
        #     net=net,
        #     full_img=subjects_dataset[0]["image"][tio.DATA],
        #     scale_factor=args.scale,
        #     out_threshold=args.mask_threshold,
        #     true_mask=subjects_dataset[0]["mask"][tio.DATA],
        #     device=device,
        # )

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f"Mask saved to {out_filename}")

        # if args.viz:
        #     logging.info(
        #         f"Visualizing results for image {filename}, close to continue..."
        #     )
        #     plot_img_and_mask(img, mask)
