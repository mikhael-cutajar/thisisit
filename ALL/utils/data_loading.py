import os
from torch.utils.data import Dataset
from torchvision import transforms as transforms
import numpy as np
import nrrd
import cv2
import torch
import matplotlib.pyplot as plt

# from patchify import patchify, unpatchify


class CT_data(Dataset):
    # (self,
    #              images_dir: str,
    #              scale: float = 1.0,
    #              mask_suffix: str = '')
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir

        self.all_image_urls = []
        self.all_scan_mask_urls = []

        mask_structures = [
            "BrainStem.nrrd",  # 1
            "Chiasm.nrrd",  # 2
            "Mandible.nrrd",  # 3
            "OpticNerve_L.nrrd",  # 4
            "OpticNerve_R.nrrd",  # 5
            "Parotid_L.nrrd",  # 6
            "Parotid_R.nrrd",  # 7
            "Submandibular_L.nrrd",  # 8
            "Submandibular_R.nrrd",  # 9
        ]

        for root, d_names, files in os.walk(self.image_dir):
            folder_names = d_names
            for folder in folder_names:
                self.new_path = os.path.join(self.image_dir, folder)
                img_path = os.path.join(self.new_path, "img.nrrd")
                self.all_image_urls.append(img_path)

                scan_mask_paths = []
                for structure in mask_structures:
                    mask_path = str(self.new_path) + "/structures/" + str(structure)
                    scan_mask_paths.append(mask_path)
                self.all_scan_mask_urls.append(scan_mask_paths)

            break

    def __len__(self):
        return len(self.all_image_urls)

    def __getitem__(self, index):

        img_path = self.all_image_urls[index]
        img, header = nrrd.read(img_path)
        # print(img_path)

        mask_paths = self.all_scan_mask_urls[index]
        # #
        all_class_masks = []
        # print(len(mask_paths))

        for i, path in enumerate(mask_paths):
            # print(i)
            # print(path)
            x = i + 1
            single_class_mask, header = nrrd.read(path)
            single_class_mask = single_class_mask * x
            all_class_masks.append(single_class_mask)
        # exit()
        mask = np.zeros(shape=(all_class_masks[0].shape))

        all_class_masks = np.array(all_class_masks)
        # print(all_class_masks[0].shape)
        # print(all_class_masks[1].shape)
        # for scan in all_class_masks:
        #     print(scan.shape)

        mask = np.add(all_class_masks[0], all_class_masks[1])
        mask = np.add(mask, all_class_masks[2])
        mask = np.add(mask, all_class_masks[3])
        mask = np.add(mask, all_class_masks[4])
        mask = np.add(mask, all_class_masks[5])
        mask = np.add(mask, all_class_masks[6])
        mask = np.add(mask, all_class_masks[7])
        mask = np.add(mask, all_class_masks[8])
        # mask = np.add(mask, all_class_masks[9])

        # print(all_class_masks.shape)
        # mask = np.stack(all_class_masks)
        # print(mask.shape)
        # exit()
        # (unique, counts) = np.unique(all_class_masks[0], return_counts=True)
        # frequencies = np.asarray((unique, counts)).T

        # print(frequencies)

        # (unique, counts) = np.unique(all_class_masks[1], return_counts=True)
        # frequencies = np.asarray((unique, counts)).T

        # print(frequencies)

        # mask3 = np.add(all_class_masks[0], all_class_masks[2])

        # (unique, counts) = np.unique(mask, return_counts=True)
        # frequencies = np.asarray((unique, counts)).T

        # print(frequencies)
        # exit()

        # # print(header)

        # img = np.moveaxis(img, 0, -1)
        # mask = np.moveaxis(mask, 0, -1)

        # # img = cv2.resize(img, (512, 16))
        # # mask = cv2.resize(mask, (512, 16))

        # img = cv2.resize(img, (64, 64))
        # mask = cv2.resize(mask, (64, 64))

        # print(img.shape)

        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)

        img = cv2.resize(img, (512, 128))
        mask = cv2.resize(mask, (512, 128))

        # plt.imshow(mask[67])
        # plt.show()
        # exit()
        # plt.imshow(mask[67])
        # plt.show()
        # img_patches = np.zeros(shape=(16, 128, 128, 128))
        # mask_patches = np.zeros(shape=(16, 128, 128, 128))

        img_patches = np.zeros(shape=(64, 128, 64, 64))
        mask_patches = np.zeros(shape=(64, 128, 64, 64))

        img_set = []
        mask_set = []

        img_split1 = np.dsplit(img, 8)
        stack_num = 0
        for i in img_split1:
            img_split2 = np.hsplit(i, 8)
            for j in img_split2:
                # img_patches[stack_num] = j
                # print(j.shape)
                # plt.imshow(j[75])
                # plt.show()
                if (
                    stack_num == 19
                    or stack_num == 20
                    or stack_num == 27
                    or stack_num == 28
                ):
                    img_set.append(j)
                    # print(img_m.shape)
                stack_num += 1

        mask_split1 = np.dsplit(mask, 8)
        stack_num = 0
        for i in mask_split1:
            mask_split2 = np.hsplit(i, 8)
            for j in mask_split2:
                # mask_patches[stack_num] = j
                if (
                    stack_num == 19
                    or stack_num == 20
                    or stack_num == 27
                    or stack_num == 28
                ):
                    mask_set.append(j)
                stack_num += 1

        # img_m = np.stack((img_m,) * 3, axis=-1)
        # img_m = np.expand_dims(img_m, axis=0)
        # print(img_set)
        return {
            "image": torch.as_tensor(np.array(img_set)).float().contiguous(),
            "mask": torch.as_tensor(np.array(mask_set)).long().contiguous(),
        }
