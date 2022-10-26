import SimpleITK as sitk

# print(sitk.Version())
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import pydicom


def previous_slice():
    pass


def next_slice():
    pass


def process_key(event):
    if event.key == "j":
        previous_slice()
    elif event.key == "k":
        next_slice()


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({"j", "k"})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect("key_press_event", process_key)
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == "j":
        previous_slice(ax)
    elif event.key == "k":
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


################################################################3


def list_images(full_img, true_mask):
    fig, ax = plt.subplots(2, 20)
    for i in range(20):
        ax[0, i].imshow(full_img[..., 25 + i])
        ax[1, i].imshow(true_mask[..., 25 + i])
    # plt.imshow(img[0][0][75].cpu())
    plt.show()
