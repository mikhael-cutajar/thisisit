import os
import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt

import pdb

reference_direction = np.identity(3).flatten()
reference_spacing = np.array((1.0, 1.0, 1))


os.chdir("ALL/data/raw")
num = 0
for i in os.listdir("."):
    # for i in ['0522c0576']:
    print(num)
    num += 1
    print(i)
    print("start")
    os.chdir(i)

    image = sitk.ReadImage("real_image.nrrd")

    image_size = image.GetSize()
    image_spacing = image.GetSpacing()
    image_origin = image.GetOrigin()
    image_direction = image.GetDirection()

    # pdb.set_trace()

    label = np.zeros((image_size[2], image_size[0], image_size[1]))

    # label_index = {
    #     "BrainStem.nrrd": 1,
    #     "Chiasm.nrrd": 2,
    #     "Mandible.nrrd": 3,
    #     "OpticNerve_L.nrrd": 4,
    #     "OpticNerve_R.nrrd": 5,
    #     "Parotid_L.nrrd": 6,
    #     "Parotid_R.nrrd": 7,
    #     "Submandibular_L.nrrd": 8,
    #     "Submandibular_R.nrrd": 9,
    # }
    label_index = {
        # anchor
        "Brain Stem.nrrd": 1,
        "Eye-L.nrrd": 2,
        "Eye-R.nrrd": 3,
        "Mandible.nrrd": 4,
        "Spinal Cord.nrrd": 5,  # 21
        "Trachea.nrrd": 6,  # 28
        # mid
        # "Brachial Plexus.nrrd": 7,  # 1
        "ConstrictorNaris.nrrd": 7,  # 3
        "Larynx.nrrd": 8,  # 9
        # "Oral Cavity.nrrd": 10,  # 16
        "Parotid L.nrrd": 9,  # 17
        "Parotid R.nrrd": 10,  # 18
        "SmgL.nrrd": 11,  # 19
        "SmgR.nrrd": 12,  # 20
        "Temporal Lobe L.nrrd": 13,  # 23
        "Temporal Lobe R.nrrd": 14,  # 24
        "Thyroid.nrrd": 15,  # 25
        # "TMJL.nrrd": 18,  # 26
        # "TMJR.nrrd": 19,  # 27
        # "Lens L.nrrd": 20,  # 10
        # "Lens R.nrrd": 21,  # 11
        # "Sublingual Gland.nrrd": 22,  # 22
        # low
        # "Ear-L.nrrd": 23,  # 4
        # "Ear-R.nrrd": 24,  # 5
        # "Hypophysis.nrrd": 25,  # 8
        "Optical Chiasm.nrrd": 16,  # 13
        "Optical Nerve L.nrrd": 17,  # 14
        "Optical Nerve R.nrrd": 18,  # 15
    }
    # generate itk label

    for j in os.listdir("./structures"):
        if j in label_index:
            print(j)
            structure = sitk.ReadImage("./structures/" + j)
            npstructure = sitk.GetArrayFromImage(structure)
            label[npstructure == 1] = label_index[j]

    print(np.unique(label))

    itkLabel = sitk.GetImageFromArray(label)
    itkLabel.SetSpacing(image_spacing)
    itkLabel.SetOrigin(image_origin)
    itkLabel.SetDirection(image_direction)

    sitk.WriteImage(itkLabel, "inter_label.nii.gz")

    new_size = np.array(image_size) * np.array(image_spacing) / reference_spacing
    reference_size = [int(new_size[0]), int(new_size[1]), int(new_size[2])]

    reference_image = sitk.Image(reference_size, sitk.sitkFloat32)
    reference_image.SetOrigin(image_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # set interpolator for CT data
    resampler.SetInterpolator(sitk.sitkLinear)

    out_img = resampler.Execute(image)
    # set interpolator for label
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    out_label = resampler.Execute(itkLabel)

    # print(out_img.shape)
    # print(out_label.shape)
    sitk.WriteImage(out_img, "data.nii.gz")
    sitk.WriteImage(out_label, "label.nii.gz")

    print("done")

    os.chdir("..")
