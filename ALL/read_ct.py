import numpy as np
import nrrd

# Some sample numpy data
# data = np.zeros((5,4,3,2))
# filename = "data/raw/0522c0012/img.nrrd"

# # Read the data back from file
# readdata, header = nrrd.read(filename)
# print(readdata.shape)
# print(header)

listofzeros = [0] * 7
print(listofzeros)

label_index = {
    # anchor
    "Backgounrd": 0,
    "Brain Stem.nrrd": 1,
    "Eye-L.nrrd": 2,
    "Eye-R.nrrd": 3,
    "Mandible.nrrd": 4,
    "Spinal Cord.nrrd": 5,  # 21
    "Trachea.nrrd": 6,  # 28
    # mid
    "Brachial Plexus.nrrd": 7,  # 1
    "ConstrictorNaris.nrrd": 8,  # 3
    "Larynx.nrrd": 9,  # 9
    "Oral Cavity.nrrd": 10,  # 16
    "Parotid L.nrrd": 11,  # 17
    "Parotid R.nrrd": 12,  # 18
    "SmgL.nrrd": 13,  # 19
    "SmgR.nrrd": 14,  # 20
    "Temporal Lobe L.nrrd": 15,  # 23
    "Temporal Lobe R.nrrd": 16,  # 24
    "Thyroid.nrrd": 17,  # 25
    "TMJL.nrrd": 18,  # 26
    "TMJR.nrrd": 19,  # 27
    "Lens L.nrrd": 20,  # 10
    "Lens R.nrrd": 21,  # 11
    "Sublingual Gland.nrrd": 22,  # 22
    # low
    "Ear-L.nrrd": 23,  # 4
    "Ear-R.nrrd": 24,  # 5
    "Hypophysis.nrrd": 25,  # 8
    "Optical Chiasm.nrrd": 26,  # 13
    "Optical Nerve L.nrrd": 27,  # 14
    "Optical Nerve R.nrrd": 28,  # 15
}

print(label_index.keys())
