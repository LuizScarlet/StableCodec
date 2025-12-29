import numpy as np
import os
import h5py
from PIL import Image
from tqdm import tqdm
import glob

with h5py.File('Dataset/dataset.hdf5', 'w') as f:
    idx = 0
    for file in tqdm(os.listdir('Dataset/Flickr2K/')):
        img = Image.open(os.path.join('Dataset/Flickr2K/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('Dataset/DIV2KTrain/')):
        img = Image.open(os.path.join('Dataset/DIV2KTrain/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('Dataset/CLIC2020Train/')):
        img = Image.open(os.path.join('Dataset/CLIC2020Train/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('Dataset/CLIC2020Val/')):
        img = Image.open(os.path.join('Dataset/CLIC2020Val/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('Dataset/CLIC2021Test/')):
        img = Image.open(os.path.join('Dataset/CLIC2021Test/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    