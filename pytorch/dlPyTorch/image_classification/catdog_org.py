import os
from glob import glob

import numpy as np


PATH = '../data/dogs_and_cats/'
N_VALID = 2000

files = glob(os.path.join(PATH, '*/*.jpg'))
n_images = len(files)
print('Total number of images: %d' % n_images)

shuffle = np.random.permutation(n_images)
os.mkdir(os.path.join(PATH, 'valid'))
for t in ['train', 'valid']:
    for folder in ['dog/', 'cat/']:
        os.mkdir(os.path.join(PATH, t, folder))
for i in shuffle[:N_VALID]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(PATH, 'valid', folder, image))
for i in shuffle[N_VALID:]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(PATH, 'train', folder, image))

    
