import os
import shutil

ORIG_DATA_DIR = '../data/kaggle_catdog/'
MINI_DATA_DIR = '../data/kaggle_catdog_small/'

for dataset in ['train', 'valid', 'test']:
    dataset_dir = os.path.join(MINI_DATA_DIR, dataset)
    os.mkdir(dataset_dir)
    for label in ['cat', 'dog']:
        label_dir = os.path.join(dataset_dir, label)
        os.mkdir(label_dir)

        
def copy_files(dataset, label, rng):
    f_names = ['{}.{}.jpg'.format(label, i) for i in rng]
    for f in f_names:
        src = os.path.join(ORIG_DATA_DIR, 'train', f)
        dst = os.path.join(MINI_DATA_DIR, dataset, label, f)
        shutil.copyfile(src, dst)

copy_files('train', 'cat', range(1000))
copy_files('train', 'dog', range(1000))
copy_files('valid', 'cat', range(1000, 1500))
copy_files('valid', 'dog', range(1000, 1500))
copy_files('test',  'cat', range(1500, 2000))
copy_files('test',  'dog', range(1500, 2000))

for dataset in ['train', 'valid', 'test']:
    for label in ['cat', 'dog']:
        print ('%s images for %s: %d'
               % (dataset,
                  label,
                  len(os.listdir(MINI_DATA_DIR + dataset + '/' + label))))


