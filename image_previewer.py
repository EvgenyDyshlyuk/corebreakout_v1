#%%
import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
import pickle

from PIL import Image
import cv2

from corebreakout import CoreColumn


# get specific filenames in directory
def get_filenames(directory, extension):    
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

# def jp2_to_jpg(file_paths):
#     for file_path in file_paths:
#         file_name = file_path.split(os.sep)[-1]
#         file_stem = file_name.split('.')[0]
#         image = cv2.imread(file_path)
#         cv2.imwrite(directory + os.sep + file_stem + '.jpeg', image)

def preview_images(file_paths, size=8):
    for file_path in file_paths:
        file_name = file_path.split(os.sep)[-1]
        file_stem  = os.path.splitext(file_name)[0]

        image_arr = cv2.imread(file_path)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(size, size))       
        plt.imshow(image_arr)
        plt.title(label = file_stem, c = 'white')
        # remove ticks
        plt.xticks([])
        plt.yticks([])
        plt.show()

#%%
directory = 'assets/204_19-3A'

file_paths = get_filenames(directory, '.jp2')
print(len(file_paths))

preview_images(file_paths[0:3])

file_path = 'assets/204_19-3A/CoreColumn_2211.50_2262.30.pkl'
with open(file_path, 'rb') as f:
    col = pickle.load(f)
    col.slice_depth(top = 2211.5, base = 2217).plot(figsize=(10, 40), major_kwargs = {'labelsize' : 12}, minor_kwargs={'labelsize' : 8})
# %%
# # make column from depths and  image
# labels = np.load('assets/Q204_data/train_data_figshare/204-19-3A_labels.npy')
# print(labels)
# depths = np.load('assets/Q204_data/train_data_figshare/204-19-3A_depth.npy')
# print(depths)
# image = np.load('assets/Q204_data/train_data_figshare/204-19-3A_image.npy')
# #plt.figure(figsize = (10,800))
# #plt.imshow(image)

# # irregular depths data some points are above the others...
# print(depths[np.where(np.diff(depths) < 0)])
# print(np.where(np.diff(depths) < 0)[0])
# print(depths[129990:130009])
# print(depths[259990:260009])
# col = CoreColumn(image, depths=depths, add_mode='collapse')
# %%
