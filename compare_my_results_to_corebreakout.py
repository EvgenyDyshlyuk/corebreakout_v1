#%%
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
import pickle

from PIL import Image
import cv2

from corebreakout import CoreColumn

from sklearn.preprocessing import LabelEncoder


# get specific filenames in directory
def get_filenames(directory, extension):    
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

# def jp2_to_jpg(file_paths):
#     for file_path in file_paths:
#         file_name = file_path.split(os.sep)[-1]
#         file_stem = file_name.split('.')[0]
#         image = cv2.imread(file_path)
#         cv2.imwrite(directory + os.sep + file_stem + '.jpeg', image)

def img_preview(file_path, figsize=(10, 800)):

    file_name = file_path.split(os.sep)[-1]
    file_stem = file_name.split('.')[0]
    file_ext = file_name.split('.')[-1]

    if file_ext == 'npy':
        image_arr=np.load(img_path)        
    else:
        image_arr = cv2.imread(file_path)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)       
    plt.imshow(image_arr)
    plt.title(label = file_stem, c = 'white')
    plt.xticks([])
    plt.yticks([])
    plt.show()

#%%
# Downloaded data from CoreMDLR

img_path = r'C:\Users\evgen\Documents\coremdlr\Q204_data\train_data_figshare\204-19-3A_image.npy'
depth_path = r'C:\Users\evgen\Documents\coremdlr\Q204_data\train_data_figshare\204-19-3A_depth.npy'
labels_path = r'C:\Users\evgen\Documents\coremdlr\Q204_data\train_data_figshare\204-19-3A_labels.npy'
# labelsII_path = r'C:\Users\evgen\Documents\coremdlr\Q204_data\train_data_figshare\204-19-3A_labelsII.npy'
#logs_path = r'C:\Users\evgen\Documents\coremdlr\Q204_data\train_data_figshare\204-19-3A_logs.las'

img=np.load(img_path)
print(img.shape)
depths=np.load(depth_path)
print(depths.shape)
labels = np.load(labels_path)
print(labels.shape)
plt.plot(depths)
# For some reason (storage? npy?) the depths data is not monotonic so:
# sort and return sorted indices
sorted_indices = np.argsort(depths)
depths = depths[sorted_indices]
plt.plot(depths)

labels = labels[sorted_indices]
img = img[sorted_indices]

col = CoreColumn(img, depths)
print(col)

col.slice_depth(top = 2185, base = 2190).plot(figsize=(10, 40), major_kwargs = {'labelsize' : 12}, minor_kwargs={'labelsize' : 8})


#%%
# Compare to my download and processing 
    
article_my = 'output\\article'
pkl_files = glob.glob(pathname=article_my + '/*/*.pkl')
print(len(pkl_files))
for f in pkl_files:    
    print(f)

# same well index 1
pkl_file = pkl_files[1]
with open(pkl_file, 'rb') as f:
    core_column = pickle.load(f)
    print(core_column)

core_column.slice_depth(top = 2185, base = 2190).plot(figsize=(10, 40), major_kwargs = {'labelsize' : 12}, minor_kwargs={'labelsize' : 8})
#%%