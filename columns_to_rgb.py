# %%
#import image_previewer
import glob

from corebreakout import CoreColumn
import pickle

import numpy as np
import matplotlib.pyplot as plt

import colorsys


def slice_depths(top, base, slice_length):
    length = base - top
    n_slices = int(np.ceil(length / slice_length))
    slices = []
    for i in range(n_slices):
        top_slice = top + i * slice_length        
        if i == n_slices-1:
            base_slice = base
        else:
            base_slice = top + (i + 1) * slice_length
        slices.append((top_slice, base_slice))
    return slices


# def plot_column_slices(column_path, slice_length, figsize = (9, 800)):    
#     with open(column_path, 'rb') as f:
#         col = pickle.load(f)
#         column_top, column_base = column_depths_from_path(column_path)
#         column_length = column_base - column_top
#         if column_length <= slice_length:
#             col.slice_depth(top = column_top, 
#                                 base = column_base).plot(
#                                     figsize=figsize,
#                                     major_kwargs = {'labelsize' : 10},
#                                     minor_kwargs={'labelsize' : 6})
#         else:
#             depths = slice_depths(column_top, column_base, slice_length)
#             for i in range(len(depths)):
#                 top_slice, base_slice = depths[i]
#                 col.slice_depth(top = top_slice,
#                                 base = base_slice).plot(
#                                     figsize=figsize,
#                                     major_kwargs = {'labelsize' : 15},
#                                     minor_kwargs={'labelsize' : 10})
#     plt.show()


def img_features(img):
    """retruns mean and std of img per channel ignoring 0 values (background)

    Args:
        img (np.ndarray): image array

    Returns:
        avgs list, means list: lists of means and stds
    """
    features = []
    for ch in range(3):
        pixels = img[:,:,ch].flatten()
        pixels = pixels[pixels!=0]
        if len(pixels) == 0:
            avg = np.nan
            std = np.nan
        else:
            avg = np.average(pixels)/255.0
            std = np.std(pixels)/255.0#255.0
        features.append(avg)
        features.append(std)
    return features


def column_features(column, slice_length=0.01, color_scheme = 'rgb'):
    print('Processing column: {}'.format(column_path.split(os.sep)[-1]))
    col_features=[]
    column_top = column.top
    column_base = column.base
    slices = slice_depths(column_top, column_base, slice_length)
    for i in range(len(slices)):
        top, base = slices[i]
        img = col.slice_depth(top = top, base = base).img            
        features = img_features(img)
        if color_scheme == 'hls':                
            features = colorsys.rgb_to_hls(*color)
        col_features.append(features)
    return np.array(col_features)



directory = 'output\\article'
column_paths = glob.glob(directory + '/*/*.pkl')
print(len(column_paths), 'colomns detected')

# DELETE COLLAPSED COLUMNS
# collapse_columns = []
# for col_idx, column_path in enumerate(column_paths):
#     with open(column_path, 'rb') as f:
#         col = pickle.load(f)
#         if col.add_mode == 'collapse':
#             collapse_columns.append(column_path)

# print(len(collapse_columns), 'collapsed columns')
# for column_path in collapse_columns:
#     os.remove(column_path)

#%%



step = 0.05 #0.1524

for col_idx, column_path in enumerate(column_paths):
    if col_idx == 1:
        break
    
    with open(column_path, 'rb') as f:
        col = pickle.load(f)
        print(col_idx, col, col.add_mode)
        img = col.img
        img_depths = col.depths
        column_top = col.top
        column_base = col.base
        column_length = column_base - column_top
        print('column path:', column_path, 'Column length:', column_length)

        features = column_features(col, slice_length=step, color_scheme='rgb')
        n_steps = int(np.ceil((column_base-column_top)/step))
        depths = np.linspace(column_top, column_base, n_steps)
        print('Features shape:',features.shape,'Depth shape:', depths.shape)

        # create two columns figure
        figure_length = int(column_length)*8
        figsize = (10, figure_length)
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize = figsize)
        axs[0].imshow(img)
        axs[1].plot(features[:,0], depths, label='red', color='red')
        axs[1].plot(features[:,1], depths, label='red_std', color='lightcoral')
        axs[1].plot(features[:,2], depths, label='green', color='green')
        axs[1].plot(features[:,3], depths, label='green_std', color='lightgreen')
        axs[1].plot(features[:,4], depths, label='blue', color='blue')
        axs[1].plot(features[:,5], depths, label='blue_std', color='lightblue')
        axs[1].set_ylim(column_base, column_top)
        plt.grid()
        plt.show()
# %%


directory = r'C:\Users\evgen\Documents\coremdlr\Q204_data\train_data_figshare'
wells = [
    '204-19-3A',
    '204-19-6',
    '204-19-7',
    '204-20-1Z',
    '204-20-1',
    '204-20-2',
    '204-20-3',
    '204-20-6a',
    '204-20a-7',
    '204-24a-6',
    '204-24a-7',
    '205-21b-3',
]

labels_files = [os.path.join(directory, well + '_labels.npy') for well in wells]
image_files = [os.path.join(directory, well + '_image.npy') for well in wells]
depth_files = [os.path.join(directory, well + '_depth.npy') for well in wells]

for i in range(len(image_files)):
    image = np.load(image_files[i])
    labels = np.load(labels_files[i])
    depth = np.load(depth_files[i])
    print(wells[i], image.shape, labels.shape, depth.shape)

# %%

image = np.load(image_files[0])
labels = np.load(labels_files[0])

print(image.shape, labels.shape)

# print unique labels
unique_labels = np.unique(labels)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))

# %%
# calculate statistics for each z position of image
def statistics(image):
    stats = []
    for z in range(image.shape[0]):        
        img_slice = image[z,:,:]
        slice_features = []
        for ch in range(3):
            pixels = img_slice[:,ch].flatten()
            pixels = pixels[pixels!=0]
            if len(pixels) == 0:
                avg = np.nan
                std = np.nan
            else:
                avg = np.average(pixels)/255.0
                std = np.std(pixels)/255.0
            slice_features.append(avg)
            slice_features.append(std)
        stats.append(slice_features)
    arr = np.array(stats)
    return arr

# stats = statistics(image)
# print(stats.shape)
# %%
test_indices  = [2,5,8]
train_indices = [0,1,3,4,6,7,9,10,11]

train_labels_files = [labels_files[i] for i in train_indices]
train_images_files = [image_files[i] for i in train_indices]

test_labels_files = [labels_files[i] for i in test_indices]
test_images_files = [image_files[i] for i in test_indices]


X_train = np.vstack([statistics(np.load(f)) for f in train_images_files])
X_test = np.vstack([statistics(np.load(f)) for f in test_images_files])

y_train=np.hstack([label_encoder.transform(np.load(f)) for f in train_labels_files])
y_test = np.hstack([label_encoder.transform(np.load(f)) for f in test_labels_files])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#%%

# get nan indices in train
nan_indices_train = np.where(np.isnan(X_train))
X_train = np.delete(X_train, nan_indices_train, axis=0)
y_train = np.delete(y_train, nan_indices_train, axis=0)

# do the same for test set
nan_indices_test = np.where(np.isnan(X_test))
X_test = np.delete(X_test, nan_indices_test, axis=0)
y_test = np.delete(y_test, nan_indices_test, axis=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#%%
# import dummy classifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#%%
print('Dummy classifier...')
dc = DummyClassifier(strategy='most_frequent')
y_pred = dc.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print('Logistic regression...')
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10)
y_pred = lr.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print('Random forest...')
rf=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
y_pred = rf.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# %%
import lightgbm as lgb

# lgb multiclass classifier with default parameters
lgb_clf = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, class_weight='balanced')
lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# %%

# import accuracy matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
# show confusion matrix as percentage 
def show_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, xticklabels=classes, fmt='d' ,yticklabels=classes, cmap='Blues')
    # rotate yticklabels horizontally
    plt.yticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

show_confusion_matrix(y_test, y_pred, classes=label_encoder.classes_)


# %%
# plot y_pred, y_test
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
# plot as density plot
plt.hist(y_pred, bins=len(label_encoder.classes_), density=True, label='Predicted', alpha=0.5)
plt.hist(y_test, bins=len(label_encoder.classes_), density=True, label='True', alpha=0.5)
plt.legend()
plt.show()

# %%
