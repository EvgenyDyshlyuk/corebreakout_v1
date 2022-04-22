#%%
"""
Script for processing batch of raw images in a directory into saved `CoreColumn`s.

The `path` given should contain images as jpeg files, and a `depth_csv`.csv file in the format:

```
           ,    top,    bottom
<filename1>, <top1>, <bottom1>
...
<filenameN>, <topN>, <bottomN>
```

NOTE: model `Config`, `class_names`, and segmentation `layout_params` can only be
changed manually at the top of script, and default to those configured in `corebreakout/defaults.py`

Run with --help argument to see full options.
"""
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from operator import add

from corebreakout import defaults
from corebreakout import CoreSegmenter, CoreColumn


# Change Config selection manually
model_config = defaults.DefaultConfig()
# Change class_names manually
class_names = defaults.CLASSES
# Change any non-default layout_params manually
layout_params = defaults.LAYOUT_PARAMS
# load model from directory
model_dir = defaults.MODEL_DIR
# Load model weights from directory
weights_path = defaults.CB_MODEL_PATH


def read_and_check(directory: str, depth_csv_name: str) -> pd.DataFrame:
    """
    Check that the path and depth_csv are valid.
    """    
    # Check that directory exists
    if not os.path.isdir(directory):
        raise AssertionError(f'{directory} does not exist.')

    # Check that depth_csv exists
    depth_csv_path = directory + os.sep + depth_csv_name
    if not os.path.exists(depth_csv_path):
        raise AssertionError(f'{directory} does not contain {depth_csv_name}')
    
    # Get all image file paths in directory, check if images exist in directory
    img_paths_directory = sorted(glob(os.path.join(directory,'*.jpeg'))+
                                 glob(os.path.join(directory,'*.jpg'))+
                                 glob(os.path.join(directory,'*.jp2')))
    img_names_directory = [p.split(os.sep)[-1] for p in img_paths_directory]    
    if len(img_paths_directory) == 0:
        raise AssertionError(f'No images found in {directory}.')
    
    # Sanity check img_names csv is a subset of img_names_directory
    df = pd.read_csv(depth_csv_path)
    df.dropna(subset=['image_name'], inplace=True)
    img_names_csv = df['image_name'].values.astype(str)   
    if not set(img_names_csv).issubset(set(img_names_directory)):
        raise AssertionError( f'Image sets in csv {set(img_names_csv)} not a subset of image files in directory: {set(img_names_directory)}')
    
    return df

def split_df(df: pd.DataFrame(), split_col: str, split_val: float) -> list:
    """
    Split a dataframe into several dataframes based on a split_col if split_val is exceeded.

    Args:
    ----------
    df : pd.DataFrame
        Dataframe to split.
    split_col : str
        Column name to split on.
    split_val : float
        Value to split on.

    Returns:
    -------
    list of splitted pd.DataFrames
    """
    index = df.index[df['gap'] > 10]+1
    arr_list = np.split(df, index)
    df_list = [pd.DataFrame(x) for x in arr_list]
    if len(df_list) > 1:
        print('\t splitted column to', len(df_list), 'columns based on gap_tol')
    return df_list

def filter_df(df: pd.DataFrame()) -> tuple:
    """
    Filter a dataframe to remove rows based on several criteria:
    - nan in image_name
    - image_size < 9 MB
    - duplicated = True
    - gap < 0
    - box_length > 3.0 m 
    """    
    l = len(df)
    # drop rows with nan values in image_name column    
    df = df.dropna(subset=['image_name'])
    # drop rows with image_size < 9 MB
    df = df[df['image_size'] > 9000000]
    # drop duplicated rows
    df = df[df['duplicated'] != True]
    df.reset_index(drop=True, inplace=True)
    # drop row and next row if gap is < 10 (error in either of them)
    neg_gaps_index = df[df['gap'] < 0].index.tolist()
    next_gaps_index = [i+1 for i in neg_gaps_index]
    all_ind = neg_gaps_index + next_gaps_index
    df.drop(all_ind, inplace=True)
    # drop rows if box_length > 3.0 m
    df = df[df['box_length'] <= 3.0]
    n_dropped = l - len(df)
    if n_dropped > 0:
        print('\t', n_dropped, 'images removed from total of', l, 'images before segmenting')
    if df.shape[0] > 0:
        img_names = df['image_name'].values.astype(str).tolist()
        tops = df['top'].values.astype(float).tolist()
        bottoms = df['bottom'].values.astype(float).tolist()
        return img_names, tops, bottoms
    else:
        return [None], [None], [None]


def segment_images(wells_subset,
                   directory = 'output\\intersection_wells_subset',        
                   depth_csv = 'metadata.csv',
                   add_tol = 5000.0,
                   box_tol = 6.0,  
                   add_mode = 'fill',
                   save_dir = None,
                   save_name = None,
                   save_mode = 'pickle',
                   log_file = 'output\\image_segmenter_log.csv',
                   gap_tol = 50.0):
    
    # placeholder for segmenter objects
    segmenter = None

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('well_name,image_name,top,bottom,error_msg')
            f.write('\n')
            print(f'Saved a new log file: {log_file}') 
    log_df = pd.read_csv(log_file, index_col=False)


    # Get wells from well_directories only if they are in wells_subset
    well_directories = [x for x in os.listdir(directory) if os.path.isdir(directory + os.sep + x)]
    wells_subset = [x.replace('/', '_').replace(' ', '_') for x in wells_subset]
    wells_to_process = [x for x in well_directories if x in wells_subset]
    wells_to_process.sort()
   
    if wells_to_process:
        print('PROCESSING', len(wells_to_process), 'WELLS...')      
        
        for well_idx, well_name in enumerate(wells_to_process):            
            print(well_idx, 'processing:', well_name, '...')
            subfolder_path = directory + os.sep + well_name

            df=read_and_check(subfolder_path, depth_csv)

            df_list = split_df(df, 'gap', gap_tol)

            segmenter_params_list = [filter_df(x) for x in df_list]
            
            # Iterate over each segmenter parameters obtained from the csv file
            for segmenter_params in segmenter_params_list:                    
                run_img_names, tops, bottoms = segmenter_params
                skip_column = False

                # If there are no images to process, skip column
                if run_img_names == [None]:
                    error_msg = f'{well_name}, skipping segmentation, no imgs after filtering'
                    print('\t', error_msg)
                    skip_column = True
                
                # If corresponding pkl column already exists in folder -> skip
                if not skip_column:                    
                    column_paths = glob(subfolder_path+'/*.pkl')
                    #print(column_paths)
                    for col_path in column_paths:
                        col_name = col_path.split(os.sep)[-1].rstrip('.pkl')
                        col_top = float(col_name.split('_')[1])
                        col_bottom = float(col_name.split('_')[2])
                        if col_top>=min(tops) and col_bottom<=max(bottoms):
                            error_msg = f'Skipping segmentation, {col_name} already exists'
                            print('\t', error_msg)
                            skip_column = True                                        
                
                if not skip_column:                                                                           
                    run_img_paths = [subfolder_path + os.sep + x for x in run_img_names]
                    #
                    cols = []
                    for f, t, b in zip(run_img_paths, tops, bottoms):
                        image_name = f.split(os.sep)[-1]

                        # Check log file and if segmentation failed previously -> skip                        
                        if log_df[ (log_df['well_name']==well_name) & (log_df['image_name']==image_name) ].empty:
                            # if segmenter not instantiated yet, instantiate it
                            if segmenter is None:
                                segmenter = CoreSegmenter(model_dir = model_dir,
                                                          weights_path = weights_path,
                                                          model_config=model_config,
                                                          class_names=class_names,
                                                          layout_params=layout_params)

                                #segment = lambda f, t, b : segmenter.segment(f, [t, b], add_tol=add_tol, add_mode=add_mode)
                            # segment each image
                            try:
                                col_segment = segmenter.segment(f, [t, b], add_tol=add_tol, add_mode=add_mode)
                            except Exception as e:                            
                                error_msg = f'{well_name},{image_name},{t},{b},{e}'
                                print('\t', error_msg)
                                with open(log_file, 'a') as f: #write to log file
                                    f.write(error_msg)
                                    f.write('\n')
                                log_df.append({'well_name': well_name, # append to log_df
                                               'image_name': image_name,
                                               'top': t,
                                               'bottom': b,
                                               'error_msg': e},
                                               ignore_index=True)
                            else:
                                cols.append(col_segment)
                    
                    # If no columns after segmentation 
                    if not cols:
                        error_msg = f'No images after segmentation for this column'
                        print('\t', error_msg)

                    # If successful segmentation, save full column to disk
                    else:
                        full_column = reduce(add, cols)
                        save_dir_to = save_dir or subfolder_path                        
                        print(f'\t Saving CoreColumn to {save_dir_to} in {save_mode} mode...')
                        if save_mode is 'pickle':
                            full_column.save(save_dir_to, name=save_name, pickle=True)
                        else:
                            full_column.save(save_dir_to, name=save_name, pickle=False, image=True, depths=True)                  
                        # top, bottom = full_column.depth_range
                        # core_column_name = 'CoreColumn_'+str(round(top,2))+'_'+str(round(bottom,2))
    print('Done.')


if __name__ == '__main__':
    segment()


# %%
