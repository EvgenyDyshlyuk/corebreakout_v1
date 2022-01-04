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


def read_and_check(path, depth_csv, add_tol, box_tol):
    """
    Check that the path and depth_csv are valid.
    """
    # Check that path exists
    folder = path.split(os.sep)[-1]
    #print('Reading from folder:', folder)   

    if not os.path.exists(path):
        raise AssertionError(f'{path} does not exist.')

    # Check that depth_csv exists
    if not os.path.exists(path + os.sep + depth_csv):
        raise AssertionError(f'{folder} does not contain {depth_csv}.')

    # Get img file paths
    img_paths = sorted(glob(os.path.join(path,'*.jpeg'))+
                        glob(os.path.join(path,'*.jpg'))+
                        glob(os.path.join(path,'*.jp2')))
    img_names_folder = [p.split(os.sep)[-1] for p in img_paths]
    
    if len(img_paths) == 0:
        raise AssertionError(f'No images found in {folder}.')

    # read depth_csv
    depth_csv_path = path + os.sep + depth_csv
    depths_df = pd.read_csv(depth_csv_path, index_col=0)
    tops = depths_df.top.values.astype(float)
    bottoms = depths_df.bottom.values.astype(float)
    img_names_csv = depths_df.index.values
    
    # Check that tops and bottoms are the same length in csv
    if not len(tops) == len(bottoms):
        raise AssertionError(f'{folder} len(tops) does not match len(bottoms) in {depth_csv}.')

    # Check that tops/bottoms are the same length as imgs in csv
    if not len(tops) == len(img_names_csv):
        raise AssertionError(f'{folder} len(tops/bottoms) does not match len(image names) in {depth_csv}.')

    # Sanity check: csv vs imgs sets
    if not set(img_names_folder) == set(img_names_csv): 
        print(img_names_folder, img_names_csv)       
        raise AssertionError(f'{folder}, Image sets in csv and folder does not match  : {set(img_names_folder)-set(img_names_csv)}')
   
    return img_paths, tops, bottoms


def segment(
        dir = r'output\intersection_wells_subset',             
        depth_csv = 'depths.csv',
        add_tol = 5000.0,
        box_tol = 6.0,  
        add_mode = 'fill',
        save_dir = None,
        save_name = None,
        save_mode = 'pickle',
        log_file = r'output\log.csv',
        ):

    # get immediate directories in dir
    wells = [x for x in os.listdir(dir) if os.path.isdir(dir + os.sep + x)]      

    # Initialize log file if not already present
    if not os.path.exists(log_file):
        log_df = pd.DataFrame(columns=['Well name', 'error', 'column_path'])
    else:
        log_df = pd.read_csv(log_file)
    
    # Drop rows with non-empty error or column_path
    log_df = log_df.dropna(subset=['error', 'column_path'], thresh=1)
    skip_wells = log_df['Well name'].values
    skip_wells = [x.replace('/', '_').replace(' ', '_') for x in skip_wells]
    if skip_wells:
        print('Skipping', len(skip_wells),'out of total of', len(wells) ,'wells because of previous errors or existing columns paths:')

    wells_to_process = [x for x in wells if x not in skip_wells]
    wells_to_process = list(set(wells_to_process))

    
    if wells_to_process:
        print('Processing', len(wells_to_process), 'wells...')
        subfolders_paths = [dir + os.sep + x for x in wells_to_process]

        # Initialize segmenter
        segmenter = CoreSegmenter(
            model_dir = model_dir,
            weights_path = weights_path,
            model_config=model_config,
            class_names=class_names,
            layout_params=layout_params
        )
        print()

        ct_good=0
        ct_error=0
        for idx, path in enumerate(subfolders_paths):
            print(idx, path)
            # if idx == 2:
            #     break

            well_name = path.split(os.sep)[-1]
            run_img_paths, tops, bottoms = read_and_check(path, depth_csv, add_tol, box_tol)

            try:
                segment = lambda f, t, b : segmenter.segment(f, [t, b], add_tol=add_tol, add_mode=add_mode)
                cols = [segment(f, t, b) for f, t, b in zip(run_img_paths, tops, bottoms)]
                full_column = reduce(add, cols)
                print(f'Created CoreColumn with depth_range={full_column.depth_range}')
                # Save the CoreColumn
                save_dir = save_dir or path
                print(f'Saving CoreColumn to {save_dir} in mode {save_mode}')
                if save_mode is 'pickle':
                    full_column.save(save_dir, name=save_name, pickle=True)
                else:
                    full_column.save(save_dir, name=save_name, pickle=False, image=True, depths=True)
                
                # append to csv log file
                header = False  if os.path.isfile(log_file) else True
                log_df = pd.DataFrame(data = [[well_name, np.nan, save_dir + os.sep + full_column.depth_range]], columns=['Well name', 'error', 'column_path'])
                log_df.to_csv(log_file, mode='a', header=False, index=False)
                ct_good += 1

            except Exception as e:
                print(e)
                print()
                log_df = pd.DataFrame(data = [[well_name, e, np.nan]], columns=['Well name', 'error', 'column_path'])
                header = False  if os.path.isfile(log_file) else True
                log_df.to_csv(log_file, mode='a', header=header, index=False)
                ct_error += 1
            
        print()
        print(ct_good, 'wells sucsessfully processed')
        print(ct_error, 'wells failed')
    print('Done.')


if __name__ == '__main__':
    segment()


# %%
