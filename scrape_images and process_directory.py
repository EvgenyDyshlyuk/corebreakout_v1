#%%
%load_ext autoreload
%autoreload 2

import sys
import os

import random

import pandas as pd
pd.set_option('display.max_colwidth', 100)

# append scraper directory to sys.path
scraper_dir = r'C:\Users\evgen\Documents\scraper'
sys.path.append(scraper_dir)
import bgs_scraper

# append scripts directory to sys.path
scripts_dir = os.path.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
import process_directory_direct_run


# Load intersection wells data
intersection_df = pd.read_csv('output/intersection_wells.csv')
intersection_wells = intersection_df['Well name'].unique()
print(len(intersection_wells), 'intersection wells')
# reformat well names to original format
intersection_wells = [x.replace('_', '/', 1).replace('_', ' ', 1) for x in intersection_wells]
intersection_wells = sorted(intersection_wells)


# # take a subset of intersection wells
# random.seed(42)
# intersection_wells_subset = random.sample(intersection_wells, 100)
# intersection_wells_subset = sorted(intersection_wells_subset)
# print('subset:', intersection_wells_subset)


# #bgs_scraper.images_scraper(bgs_all_df, well_names = intersection_wells_subset, save_path=photos_dir)

# # 5 hr, 50GB per 100 wells/30 min and 5GB per 20 wells for scraping


#%%
# CENTIMETER SCALE ARTICLE WELLS
centimeter_scale_article_wells = [
    '204/19- 3A', # Foinaven reservoir
    '204/24a- 7', # Foinaven, not in intersection wells
    
    '204/19- 6', # Alligin
    '204/19- 7', # Alligin
    
    '204/20- 3', # Loyal
    '204/20- 6A', # Loyal
    
    '204/20- 1', # Schiehallion
    '204/20- 1Z', # Schiehallion
    '204/20- 2', # Schiehallion
    '204/20a- 7', # Schiehallion
    '205/21b- 3', # Schiehallion

    '204/24a- 6', # Non reservoir, not in intersection wells
 ]

print('Not in intersection wells:',list(set(centimeter_scale_article_wells) - set(intersection_wells)))

centimeter_scale_dir='output\\cetimeter_scale_article_wells'

bgs_scraper.images_scraper(bgs_all_df, well_names = centimeter_scale_article_wells, save_path=centimeter_scale_dir)

process_directory_direct_run.segment_images(directory = centimeter_scale_dir, wells_subset = centimeter_scale_article_wells, gap_tol=50)

#%%
# INTERSECTION WELLS SUBSET
intersection_wells_dir='output\\intersection_wells_subset'
well_dirs = [x for x in os.listdir(intersection_wells_dir) if os.path.isdir(os.path.join(intersection_wells_dir, x))]
wells = [x.replace('_', '/', 1).replace('_', ' ', 1) for x in well_dirs]

bgs_scraper.images_scraper(bgs_all_df, well_names = wells, save_path=intersection_wells_dir)

process_directory_direct_run.segment_images(directory=intersection_wells_dir, wells_subset = wells, add_mode='fill', gap_tol = 50)

#%%
# DELETE ALL DIRECTORIES EXCEPT THE ONES in GOOD WELLS
# get immediate subdirectories of photos_dir
# import shutil
# all_dirs = [x for x in os.listdir(photos_dir) if os.path.isdir(os.path.join(photos_dir, x))]
# print(len(all_dirs), 'well directories', all_dirs[0])

# good_dirs = [well.replace('/', '_').replace(' ', '_') for well in good_wells]
# print(len(good_dirs), 'good wells', good_dirs[0])

# delete_dirs = [x for x in all_dirs if x not in good_dirs]
# print(len(delete_dirs), 'well directories to delete')

# ct=0
# for directory in delete_dirs:
#     directory_path = os.path.join(photos_dir, directory)
#     if os.path.exists(directory_path):
#         print('deleting', ct, directory_path)
#         ct+=1
#         shutil.

#%%
# get all depth.csv files in intersection wells directory
#import glob
# get all *.pkl files in intersection wells directory and subdirectories
#depth_files = glob.glob(os.path.join(intersection_wells_dir, '*/*/.pkl'))

# delete 
#for depth_file in files:
#    os.remove(depth_file)

#print(len(depth_files), 'depth files')

#%%
# def filter_wells(well_names):
#     good_wells = []
#     ct_image_size = 0
#     # ct_duplicates = 0
#     # ct_gap = 0
#     ct_box_size = 0
#     for well_idx, well in enumerate(well_names):    
#         well_name = well.replace('/', '_').replace(' ', '_')    
#         well_path = os.path.join(photos_dir, well_name)
#         df = pd.read_csv(os.path.join(well_path, 'depths.csv'))
#         error = str()
#         if (df['image_size'] < 9000000).any():
#             error += '\t has files smaller than 9MB => probably bad quality vertical images \n'
#             ct_image_size += 1
#         # if (df['duplicated']==True).any():
#         #     error += '\t has duplicated images \n'
#         #     ct_duplicates += 1
#         # if (df['gap'] < 0).any():
#         #     error += '\t has negative gaps => needs manual check \n'
#         #     ct_gap += 1
#         if (df['box_length']>3.0).any():
#             error += '\t has boxes longer than 3m'
#             ct_box_size += 1
#         if error:
#             pass
#             # print(well_idx, well)
#             # print(error)
#         else:
#             good_wells.append(well)
            
#     print(ct_image_size, 'wells with image size < 9MB')
#     # print(ct_duplicates, 'wells with duplicated images')
#     # print(ct_gap, 'wells with negative gaps')
#     print(ct_box_size, 'wells with boxes longer than 3m')
#     print(len(good_wells), 'good wells')
#     return good_wells

# bgs_all_df = pd.read_csv(scraper_dir+r'\output\BGS_all_filtered.csv')
# print(len(bgs_all_df), 'length of BGS_all_filtered.csv')
# print(len(bgs_all_df['Well name'].unique()), 'unique wells')
# %%
# get files in folder and subfolders
import glob
files = glob.glob(intersection_wells_dir+'/*/*.pkl')
print(len(files), 'files')

# %%

columns = []
for well in good_wells:
    well_name = well.replace('/', '_').replace(' ', '_')
    well_path = os.path.join(photos_dir, well_name)
    #df = pd.read_csv(os.path.join(well_path, 'depths.csv'))
    core_column_paths = glob.glob(well_path+'/*.pkl')
    if len(core_column_paths)>1:
        print(well_name, 'has more than one core column')
        break
    elif len(core_column_paths)==1:
        columns.append(core_column_paths[0])
        #print(well_name)
print(len(columns), 'well columns')
# %%
