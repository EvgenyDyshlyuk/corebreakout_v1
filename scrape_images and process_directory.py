#%%
import sys
import os

import random
random.seed(42)

import pandas as pd
pd.set_option('display.max_colwidth', 100)

import process_directory


# Scraper directory
scraper_dir = r'C:\Users\evgen\Documents\scraper'
# append scraper directory to sys.path
sys.path.append(scraper_dir)
import bgs_scraper

# Load all bgs photos data

# Load intersection wells data
intersection_df = pd.read_csv('output\intersection_wells.csv')
intersection_wells = intersection_df['Well name'].unique()
print(len(intersection_wells), 'intersection wells')
# reformat well names to original format
intersection_wells = [x.replace('_', '/', 1).replace('_', ' ', 1) for x in intersection_wells]
intersection_wells = sorted(intersection_wells)


#%%
%%time

# take a subset of intersection wells
intersection_wells_subset = random.sample(intersection_wells, 20)

all_df_filtered = pd.read_csv(scraper_dir+r'\output\BGS_all_filtered.csv')
print(len(all_df_filtered), 'length of BGS_all_filtered.csv')

save_path=r'output\intersection_wells_subset'
bgs_scraper.images_scraper(all_df_filtered, well_names = intersection_wells_subset, save_path=save_path)

# 5 hr per 100 wells
# 50 GB per 100 wells

# 30 min and 5GB per 20 wells


# %%
from process_directory_direct_run import segment
segment()

# %%
