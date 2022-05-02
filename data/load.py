'''File for the load_data function.'''
import os
from itertools import product
import pandas as pd
from tqdm import tqdm
import numpy as np
import os.path
from os import path

# Some global initialization
tqdm.pandas()
prefix = os.path.dirname(os.path.abspath(__file__))

def load_data(test=False, add_day_parts=False, divided_fts=[], caching=True):
    '''Load the dataset (or cached version) with several params as feature engineering.\n
    Params:
    - test: Whether to return train or test data, by default returns train.
    - add_day_parts: Whether to create one-hot columns whether the event is in a certain part of day. Default is False.
    - divided_fts: What features to divide by each other to create new features, a list of column names, an empty list or 'all' for all column names. Default empty list.
    - caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Default is true.'''
    # If we have a cache for a function with these args load it else generate it, save it and return it
    cache_name = str((test, add_day_parts, hash(tuple(divided_fts))))
    cache_path = prefix + "/caches/" + cache_name
    if (path.isfile(cache_path)):
        print('DF with params is cached, returning cache')
        return pd.read_csv(cache_path)
    
    # In case we do not have a cache we manually have to apply the operations
    else:
        # Load data
        base_path = prefix + '/test_set_VU_DM.csv' if test else prefix + '/training_set_VU_DM.csv'
        df = pd.read_csv(base_path)

        # Always translate string to datetime for datetime column no param needed
        df['date_time'] = pd.to_datetime(df['date_time'])

        # Create one-hot columns for the different dayparts
        if add_day_parts:
            # Check if dayparts in cache
            daypart_cachename = prefix+'/caches/dayparts.csv'
            if(path.isfile(daypart_cachename)):
                df = pd.read_csv(daypart_cachename)
            
            else:  # Else generate them
                dayparts = ['early_morning', 'morning', 'noon', 'eve', 'night', 'late_night']
                for index, daypart in enumerate(tqdm(dayparts)):
                    print(index+1, "/", len(dayparts), 'Generating datetimes for', daypart)
                    df[daypart] = (df['date_time'].progress_apply(get_daypart) == daypart).astype(int)
                df.to_csv(daypart_cachename)
        
        # Create new features by dividing columns by each other
        print('Dividing columns by each other to create new columns')
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Only numeric columns
        div_fts = numeric_cols if divided_fts == 'all' else divided_fts
        for ft_1, ft_2 in tqdm(list(product(div_fts, div_fts))): # Products gives all combinations of 2 lists
            df[str(ft_1) +"/"+str(ft_2)] = df[ft_1]/df[ft_2]
        
        # Generate a string of somewhat reasonable length to cache this result under
        if(caching): df.to_csv(cache_path) # Save cache for next time function with same args is called

        return df
        



def get_daypart(h):
    '''Translate daytime to daypart'''
    x = h.hour
    if (x > 4) and (x <= 8):
        return 'early_morning'
    elif (x > 8) and (x <= 12):
        return 'morning'
    elif (x > 12) and (x <= 16):
        return'noon'
    elif (x > 16) and (x <= 20):
        return 'eve'
    elif (x > 20) and (x <= 24):
        return'night'
    elif (x <= 4):
        return'late_night'