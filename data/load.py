'''File for the load_data function.'''
import math
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

def load_data(test=False, add_day_parts=False, divided_fts=[], add_seasons=False, caching=True):
    '''Load the dataset (or cached version) with several params as feature engineering.\n
    Params:
    - test: Whether to return train or test data, by default returns train.
    - add_day_parts: Whether to create one-hot columns whether the event is in a certain part of day. Default is False.
    - divided_fts: What features to divide by each other to create new features, a list of column names, an empty list or 'all' for all column names. Default empty list. This operation is applied last so you can also divide by season, daypart or other generated features.
    - add_seasons: Whether to add in which season a date belongs. The seasons are four onne-hot columns named season_0, season_1, etc. representing winter, spring, etc.
    - caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Main reason for setting it to false is that each cached df can easily be 4GB. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.'''
    # If we have a cache for a function with these args load it else generate it, save it and return it
    cache_name = str((test, add_day_parts, hash(tuple(divided_fts)), add_seasons))
    cache_path = prefix + "/caches/" + cache_name+".csv"
    daypart_cachename = prefix+'/caches/dayparts.csv'
    seasons_cachename = prefix+'/caches/seasons.csv'

    if (path.isfile(cache_path)):
        print('DF with these specific params is cached, returning cached version.')
        return pd.read_csv(cache_path)
    
    # In case we do not have a cache we manually have to apply the operations
    else:
        # Load data
        base_path = prefix + '/test_set_VU_DM.csv' if test else prefix + '/training_set_VU_DM.csv'
        try:
            df = pd.read_csv(base_path)
        except Exception as e: print('Couldnt find the data file, can you move the training_set_VU_DM.csv to', base_path, '? This file is to large to share via github. Error:\n', e)

        # Always translate string to datetime for datetime column no param needed
        df['date_time'] = pd.to_datetime(df['date_time'])

        # Create one-hot columns for the different dayparts
        if add_day_parts:
            # Check if dayparts in cache
            if(path.isfile(daypart_cachename)):
                print('Using cache for the dayparts suboperation')
                dayparts_csv = pd.read_csv(daypart_cachename)
                df = pd.concat([df, dayparts_csv])
                del dayparts_csv
            
            else:  # Else generate them
                dayparts = ['early_morning', 'morning', 'noon', 'eve', 'night', 'late_night']
                for index, daypart in enumerate(tqdm(dayparts)):
                    print(index+1, "/", len(dayparts), 'Generating datetimes for', daypart)
                    df[daypart] = (df['date_time'].progress_apply(get_daypart) == daypart).astype(int)
                df[dayparts].to_csv(daypart_cachename)
                 
        
        # Add one-hot seasons for the dates
        if add_seasons:
            if(path.isfile(seasons_cachename)): #Again check if we have a cached version of this operation
                print('Using cache for the seasons suboperation')
                seasons_csv = pd.read_csv(seasons_cachename)
                df = pd.concat([df, seasons_csv])
                del seasons_csv
            else:
                print('generating seasons')
                seasons =  df['date_time'].progress_apply(get_season) # Get season as number 0-3
                seasons = pd.get_dummies(seasons, prefix='season') # To one-hot
                print('adding seasons to DF and saving them to cache')
                df = pd.concat([df, seasons])
                seasons.to_csv(seasons_cachename)
                del seasons
        
        # Create new features by dividing columns by each other
        if divided_fts != []:
            print('Dividing columns by each other to create new columns')
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Only numeric columns
            div_fts = numeric_cols if divided_fts == 'all' else divided_fts
            for ft_1, ft_2 in tqdm(list(product(div_fts, div_fts))): # Products gives all combinations of 2 lists
                df[str(ft_1) +"/"+str(ft_2)] = df[ft_1]/df[ft_2]
            
        # Generate a string of somewhat reasonable length to cache this result under
        if(caching): 
            print('saving result of specific arguments to cache')
            df.to_csv(cache_path) # Save cache for next time function with same args is called

        return df
        

def get_season(x):
    return x.month%12 // 3 + 1

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