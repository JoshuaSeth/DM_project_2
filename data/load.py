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
pd.set_option('io.hdf.default_format','table')
prefix = os.path.dirname(os.path.abspath(__file__))

def load_data(test=False, add_day_parts=False, fts_operations=[], add_seasons=False, num_rows=None, caching=True):
    '''Load the dataset (or cached version) with several params as feature engineering.\n
    Params:
    - test: Whether to return train or test data, by default returns train.
    - add_day_parts: Whether to create one-hot columns whether the event is in a certain part of day. Default is False.
    - fts_operations: What operationns to apply to what features to create new features. A list of 3-tuples. For example if you want to divide price by distance and get the difference between visitor_starrating and prop_starrrating you do [('price_usd', 'orig_destination_distance', 'div'), ('prop_starrating', 'visitor_hist_starrating')]. So the first two items of the tuple are column names and the third item of the tuple is the applied operation. Default empty list. This operation is applied last so you can also divide by season, daypart or other intermediary generated features. You can provide the string 'all' to any of the positions in the tuple. When doing this for the first or second position it means it will apply this operation to all the columns. When doing this for the third position it means it will apply all possible operations to the two given columns. For example: [('price_usd', 'orig_destination_distance', 'all')] will create three new columns in the df for all 3 operations. [('price_usd', 'all', 'div')] will divide the price by all other columns in the df. Theoretically you could be extremely extensive and provide all operations to all columns by [('all', 'all', 'all')] which will add 12000 columns.
    Currently supported operations:
        - 'diff': absolute difference
        - 'div': division
        - 'mult': multiplication
    - add_seasons: Whether to add in which season a date belongs. The seasons are four onne-hot columns named season_0, season_1, etc. representing winter, spring, etc.
    - num_rows: Number of rows to read from the DF, by default the full DF is returned. Useful for loading smaller pieces for testing etc.
    - caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Main reason for setting it to false is that each cached df can easily be 4GB. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.'''
    # If we have a cache for a function with these args load it else generate it, save it and return it
    # Cache files are based on the specific argument and the number of rows
    cache_name = str((test, add_day_parts, hash(tuple(fts_operations)), add_seasons))
    cache_path = prefix + '/caches/' + cache_name+str(num_rows)+'.h5'
    daypart_cachename = prefix+'/caches/dayparts'+str(num_rows)+'.h5'
    seasons_cachename = prefix+'/caches/seasons'+str(num_rows)+'.h5'
    ft_engineer_cachename = prefix+'/caches/'+ str(hash(tuple(fts_operations)))+str(num_rows)+'.h5'

    if (path.isfile(cache_path)):
        print('DF with these specific params is cached, returning cached version.')
        return pd.read_hdf(cache_path, start=0, stop=num_rows)
    
    # In case we do not have a cache we manually have to apply the operations
    else:
        # Load data
        print('No cache for this specific request, start loading base df from disk.')
        base_path = prefix + '/test_set_VU_DM.csv' if test else prefix + '/training_set_VU_DM.csv'
        try:
            df = pd.read_csv(base_path, nrows=num_rows)
        except Exception as e: 
            print('Couldnt find the data file, can you move the training_set_VU_DM.csv to', base_path, '? This file is to large to share via github. Error:\n', e) 
            return

        # Always translate string to datetime for datetime column no param needed
        df['date_time'] = pd.to_datetime(df['date_time'])

        # Create one-hot columns for the different dayparts
        if add_day_parts:
            # Check if dayparts in cache
            if(path.isfile(daypart_cachename)):
                print('Using cache for the dayparts suboperation')
                dayparts_csv = pd.read_hdf(daypart_cachename)
                df = pd.concat([df, dayparts_csv.set_index(df.index)], axis=1)
                del dayparts_csv
            
            else:  # Else generate them
                dayparts = ['early_morning', 'morning', 'noon', 'eve', 'night', 'late_night']
                for index, daypart in enumerate(tqdm(dayparts)):
                    print(index+1, "/", len(dayparts), 'Generating datetimes for', daypart)
                    df[daypart] = (df['date_time'].progress_apply(get_daypart) == daypart).astype(int)
                if caching: df[dayparts].to_hdf(daypart_cachename, key='df')
                 
        
        # Add one-hot seasons for the dates
        if add_seasons:
            if(path.isfile(seasons_cachename)): #Again check if we have a cached version of this operation
                print('Using cache for the seasons suboperation')
                seasons_csv = pd.read_hdf(seasons_cachename)
                df = pd.concat([df, seasons_csv.set_index(df.index)], axis=1)
                del seasons_csv
            else:
                print('generating seasons')
                seasons =  df['date_time'].progress_apply(get_season) # Get season as number 0-3
                seasons = pd.get_dummies(seasons, prefix='season') # To one-hot
                print('adding seasons to DF and saving them to cache')
                df = pd.concat([df, seasons.set_index(df.index)], axis=1)
                if caching: seasons.to_hdf(seasons_cachename, key='df')
                del seasons
        
        # Create new features by dividing columns by each other
        if fts_operations != []:
            if(path.isfile(ft_engineer_cachename)): #Check cache for this specific ft engineering
                print('Using cache for the feature engineering suboperation')
                ft_csv = pd.read_hdf(ft_engineer_cachename)
                df = pd.concat([df, ft_csv.set_index(df.index)], axis=1)
                del ft_csv
            
            else: # No cache, generate these features
                print('Feature engineering columns with operations')
                ft_eng_df = pd.DataFrame()                
                for col_1, col_2, op_str in tqdm(fts_operations):
                    engineer_ft(df, ft_eng_df, col_1, col_2, op_str)
                df = pd.concat([df, ft_eng_df.set_index(df.index)], axis=1)
                if caching: ft_eng_df.to_hdf(ft_engineer_cachename, key='df')
                del ft_eng_df
                
            
        if caching: 
            print('saving result of specific arguments to cache')
            # Save the resulting df to a cache in 500 chunks (so you can see progress since this takes long)            
            ixs = np.array_split(df.index, 500)
            for ix, subset in enumerate(tqdm(ixs)):
                if ix == 0:
                    df.loc[subset].to_hdf(cache_path, mode='w', index=True, key='df')
                else:
                    df.loc[subset].to_hdf(cache_path, append=True, mode='a', index=True, key='df')

        return df

def engineer_ft(df, ft_eng_df, col_1, col_2, op_str):
    '''Recursively add features to the ft_eng_df. For example (df, pd.Dataframe(), 'price_usd', 'all', 'div'). Meant to be used internally in the load_data function. Changes ft_eng_df inplace.'''
    # Do it recursively if getting an 'all' argument
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Only numeric columns
    if col_1 == 'all':
        for col_name in tqdm(numeric_cols):
            engineer_ft(df, ft_eng_df, col_name, col_2, op_str)
    elif col_2 == 'all':
        for col_name in tqdm(numeric_cols):
            engineer_ft(df, ft_eng_df, col_1, col_name, op_str)
    elif op_str == 'all':
        for operator in tqdm(['diff', 'div', 'mult']):
            engineer_ft(df, ft_eng_df, col_1, col_2, operator)
    
    # If this was no recursion (or the bottom of the recursion)
    if 'all' not in [col_1, col_2, op_str]:
        new_name = col_1 +'_'+op_str+'_' + col_2
        if op_str == 'div':
            ft_eng_df[new_name] = df[col_1]/df[col_2]
        if op_str == 'mult':
            ft_eng_df[new_name] = df[col_1]*df[col_2]
        if op_str == 'div':
            ft_eng_df[new_name] = abs(df[col_1]-df[col_2])

def div(x,y):
    return x/y       

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