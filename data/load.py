'''File for the load_data function.'''
import math
import os
from itertools import product
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import os.path
from os import path
from warnings import simplefilter


# Some global initialization
tqdm.pandas()
pd.set_option('io.hdf.default_format','table')
prefix = os.path.dirname(os.path.abspath(__file__))

def load_data(test=False, add_day_parts=False, fts_operations=[], same_value_operations=[], add_seasons=False, num_rows=None, scaling=False, fill_nan=None, caching=True):
    '''Load the dataset (or cached version) with several params as feature engineering.\n
    Params:
    - test: Whether to return train or test data, by default returns train.
    - add_day_parts: Whether to create one-hot columns whether the event is in a certain part of day. Default is False.
    - fts_operations: What operationns to apply to what features to create new features. A list of 3-tuples. For example if you want to divide price by distance and get the difference between visitor_starrating and prop_starrrating you do [('price_usd', 'orig_destination_distance', 'div'), ('prop_starrating', 'visitor_hist_starrating', 'diff')]. So the first two items of the tuple are column names and the third item of the tuple is the applied operation. Default empty list. This operation is applied last so you can also divide by season, daypart or other intermediary generated features. You can provide the string 'all' to any of the positions in the tuple. When doing this for the first or second position it means it will apply this operation to all the columns. When doing this for the third position it means it will apply all possible operations to the two given columns. For example: [('price_usd', 'orig_destination_distance', 'all')] will create three new columns in the df for all 3 operations. [('price_usd', 'all', 'div')] will divide the price by all other columns in the df. Theoretically you could be extremely extensive and provide all operations to all columns by [('all', 'all', 'all')] which will add 12000 columns.
    Currently supported operations:
        - 'diff': absolute difference
        - 'div': division
        - 'mult': multiplication
    - same_value_operations: In need of a better name. Operations applied to the set where all values are the same. A list of tuples [()] where the first item of the tuple is the column name where you check for equal values (i.e. 'site_id') and the second item of the tuple is the values for which you want to apply the operation (i.e. 'price_usd') and where the third item of the tuple is the applied operation (i.e. 'avg'). This is the example for location id we were talking about. For example [('site_id', 'price_usd', 'avg')] will give the average price for all rows with this site. 
    Possible values for operations are:
        - 'avg': average/mean
    - add_seasons: Whether to add in which season a date belongs. The seasons are four onne-hot columns named season_0, season_1, etc. representing winter, spring, etc.
    - num_rows: Number of rows to read from the DF, by default the full DF is returned. Useful for loading smaller pieces for testing etc.
    - scaling: Whether to MinMax scale the data. Added this to save some boilerplate code and so that this intesive operation is now also cached since it takes longer than loading the data itself. Default false
    - fill_nan: operation to call on the df to fill nans. Default None so nan's not filled. For example: 'mean' will do df.fillna(df.mean())
    - caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Main reason for setting it to false is that each cached df can easily be 4GB. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.'''
    # If we have a cache for a function with these args load it else generate it, save it and return it
    # Cache files are based on the specific argument and the number of rows
    cache_name = str((test, add_day_parts, hash(tuple(fts_operations)),hash(tuple(same_value_operations)), add_seasons, scaling, fill_nan))
    cache_path = prefix + '/caches/' + cache_name+str(num_rows)+'.h5'
    daypart_cachename = prefix+'/caches/dayparts'+str((test,num_rows))+'.h5'
    seasons_cachename = prefix+'/caches/seasons'+str((test,num_rows))+'.h5'
    sameid_cachename = prefix+'/caches/'+ str(hash(tuple(same_value_operations)))+str((test,num_rows))+'.h5'
    ft_engineer_cachename = prefix+'/caches/'+ str(hash(tuple(fts_operations)))+str((test,num_rows))+'.h5'

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

        # 1. Create one-hot columns for the different dayparts
        if add_day_parts:
            df, has_cache = try_add_cached(daypart_cachename, df)
            if not has_cache:  # Else generate them
                print('generating dayparts')
                dayparts = ['early_morning', 'morning', 'noon', 'eve', 'night', 'late_night']
                for index, daypart in enumerate(tqdm(dayparts)):
                    print(index+1, "/", len(dayparts), 'Generating datetimes for', daypart)
                    df[daypart] = (df['date_time'].progress_apply(get_daypart) == daypart).astype(int)
                if caching: df[dayparts].to_hdf(daypart_cachename, key='df')
                 
        
        # 2. Add one-hot seasons for the dates
        if add_seasons:
            df, has_cache = try_add_cached(seasons_cachename, df)
            if not has_cache:
                print('generating seasons')
                seasons =  df['date_time'].progress_apply(get_season) # Get season as number 0-3
                seasons = pd.get_dummies(seasons, prefix='season') # To one-hot
                df = pd.concat([df, seasons.set_index(df.index)], axis=1)
                if caching: seasons.to_hdf(seasons_cachename, key='df')
                del seasons
            
        # 3. Adding averages or stds for certain columns where for example id is the same
        if same_value_operations !=[]:
            df, has_cache = try_add_cached(sameid_cachename, df)
            if not has_cache:
                print('generating same id features')
                same_vals_df = pd.DataFrame()
                for col_1, col_2, op_str in tqdm(same_value_operations): # DONT FORGET TO CHANGE TO OPERATION, SUM MUST BE SOMETHING LIKE AGG
                    _dict = df.groupby(col_1)[col_2].mean().to_dict() # A dict with the operation applied to each group of the id 
                    same_vals_df['same_val_'+col_1+'_'+col_2+'_'+op_str] = df[col_1].progress_apply(get_from_dict, _dict=_dict)
                df = pd.concat([df, same_vals_df.set_index(df.index)], axis=1)
                if caching: same_vals_df.to_hdf(sameid_cachename, key='df')
                del same_vals_df
        
        # 4. Create new features by dividing columns by each other
        if fts_operations != []:
            df, has_cache = try_add_cached(ft_engineer_cachename, df)
            if not has_cache: # No cache, generate these features
                print('Feature engineering columns with operations')
                ft_eng_df = pd.DataFrame()                
                for col_1, col_2, op_str in fts_operations:
                    engineer_ft(df, ft_eng_df, col_1, col_2, op_str)
                df = pd.concat([df, ft_eng_df.set_index(df.index)], axis=1)
                if caching: 
                    if len(ft_eng_df.columns)<1400:
                        ft_eng_df.to_hdf(ft_engineer_cachename, key='df')
                    else: 
                        print('To many columns (1400+) to cache as hdf. Not caching.')
                del ft_eng_df

        if fill_nan: # Fill Nans, I want this in the load data function so this expesnive operation is cached
            # I don't think there'll be any NAN id's
            df.fillna(eval('df.'+fill_nan+'()'))
        
        if scaling: # This also scales the id's which I don't think is a problem since they'll go from unique to unique
            # We can't cache the scaling operation since cachinng it's result is the same as cache the results of all operationns together. This would be double. So it is cached in the end result, not intermediate.
            scaler = MinMaxScaler()
            dt = df['date_time'] # Rm datetime
            df = scaler.fit_transform(df.drop('date_time', axis=1))
            df['date_time'] = dt #Readd datetime
                
        # Save the resulting df to a cache in 500 chunks (so you can see progress since this takes long)
        if caching: 
            print('saving result of specific arguments to cache')
            if len(df.columns)<1400:
                ixs = np.array_split(df.index, 500)
                for ix, subset in enumerate(tqdm(ixs)):
                    if ix == 0:
                        df.loc[subset].to_hdf(cache_path, mode='w', index=True, key='df')
                    else:
                        df.loc[subset].to_hdf(cache_path, append=True, mode='a', index=True, key='df')
            else: print('To many columns to cache as HDF. Not caching end result.')

        return df

def get_from_dict(x, _dict):
    '''There might be a better way to do this'''
    return _dict[x]

def try_add_cached(cache_path, df):
    '''Will load and concat the cache to the df is there is a cache'''
    if(path.isfile(cache_path)): #Check cache for this specific ft engineering
        print('Using cache:', cache_path)
        loaded_cache = pd.read_hdf(cache_path)
        df = pd.concat([df, loaded_cache.set_index(df.index)], axis=1)
        del loaded_cache
        return df, True
    return df, False

def engineer_ft(df, ft_eng_df, col_1, col_2, op_str):
    '''Recursively add features to the ft_eng_df. For example (df, pd.Dataframe(), 'price_usd', 'all', 'div'). Meant to be used internally in the load_data function. Changes ft_eng_df inplace.'''
    # Do it recursively if getting an 'all' argument
    if 'all' in [col_1, col_2, op_str]:
        # Filter this warning cause you'll get a thousand
        simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Only numeric columns
        if col_1 == 'all':
            for col_name in tqdm(numeric_cols):
                engineer_ft(df, ft_eng_df, col_name, col_2, op_str)
        elif col_2 == 'all':
            for col_name in numeric_cols:
                engineer_ft(df, ft_eng_df, col_1, col_name, op_str)
        elif op_str == 'all':
            for operator in ['diff', 'div', 'mult']:
                engineer_ft(df, ft_eng_df, col_1, col_2, operator)
    
    # If this was no recursion (or the bottom of the recursion)
    else:
        new_name = col_1 +'_'+op_str+'_' + col_2
        if op_str == 'div':
            ft_eng_df[new_name] = df[col_1]/df[col_2]
        if op_str == 'mult':
            ft_eng_df[new_name] = df[col_1]*df[col_2]
        if op_str == 'diff':
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