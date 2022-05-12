'''File for the load_data function.'''
import math
import os
from itertools import product
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import numpy as np
import os.path
from os import path
from warnings import simplefilter
from helpers.constants import kaggle_test_ids
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Some global initialization
tqdm.pandas()
pd.set_option('io.hdf.default_format','table')
prefix = os.path.dirname(os.path.abspath(__file__))



def load_data(mode='train', add_day_parts=False, fts_operations=[], same_value_operations=[], order_operations = [], add_seasons=False, add_dist_holidays=False, num_rows=None, scaling=False, fill_nan=None,caching=True):
    '''Load the dataset (or cached version) with several params as feature engineering.\n A set of 30 Search ids is withheld to function as test_set for our kaggle-like evaluation function.
    Params:
    - test: Whether to return train personal ndcg test (100 srch_ids) or kaggle test data (large). Possible values 'train', 'kaggle_test', 'our_test'. Default: 'train'
    - add_day_parts: Whether to create one-hot columns whether the event is in a certain part of day. Default is False.
    - fts_operations: What operationns to apply to what features to create new features. A list of 3-tuples. For example if you want to divide price by distance and get the difference between visitor_starrating and prop_starrrating you do [('price_usd', 'orig_destination_distance', 'div'), ('prop_starrating', 'visitor_hist_starrating', 'diff')]. So the first two items of the tuple are column names and the third item of the tuple is the applied operation. Default empty list. This operation is applied last so you can also divide by season, daypart or other intermediary generated features. You can provide the string 'all' to any of the positions in the tuple. When doing this for the first or second position it means it will apply this operation to all the columns. When doing this for the third position it means it will apply all possible operations to the two given columns. For example: [('price_usd', 'orig_destination_distance', 'all')] will create three new columns in the df for all 3 operations. [('price_usd', 'all', 'div')] will divide the price by all other columns in the df. Theoretically you could be extremely extensive and provide all operations to all columns by [('all', 'all', 'all')] which will add 12000 columns.
    Currently supported operations:
            - 'diff': absolute difference
            - 'div': division
            - 'mult': multiplication
    - same_value_operations: In need of a better name. Operations applied to the set where all values are the same. A list of tuples [()] where the first item of the tuple is the column name where you check for equal values (i.e. 'site_id') and the second item of the tuple is the values for which you want to apply the operation (i.e. 'price_usd') and where the third item of the tuple is the applied operation (i.e. 'avg'). This is the example for location id we were talking about. For example [('site_id', 'price_usd', 'avg')] will give the average price for all rows with this site. This operation is now also extended to include the 'all' keyword. So ('prop_id', 'all', 'avg') will give the average per property id for all other numeric columns. So avg usd for every property, avg desriability for every property, etc. Like competition-winner Zhang had done in his feature engineering (see ppt).
    Possible values for operations are:
            - 'avg': average/mean
    - order_operations: Gives the order of a quanity among others with the same value. For example ('srch_id', 'price_usd') gives 1 if this was the highest price for this srch_id, 5 if the was the 5th highest price, etc.  
    - add_seasons: Whether to add in which season a date belongs. The seasons are four onne-hot columns named season_0, season_1, etc. representing winter, spring, etc.
    - add_dist_to_holidays: Adds a distance to the next and from the last holiday. 
    - num_rows: Number of rows to read from the DF, by default the full DF is returned. Useful for loading smaller pieces for testing etc.
    - scaling: Whether to MinMax scale the data. Added this to save some boilerplate code and so that this intesive operation is now also cached since it takes longer than loading the data itself. Default false
    - fill_nan: UNTESTED operation to call on the df to fill nans. Default None so nan's not filled. For example: 'mean' will do df.fillna(df.mean())
    - caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Main reason for setting it to false is that each cached df can easily be 4GB. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.'''
    # If we have a cache for a function with these args load it else generate it, save it and return it
    # Cache files are based on the specific argument and the number of rows
    cache_name = str((mode, add_day_parts, hash(tuple(fts_operations)),hash(tuple(same_value_operations)), add_seasons, add_dist_holidays, scaling, fill_nan))
    cache_path = prefix +os.sep+ 'caches'+os.sep + cache_name+str(num_rows)+'.h5'
    daypart_cachename = prefix+os.sep+'caches'+os.sep+'dayparts'+str((mode,num_rows))+'.h5'
    seasons_cachename = prefix+os.sep+'caches'+os.sep+'seasons'+str((mode,num_rows))+'.h5'
    dist_holiday_cachename = prefix+os.sep+'caches'+os.sep+'dist_holiday'+str((mode,num_rows))+'.h5'
    sameid_cachename = prefix+os.sep+'caches'+os.sep+ str(hash(tuple(same_value_operations)))+str((mode,num_rows))+'.h5'
    ft_engineer_cachename = prefix+os.sep+'caches'+os.sep+ str(hash(tuple(fts_operations)))+str((mode,num_rows))+'.h5'
    order_cachename = prefix+os.sep+'caches'+os.sep+ str(hash(tuple(order_operations)))+str((mode,num_rows))+'.h5'

    if (path.isfile(cache_path)):
        print('DF with these specific params is cached, returning cached version.')
        df= pd.read_hdf(cache_path, start=0, stop=num_rows)
        if mode == 'train':
            df=df[~df['srch_id'].isin(kaggle_test_ids)] # Filter kaggle test set
        return df
    
    # In case we do not have a cache we manually have to apply the operations
    else:
        # Load data
        print('No cache for this specific request, start loading base df from disk.')
        base_path = prefix + os.sep+'test_set_VU_DM.csv' if mode=='kaggle_test' else prefix + os.sep+'training_set_VU_DM.csv'
        try:
            df = pd.read_csv(base_path, nrows=num_rows)
        except Exception as e: 
            print('Couldnt find the data file, can you move the training_set_VU_DM.csv to', base_path, '? This file is to large to share via github. Error:\n', e) 
            return
        
        # Always translate string to datetime for datetime column no param needed
        df['date_time'] = pd.to_datetime(df['date_time'])

        # Withhold kaggle test set or only return that one
        if mode == 'train':
            df=df[~df['srch_id'].isin(kaggle_test_ids)]
        if mode == 'our_test':
            df=df[df['srch_id'].isin(kaggle_test_ids)]

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

        # 3. Add the distance to the next holiday
        if add_dist_holidays:
            df = df.sort_values('date_time')
            df, has_cache = try_add_cached(dist_holiday_cachename, df)
            if not has_cache:
                print('generating holidays')
                cal = calendar()
                holidays = cal.holidays(start=df['date_time'].min(), end=df['date_time'].max())
                holidays = pd.DataFrame(holidays, columns=['holiday'])
                
                df = pd.merge_asof(df, holidays, left_on='date_time', right_on='holiday', direction='forward')
                df = pd.merge_asof(df, holidays, left_on='date_time', right_on='holiday')
                df['until_holiday'] = df.pop('holiday_x').sub(df['date_time']).dt.days
                df['since_holiday'] = df['date_time'].sub(df.pop('holiday_y')).dt.days
                
                print(df[['date_time', 'until_holiday', 'since_holiday']])
                if caching: df[['until_holiday', 'since_holiday']].to_hdf(dist_holiday_cachename, key='df')
                del holidays
            df = df.sort_values('srch_id')

        # 4. Adding averages or stds for certain columns where for example id is the same
        if same_value_operations !=[]:
            df, has_cache = try_add_cached(sameid_cachename, df)
            if not has_cache:
                print('generating same id features')
                same_vals_df = pd.DataFrame()
                for col_1, col_2, op_str in tqdm(same_value_operations):
                    same_val_op(df, same_vals_df, col_1, col_2, op_str)
                df = pd.concat([df, same_vals_df.set_index(df.index)], axis=1)
                if caching: same_vals_df.to_hdf(sameid_cachename, key='df')
                del same_vals_df
        
        # 5. Do order operations for example order of this price for the srch id
        if order_operations !=[]:
            df, has_cache = try_add_cached(order_cachename, df)
            if not has_cache:
                print('generating order features')
                order_df = pd.DataFrame()
                for col_1, col_2 in tqdm(order_operations):
                    order_operate(df, order_df, col_1, col_2)
                df = pd.concat([df, order_df.set_index(df.index)], axis=1)
                if caching: order_df.to_hdf(order_cachename, key='df')
                del order_df
        
        # 6. Create new features by dividing columns by each other
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
            df[df==np.Infinity]=np.nan
            scaler = MinMaxScaler()
            cols = df.columns[~df.columns.isin(['date_time', 'srch_id', 'prop_id'])]
            df[cols]= scaler.fit_transform(df[cols])
                
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

def same_val_op(df, same_vals_df, col_1, col_2, op_str):
    '''Internal. Recursively add the avg where another column is the same. For example (col_1:'prop_id', col_2: 'price_usd', op_str:'avg' will give the average usd price for all properties, so in whichever row property is the same.'''

    # Don't do operations on things that are only in train set, then tests set will not have the same num cols
    if not set(['booking_bool','click_bool', 'position', 'gross_bookings_usd']).isdisjoint([col_1, col_2]):
        return

    if 'all' in [col_1, col_2, op_str]:
        # Filter this warning cause you'll get a thousand
        simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Only numeric columns
        if col_1 == 'all':
            for col_name in df.columns: #Doesnt need to be numeric
                same_val_op(df, same_vals_df, col_name, col_2, op_str)
        elif col_2 == 'all':
            for col_name in tqdm(numeric_cols): #We do tqdm here cause probably the 'all' argument will be in this position and not in the first or last position. (makes the most sense)
                same_val_op(df, same_vals_df, col_1, col_name, op_str)
        elif op_str == 'all':
            for operator in ['avg']: # Currently only avg want to extend this
                same_val_op(df, same_vals_df, col_1, col_2, operator)

    else:
        # For a normal item or the bottom of recursion
        _dict = df.groupby(col_1)[col_2].mean().to_dict() # A dict with the operation applied to each group of the id 
        same_vals_df['same_val_'+col_1+'_'+col_2+'_'+op_str] = df[col_1].progress_apply(get_from_dict, _dict=_dict)
    
        

def order_operate(df, order_df, col_1, col_2):
    '''Internal. Get other of item in quantity.'''
    # Don't do operations on things that are only in train set, then tests set will not have the same num cols
    if not set(['booking_bool','click_bool', 'position', 'gross_bookings_usd']).isdisjoint([col_1, col_2]):
        return

    if 'all' in [col_1, col_2]:
        # Filter this warning cause you'll get a thousand
        simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # Only numeric columns
        if col_1 == 'all':
            for col_name in df.columns: #Doesnt need to be numeric
                order_operate(df, order_df, col_name, col_2)
        elif col_2 == 'all':
            for col_name in tqdm(numeric_cols): #We do tqdm here cause probably the 'all' argument will be in this position and not in the first or last position. (makes the most sense)
                order_operate(df, order_df, col_1, col_name)

    else:
        # For a normal item or the bottom of recursion
        df = df.sort_values([col_1, col_2], ascending=[True, False])
        order_df['order_'+col_1+'_'+col_2]= df.groupby(col_1).cumcount()
    

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
    # Don't do operations on things that are only in train set, then tests set will not have the same num cols
    if not set(['booking_bool','click_bool', 'position', 'gross_bookings_usd']).isdisjoint([col_1, col_2]):
        return
        
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