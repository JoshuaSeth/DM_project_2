'''The kaggle-like rank evaluation will come here'''

from data.load import load_data
from helpers.constants import kaggle_test_ids
import os
import pandas as pd
import os.path
from os import path

prefix = os.path.dirname(os.path.abspath(__file__))


def kaggle_like_rank_score():
    '''A rough proxy for the kaggle competition score a model gets. At least allows for internal comparison'''
    print('\nevaluating kaggle score. The first time this will be a bit slow since it must create the test set from the whole data. Times after the first, this should be fast.')
    # Try load test data from cache
    cache_name = prefix +os.sep+ 'caches'+os.sep + 'kaggle_test_data'
    if (path.isfile(cache_name)):
        df = pd.read_hdf(cache_name)
    
    else: # If first time calling function generate and cache this data
        df = pd.read_csv(prefix + os.sep+'..'+os.sep+'data'+os.sep+'training_set_VU_DM.csv')
        df = df[df['srch_id'].isin(kaggle_test_ids)]
        df.to_csv('test_kaggle.csv')
        print(df)