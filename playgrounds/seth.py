from data.load import load_data
import os
import numpy as np
import pandas as pd
from helpers.evaluation import kaggle_like_rank_score
from helpers.constants import kaggle_test_ids
from models.ndcg import ndcg_eval

prefix = os.path.dirname(os.path.abspath(__file__))

# df = pd.read_csv(prefix + os.sep+'..'+os.sep+'data'+os.sep+'training_set_VU_DM.csv')
# df = df[df['srch_id'].isin(kaggle_test_ids)]

# t=ndcg_eval(df, df)

t = load_data(mode='kaggle_test')

print(t)