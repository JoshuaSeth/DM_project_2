import pandas as pd

df = pd.read_csv('/Users/sethvanderbijl/Coding Projects/DM_project_2/test.csv')

df = df.sort_values(['srch_id', 'y'], ascending=[True, False])

df=df.drop(['y', 'Unnamed: 0'], axis=1)

df.to_csv('preds.csv', index=False)