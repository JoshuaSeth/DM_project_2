from doctest import OutputChecker
from data.load import load_data

df = load_data(test=False, add_day_parts=True, same_value_operations=[('site_id', 'price_usd', 'avg')], add_seasons=True, num_rows=100)

df.to_csv('test.csv')

print(df)
# num_rows
# average over prop_id

# mail jordi about OutputChecker

# vis_hist_starrating difference

# why NAN month datetime?