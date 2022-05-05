'''Quick file for generating a ranking from your predictions of the test set.'''
import pandas as pd 


def to_ranking(test_df, predictions):
    '''Quick function for generating a ranking from your predictions of the test set. 
    returns: A ranking of the properties from the given book_bool prediction for each row
    - df: the test_df given to us. Might be a transformed feature engineered one, doesn't matter as long as the columns srch_id and property_id are present.
    - preds: predictions for each row in the test set'''
    # Convert to three cols with srch property and predicted book bool
    results = pd.DataFrame(zip(test_df['srch_id'], test_df['prop_id'],predictions), columns=['srch_id', 'prop_id', 'y'])
    # Sort the book bools per search
    results = results.sort_values(['srch_id','y'],ascending=False)
    results = results.drop('y', axis=1)
    return results