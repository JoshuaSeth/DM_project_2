# Setup

For easy importing from sibling directories/folders.

```
pip install -e .
```

You will find out whatever libraries you need, the most confusing one might be pytables:
```
pip install tables
```

# Examples

## Loading the data and include some (pre) feature-engineering
```
from data.load import load_data

# Most simple: full dataset no feature generation operations
df = load_data()

# Load test set with reading it from to cache but without writing it to the cache
df = load_data(test=True, caching=False)

# Dividing price_usd column by all other columns
df = load_data(test=False, add_day_parts=True, fts_operations=[('price_usd', 'all', 'div')], add_seasons=True, num_rows=1000)

# Every possible feature engineering operation
df = load_data(add_day_parts=True, fts_operations=[('all', 'all', 'all')], add_seasons=True)
```

### Load data parameters
The load data function has several parameters that allow some basic fearture engineering of the data. This allows us to both tweak the feature engineering live and integrate it easily into some hyperparameter optimization. This might be especially useful considering that last time part of the model improvement iterations was selecting different hyperparams.

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
CURRENTLY CACHING DOES NOT WORK FOR THIS OPERATION. IT DOES SAVE THE RESULTS TO CACHE BUT WHEN YOU CALL LOAD-DATA WITH EXACTLY THE SAME ARGUMENTS THE FUNCTION DOES NOT RECOGNIZE THIS ARGUMENT AS BEING EXACTLY THE SAME AS LAST TIME. MEANING IT WILL REGENERATE THE DATA EVEN THOUGH IT HAS IT IN CACHE.
- add_seasons: Whether to add in which season a date belongs. The seasons are four onne-hot columns named season_0, season_1, etc. representing winter, spring, etc.
- num_rows: Number of rows to read from the DF, by default the full DF is returned. Useful for loading smaller pieces for testing etc.
- scaling: Whether to MinMax scale the data. Added this to save some boilerplate code and so that this intesive operation is now also cached since it takes longer than loading the data itself. Default false
- fill_nan: operation to call on the df to fill nans. Default None so nan's not filled. For example: 'mean' will do df.fillna(df.mean())
- caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Main reason for setting it to false is that each cached df can easily be 4GB. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.

### Caching of load_data
When running hyperparam selection don't worry about regenerating the dataset using different parameters giving overhead. I'll try to integrate memoization/caching for this (locally on your PC, so nothing pushed to github). So the next time you call the function with the same arguments it will return something cached.

Caching does not work with same_value_operations. It does save this to cache but when requesting exactly the same data it regenerates the data instead of loading it from cache.

# Guide

## Data loading
Folder for loading the dataset with the load_data function.

## Models
Folder for building, training and evaluating the ML models for the task. 

### StackingRegressor
Code complete and working. Did not finished training yet. no evaluation yet.

### NeuralNet
Code complete and working. Did not finished training yet. no evaluation yet.


## Experiments / results
Maybe here we can run the official experiments? So we will have a uniformalized experiment with clear parameters for all the models. Let's see.

## Helpers
Helper functions for converting scores per row to ranks and kaggle-like evaluation.

## Statistics
Perform the statistical measurements for the performance of the final model. Currently we have chosen for the statistics:
- To be decided

## Plots
Generate the plots for the final chosen model. We have chosen to include these plots:
- To be decided

## Playgrounds
Last time you guys took a different approach for moving your code into the project, that would be fine too, so see what works best for you!
