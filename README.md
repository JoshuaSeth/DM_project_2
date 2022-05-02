# Setup

For easy importing from sibling directories/folders.

```
pip install -e .
```

# Examples

## Loading the data and include some (pre) feature-engineering
```
from data.load import load_data

load_data(test=False, add_day_parts=True, divided_fts=[])
```

### Load data parameters
The load data function has several parameters that allow some basic fearture engineering of the data. This allows us to both tweak the feature engineering live and integrate it easily into some hyperparameter optimization. This might be especially useful considering that last time part of the model improvement iterations was selecting different hyperparams.

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
- caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Main reason for setting it to false is that each cached df can easily be 4GB. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.

### Caching of load_data
When running hyperparam selection don't worry about regenerating the dataset using different parameters giving overhead. I'll try to integrate memoization/caching for this (locally on your PC, so nothing pushed to github). So the next time you call the function with the same arguments it will return something cached.

# Guide

## Data loading
Folder for loading the dataset with the load_data function.

## Models
Folder for building, training and evaluating the ML models for the task. 

## Experiments / results
Maybe here we can run the official experiments? So we will have a uniformalized experiment with clear parameters for all the models. Let's see.

## Statistics
Perform the statistical measurements for the performance of the final model. Currently we have chosen for the statistics:
- To be decided

## Plots
Generate the plots for the final chosen model. We have chosen to include these plots:
- To be decided

## Playgrounds
Last time you guys took a different approach for moving your code into the project, that would be fine too, so see what works best for you!
