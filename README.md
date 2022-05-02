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
- divided_fts: What features to divide by each other to create new features, a list of column names, an empty list or 'all' for all column names. Default empty list. This operation is applied last so you can also divide by season, daypart or other generated features.
- add_seasons: Whether to add in which season a date belongs. The seasons are four onne-hot columns named season_0, season_1, etc. representing winter, spring, etc.
- caching: Whether to cache a function call for next time. With caching the next time you call a function with the same args it will be loaded from disk and returned. Set it to false to not generate a cache everytime you call a function. Default is true. Caching happens in two ways. First it checks all the arguments passed to the function and checks whether we have a df exactly matching these requirements. Else it starts building the df according to the feature generation requirements, but for most buildsteps it checks whether it has the result of this build step in cache already. Of course the latter caching is assuming non-interactive build steps (i.e. the results of dayparts doesn't change to operation for adding seasons) else the needed cache becomes exponential and instead of saving the result fo a build operation and concatenating it with the df we would have to save the result of the build operation on the df in the df itself making for a huge cache. Improvements welcome.

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
