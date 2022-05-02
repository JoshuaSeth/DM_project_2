# Setup

For easy importing from sibling directories/folders.

```
pip install -e .
```

# Examples

## Loading the data and include some (pre) feature-engineering
```
from data.load import load_data

load_data(args)
```

### Load data parameters
The load data function has several parameters that allow some basic fearture engineering of the data. This allows us to both tweak the feature engineering live and integrate it easily into some hyperparameter optimization. This might be especially useful considering that last time part of the model improvement iterations was selecting different hyperparams.

### Caching of load_data
When running hyperparam selection don't worry about regenerating the dataset using different parameters giving overhead. I'll try to integrate memoization/caching for this (locally on your PC, so nothing pushed to github). So the next time you call the function with the same arguments it will return something cached (the way of caching dependening on how much data this is going to take (on disk? in memory?))

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
