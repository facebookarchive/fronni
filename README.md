
# fronni
A Python library for quickly calculating & displaying machine learning model performance metrics with confidence intervals.

# How fronni works?

https://medium.com/@kaushikm/fronni-a-python-library-for-quickly-calculating-machine-learning-model-performance-metrics-with-3baf28eaa5c0

## Requirements

* Python >= 3.6

* numba

* numpy

* scikit-learn

* plotly

## Installing fronni

pip install fronni


## Full documentation

Functions from the classification module:

### classification_report

Generates confidence intervals for precision, recall, & F1 metrics for a binary or multi-class classification model, given arrays of predicted & label values.

| Parameter | Type | Default |
|--|--|--|
| label | Numpy array or Pandas series | None
| predicted | Numpy array or Pandas series | None
| n | integer, number of bootstrap iterations | 1,000
| confidence_level | integer value between 1 & 100 | 95
| as_dict | Boolean, return nested dictionary if True otherwise Pandas dataframe | False
| confidence_level | value between 1 & 100 | 95
| sort_by_sample_size | Boolean, return the Pandas dataframe, sorted in descending order of class sample size | False

### plot_classification_report

Plots precision, recall, & confidence intervals for F1 metrics for a binary or multi-class classification model, given a classification report input.

| Parameter | Type | Default |
|--|--|--|
| report | output from classification_report | None
| save_to_filename | string, path of filename image to save like "image.png" | None

From the regression module:

### regression_report

Generates confidence intervals for RMSE, MAE, and R^2 metrics for a regression model, given arrays of predicted & label values.

| Parameter | Type | Default |
|--|--|--|
| label | Numpy array or Pandas series | None
| predicted | Numpy array or Pandas series | None
| n | integer, number of bootstrap iterations | 1,000
| as_dict | Boolean, return nested dictionary if True otherwise Pandas dataframe | False

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

fronni is Apache 2.0 licensed, as found in the LICENSE file.
