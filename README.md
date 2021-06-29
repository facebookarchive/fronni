
# fronni
A Python library for quickly calculating & displaying machine learning model performance metrics with confidence intervals.

Data scientists spend a lot of time evaluating the performance of their machine learning models. A common means of doing so is a classification report, such as the one below built-into the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) (referenced over 440,000 times on Github).


The problem with depending on this is that when you have imbalanced and/or small datasets in your test sample, the volume of data that makes up each point estimate can be very small. As a result, whatever estimates you have for the precision and recall values will likely be unrepresentative of what happens in the real world. Instead, it would be far better to present to the reader a range of values from confidence intervals.

We can easily create confidence intervals for any metric of interest by using the [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29#:~:text=This%20technique%20allows%20estimation%20of,sampling%20from%20an%20approximating%20distribution) technique to create hundreds of samples from the original dataset (with replacement), and then throw away the most extreme 5% of values to get a 95% confidence interval, for example.




## Examples

With a pandas dataframe named 'df', you can generate your metrics like this:

    from fronni.classification import classification_report
    report = classification_report(df['label'], df['predicted'], n=1000)
    print(report)

## Requirements

* Python >= 3.6

* numba

* numpy

* scikit-learn

* plotly



## Installing fronni

pip install fronni



## How fronni works

When the datasets are large, running the bootstrap as described above becomes a computationally expensive calculation. Even a relatively small sample of a million rows, when resampled 1,000 times, turns into a billion-row dataset and causes the calculations to take close to a minute. This isn’t a problem when doing these calculations offline, but engineers are often working interactively in a Jupyter notebook and want answers quickly. For this reason we decided to create a few fast implementations of these calculations for common machine learning metrics, such as precision & recall using the [numba library](http://numba.pydata.org/), which provides a speedup of approximately 23X over regular Python parallel-processing code.



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
