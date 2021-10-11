# Copyright (c) Facebook, Inc. and its affiliates.
#!/usr/bin/env python3
import math
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length
import asyncio
import math
from functools import wraps
import time
from itertools import repeat, product
from concurrent.futures import *
from collections import defaultdict
import logging
import sys

import pandas as pd
import numpy as np
from numba import jit, prange
from numba import types


estimates = defaultdict(lambda: defaultdict(list))
intervals = defaultdict(lambda: defaultdict(list))

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

@jit(nopython=True, cache=True)
def calculate_r2(y_true, y_pred, length_of_dataset):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    else:
        return 1 - numerator/denominator


@jit(nopython=True, parallel=True, cache=True)
def bootstrap_r2(df, start_index, end_index, n):
    """
    return true, upper and lower bound of r2 using standard bootstrapped technique
    """
    # we know ahead of time the shape of the final matrix
    all_results = np.zeros(n, dtype=types.float64)
    true_result = 0

    for i in prange(n):
        df_local = df[np.random.choice(len(df), size=len(df), replace=True), :]
        all_results[i] = (calculate_r2(df_local[:,0], df_local[:,1], len(df_local)))

    all_results.sort()
    true_result = calculate_r2(df[:,0], df[:,1], len(df))

    intervals = np.zeros( 3, dtype=types.float64)

    intervals[0] = all_results[start_index]
    intervals[1] = all_results[end_index]
    intervals[2] = true_result

    return intervals

@jit(nopython=True, parallel=True, cache=True)
def bootstrap_mae(df, start_index, end_index, n):
    """
    return true, upper and lower bound of mean absolute error using standard bootstrapped technique
    """
    all_results = np.zeros(n, dtype=types.float64)
    true_result = 0

    for i in prange(n):
        df_local = df[np.random.choice(len(df), size=len(df), replace=True), :]
        all_results[i] = (np.mean(np.abs(df_local[:,0]- df_local[:,1])))

    all_results.sort()
    true_result = np.mean(np.abs(df[:,0]- df[:,1]))

    intervals = np.zeros( 3, dtype=types.float64)

    intervals[0] = all_results[start_index]
    intervals[1] = all_results[end_index]
    intervals[2] = true_result

    return intervals

@jit(nopython=True, parallel=True, cache=True)
def bootstrap_rmse(df, start_index, end_index, n):
    """
    return true, upper and lower bound of root mean squared error error using standard bootstrapped technique
    """
    all_results = np.zeros(n, dtype=types.float64)
    true_result = 0

    for i in prange(n):
        df_local = df[np.random.choice(len(df), size=len(df), replace=True), :]
        all_results[i] = (np.sqrt(np.mean((df_local[:,0]- df_local[:,1]) ** 2)))

    all_results.sort()
    true_result = np.sqrt(np.mean((df[:,0]- df[:,1]) ** 2))

    intervals = np.zeros( 3, dtype=types.float64)

    intervals[0] = all_results[start_index]
    intervals[1] = all_results[end_index]
    intervals[2] = true_result

    return intervals


def regression_report(label, predicted, n=1000, as_dict=False):
    """
    return either a dataframe or a dict of true, upper and lower bound of various metrics using bootstrapped technique
    """
    assert type_of_target(label) == 'continuous'
    assert type_of_target(predicted) == 'continuous'
    check_consistent_length(label, predicted)

    df = pd.DataFrame({'label': label, 'predicted': predicted})
    df_encoded_np = df.to_numpy()

    start_index = math.floor(n*0.025)
    end_index = math.floor(n*0.975)

    results_rmse = bootstrap_rmse(df_encoded_np, start_index, end_index, n)
    results_r2 = bootstrap_r2(df_encoded_np, start_index, end_index, n)
    results_mae = bootstrap_mae(df_encoded_np, start_index, end_index, n)


    dict_results = {}

    dict_results['rmse'] = {'lower': results_rmse[0], 'upper': results_rmse[1], 'true' : results_rmse[2]}
    dict_results['r2'] = {'lower': results_r2[0], 'upper': results_r2[1], 'true' : results_r2[2]}
    dict_results['mae'] = {'lower': results_mae[0], 'upper': results_mae[1], 'true' : results_mae[2]}

    if as_dict:
        return dict_results
    # otherwise, return the results as a pandas dataframe that can be printed out
    formatted_results = []

    formatted_results.append({ 'metric': 'rmse', 'lower': results_rmse[0], 'upper': results_rmse[1], 'true' : results_rmse[2]})
    formatted_results.append({ 'metric': 'r2', 'lower': results_r2[0], 'upper': results_r2[1], 'true' : results_r2[2]})
    formatted_results.append({ 'metric': 'mae', 'lower': results_mae[0], 'upper': results_mae[1], 'true' : results_mae[2]})

    df = pd.DataFrame(formatted_results)
    return df
