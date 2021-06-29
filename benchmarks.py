Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

import logging
import sys
import pandas as pd
from functools import wraps
import time
from collections import defaultdict
import math
from itertools import repeat
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.metrics import classification_report as sklearn_classification_report
from .classification import classification_report
from sklearn.datasets import make_gaussian_quantiles

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

RANDOM_SEED = 27

def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):

        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output
    return timed

class TestPerformance:

    def __init__(self, n_samples=10000, bootstrap_samples=1000):
        self.n_samples = n_samples
        self.bootstrap_samples = bootstrap_samples
        self.start_index = math.floor(bootstrap_samples*0.025)
        self.end_index = math.floor(bootstrap_samples*0.975)
        self.create_dataset()
        self.test_pandas_sklearn_series()
        self.test_pandas_sklearn_parallel()
        self.test_numba_nosklearn_parallel()

    def create_dataset(self):
        X1, y1 = make_gaussian_quantiles(cov=3.,
                                        n_samples=self.n_samples, n_features=2,
                                        n_classes=5, random_state=RANDOM_SEED)
        X1 = pd.DataFrame(X1)
        self.label = pd.Series(y1).to_numpy()
        np.random.seed(RANDOM_SEED)
        self.predicted = pd.Series(y1)[np.random.choice(len(y1), size=len(y1), replace=False)].to_numpy()

        #self.labels = List(np.unique(self.label))
        #self.n_labels = len(self.labels)
        self.df = pd.DataFrame({'label': self.label, 'predicted': self.predicted})

    def interval_calculator(self, values):
        sorted_values = sorted(values)
        return [sorted_values[self.start_index], sorted_values[self.end_index]]

    @timeit
    def test_pandas_sklearn_series(self):
        """
        resample in pandas
        call classification_report n times
        """
        results = defaultdict(lambda: defaultdict(list))
        for _i in range(self.bootstrap_samples):
            resampled = self.df.sample(n = self.n_samples, replace=True)
            report = sklearn_classification_report(resampled['label'], resampled['predicted'], output_dict=True)
            for key, val in report.items():
                if key not in ['accuracy','macro avg','weighted avg']:
                    for metric_name, metric_val in val.items():
                        if metric_name in ['precision','recall','f1']:
                            results[key][metric_name].append(metric_val)
        intervals = defaultdict(lambda: defaultdict(list))
        for class_name, metrics in results.items():
            for metric_name, values in metrics.items():
                results = self.interval_calculator(values)
                intervals[class_name][metric_name] = results

    @staticmethod
    def parallel_reporter(iteration, df, n_samples):
        resampled = df.sample(n = n_samples, replace=True)
        report = sklearn_classification_report(resampled['label'], resampled['predicted'], output_dict=True)
        # report = classification_report(df['label'], df['predicted'], output_dict=True)
        # store in a nested dict by [class_name][metric_name], where the value is a list
        results = {}
        for key, val in report.items():
            if key not in ['accuracy','macro avg','weighted avg']:
                for metric_name, metric_val in val.items():
                    if metric_name in ['precision','recall','f1']:
                        results[key + '_' + metric_name] = metric_val
        return results

    @staticmethod
    def parallel_interval_calculator(key, values, start_index, end_index):
        sorted_values = sorted(values)
        return [key, sorted_values[start_index], sorted_values[end_index]]

    @timeit
    def test_pandas_sklearn_parallel(self):
        with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor: # processes can't share memory so can't write to estimates dict
            results = list(executor.map(self.parallel_reporter, range(self.bootstrap_samples), repeat(self.df), repeat(self.n_samples)))
            all_results = defaultdict(list)
            for result in results:
                # each result is a dict
                for key, val in result.items():
                    all_results[key].append(val)
            results = list(executor.map(self.parallel_interval_calculator, *zip(*all_results.items()), repeat(self.start_index), repeat(self.end_index)))

    def test_numpy_sklearn_series(self):
        pass

    def test_numpy_nosklearn_series(self):
        pass

    def test_numpy_sklearn_parallel(self):
        pass

    def test_numpy_nosklearn_parallel(self):
        pass

    def test_numba_nosklearn_series(self):
        pass

    @timeit
    def cache_numba(self):
        classification_report(self.label, self.predicted, n=2)

    @timeit
    def test_numba_nosklearn_parallel(self):
        report = classification_report(self.label, self.predicted, n=self.bootstrap_samples)
        print(report)

#test = TestPerformance(n_samples=10000, bootstrap_samples=1000)
