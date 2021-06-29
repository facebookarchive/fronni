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
import unittest

import numpy as np
import pandas as pd
from numba.typed import List
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .classification import (
    create_numba_list,
    classification_report,
    metrics_from_confusion_matrix,
    confusion_matrix,
    plot_classification_report,
)
from .regression import regression_report

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

RANDOM_SEED = 27


class TestMultiClassClassification:
    @classmethod
    def setup_class(cls):
        """"""
        # Construct dataset
        X1, y1 = make_gaussian_quantiles(
            cov=3.0,
            n_samples=10000,
            n_features=2,
            n_classes=5,
            random_state=RANDOM_SEED,
        )
        X1 = pd.DataFrame(X1)
        cls.label = pd.Series(y1).to_numpy()
        np.random.seed(RANDOM_SEED)
        cls.predicted = pd.Series(y1)[
            np.random.choice(len(y1), size=len(y1), replace=False)
        ].to_numpy()

        cls.labels = create_numba_list(np.unique(cls.label))
        cls.n_labels = len(cls.labels)
        df = pd.DataFrame({"label": cls.label, "predicted": cls.predicted})
        cls.df_np = df.to_numpy()
        cls.matrix = confusion_matrix(
            cls.df_np[:, 0], cls.df_np[:, 1], labels=cls.labels, n_labels=cls.n_labels
        )

    def test_confusion_matrix(self):
        sklearn_matrix = sklearn_confusion_matrix(self.label, self.predicted)
        assert np.array_equal(self.matrix, sklearn_matrix)

    def test_metrics(self):
        """
        put into a identically-formatted dictionaries to be able to use == operator
        """
        report_sklearn = sklearn_classification_report(
            self.label, self.predicted, output_dict=True
        )
        formatted_report = List()
        for key, val in report_sklearn.items():
            if key not in ["accuracy", "macro avg", "weighted avg"]:
                formatted_report.append(val["precision"])
                formatted_report.append(val["recall"])
                formatted_report.append(val["f1-score"])
        report = metrics_from_confusion_matrix(self.matrix, self.labels)
        assert formatted_report == report

    def test_confidence_intervals(self):
        """
        validate that the lower is always <= upper
        and that the point estimate lies within
        """
        np.random.seed(RANDOM_SEED)
        report = classification_report(self.label, self.predicted)
        assert (report["lower"] <= report["upper"]).all()


class TestBinaryClassification:
    @classmethod
    def setup_class(cls):
        """"""
        # Construct dataset
        X1, y1 = make_gaussian_quantiles(
            cov=3.0,
            n_samples=10000,
            n_features=2,
            n_classes=2,
            random_state=RANDOM_SEED,
        )

        cls.label = pd.Series(y1).to_numpy()
        np.random.seed(RANDOM_SEED)
        cls.predicted = pd.Series(y1)[
            np.random.choice(len(y1), size=len(y1), replace=False)
        ].to_numpy()
        cls.labels = create_numba_list(np.unique(cls.label))
        n_labels = len(cls.labels)
        df = pd.DataFrame({"label": cls.label, "predicted": cls.predicted})
        df_np = df.to_numpy()
        cls.matrix = confusion_matrix(
            df_np[:, 0], df_np[:, 1], labels=cls.labels, n_labels=n_labels
        )

    def test_confusion_matrix(self):
        sklearn_matrix = sklearn_confusion_matrix(self.label, self.predicted)
        assert np.array_equal(self.matrix, sklearn_matrix)

    def test_metrics(self):
        """
        put into a identically-formatted dictionaries to be able to use == operator
        """
        report_sklearn = sklearn_classification_report(
            self.label, self.predicted, output_dict=True
        )
        formatted_report = List()
        for key, val in report_sklearn.items():
            if key not in ["accuracy", "macro avg", "weighted avg"]:
                formatted_report.append(val["precision"])
                formatted_report.append(val["recall"])
                formatted_report.append(val["f1-score"])

        report = metrics_from_confusion_matrix(self.matrix, self.labels)
        assert formatted_report == report

    def test_confidence_intervals(self):
        """
        validate that the lower is always <= upper
        and that the point estimate lies within
        """
        np.random.seed(RANDOM_SEED)
        report = classification_report(self.label, self.predicted)
        assert (report["lower"] <= report["upper"]).all()


class TestBootstrap:
    @classmethod
    def setup_class(cls, input_type="integer"):
        """"""
        # Construct dataset
        X1, y1 = make_gaussian_quantiles(
            cov=3.0,
            n_samples=10000,
            n_features=2,
            n_classes=2,
            random_state=RANDOM_SEED,
        )

        if input_type == "string":
            covert_array_to_string = np.vectorize(
                lambda x: "True" if x == 1 else "False"
            )
            y1 = covert_array_to_string(y1)
        cls.label = pd.Series(y1).to_numpy()
        np.random.seed(RANDOM_SEED)
        cls.predicted = pd.Series(y1)[
            np.random.choice(len(y1), size=len(y1), replace=False)
        ].to_numpy()
        # cls.labels = List(np.unique(cls.label))
        cls.df = pd.DataFrame({"label": cls.label, "predicted": cls.predicted})
        cls.df_np = cls.df.to_numpy()

    def test_num_rows(self):
        """
        verify we get df_length samples back
        """
        df_length = self.df.shape[0]
        df_local = self.df_np[
            np.random.choice(df_length, size=df_length, replace=True), :
        ]
        assert df_local.shape[0] == df_length

    def test_randomness(self):
        """
        verify the resampled dataframe is different from the original
        """
        df_length = self.df.shape[0]
        df_local = self.df_np[
            np.random.choice(df_length, size=df_length, replace=True), :
        ]
        assert np.array_equal(self.df, df_local) is False

    def test_confidence_intervals(self):
        """
        validate that the lower is always <= upper
        and that the point estimate lies within
        """
        np.random.seed(RANDOM_SEED)
        report = classification_report(self.label, self.predicted)
        assert (report["lower"] <= report["upper"]).all()


class TestBootstrap_Regression:
    @classmethod
    def setup_class(cls):
        """"""
        # Construct dataset
        mu = np.array([0.0, 0.0])
        num_samples = 10000
        # The desired covariance matrix.
        r = np.array(
            [
                [1.414, 1.414],
                [1.414, 1.414],
            ]
        )
        y = np.random.multivariate_normal(mu, r, size=num_samples)
        cls.y_true = y[:, 0]
        cls.y_pred = y[:, 1]
        cls.df_dict = regression_report(y[:, 0], y[:, 1], as_dict=True)
        cls.df = regression_report(y[:, 0], y[:, 1], as_dict=False)

    def test_r2_score(self):
        """
        verify we get the r2 score
        """
        scikit_r2_score = round(r2_score(self.y_true, self.y_pred), 2)
        numba_r2_score = round(self.df_dict["r2"]["true"], 2)

        assert scikit_r2_score == numba_r2_score

    def test_mae(self):
        """
        verify we get the r2 score
        """
        scikit_mae = round(mean_absolute_error(self.y_true, self.y_pred), 2)
        numba_mae = round(self.df_dict["mae"]["true"], 2)

        assert scikit_mae == numba_mae

    def test_rmse(self):
        """
        verify we get the r2 score
        """
        scikit_rmse = round(np.sqrt(mean_squared_error(self.y_true, self.y_pred)), 2)
        numba_rmse = round(self.df_dict["rmse"]["true"], 2)

        assert scikit_rmse == numba_rmse

    def test_confidence_intervals(self):
        """
        validate that the lower is always <= upper
        and that the point estimate lies within
        """

        assert (self.df["lower"] <= self.df["upper"]).all()


class TestPlots:
    @classmethod
    def setup_class(cls):
        """"""
        # Construct dataset
        X1, y1 = make_gaussian_quantiles(
            cov=3.0,
            n_samples=10000,
            n_features=2,
            n_classes=2,
            random_state=RANDOM_SEED,
        )

        cls.label = pd.Series(y1).to_numpy()
        np.random.seed(RANDOM_SEED)
        cls.predicted = pd.Series(y1)[
            np.random.choice(len(y1), size=len(y1), replace=False)
        ].to_numpy()
        cls.labels = create_numba_list(np.unique(cls.label))
        n_labels = len(cls.labels)
        df = pd.DataFrame({"label": cls.label, "predicted": cls.predicted})
        df_np = df.to_numpy()
        cls.matrix = confusion_matrix(
            df_np[:, 0], df_np[:, 1], labels=cls.labels, n_labels=n_labels
        )

    def test_plot(self):
        """
        verify that we're able to plot the results without errors
        """
        np.random.seed(RANDOM_SEED)
        report = classification_report(self.label, self.predicted)
        assert plot_classification_report(report) == None


class fronni_unittests(unittest.TestCase):
    def test_bootstrap_binary_integer_input(self):
        bootstrap_test_instance = TestBootstrap()
        bootstrap_test_instance.setup_class()
        bootstrap_test_instance.test_num_rows()
        bootstrap_test_instance.test_randomness()
        bootstrap_test_instance.test_confidence_intervals()

    def test_bootstrap_binary_string_input(self):
        bootstrap_test_instance = TestBootstrap()
        bootstrap_test_instance.setup_class(input_type="string")
        bootstrap_test_instance.test_num_rows()
        bootstrap_test_instance.test_randomness()
        bootstrap_test_instance.test_confidence_intervals()

    def test_binary_classification(self):
        binary_classification_test_instance = TestBinaryClassification()
        binary_classification_test_instance.setup_class()
        binary_classification_test_instance.test_confusion_matrix()
        binary_classification_test_instance.test_metrics()
        binary_classification_test_instance.test_confidence_intervals()

    def test_multi_class_classification(self):
        multi_class_classification_test_instance = TestMultiClassClassification()
        multi_class_classification_test_instance.setup_class()
        multi_class_classification_test_instance.test_confusion_matrix()
        multi_class_classification_test_instance.test_metrics()
        multi_class_classification_test_instance.test_confidence_intervals()

    def test_bootstrap_regression(self):
        bootstrap_test_instance = TestBootstrap_Regression()
        bootstrap_test_instance.setup_class()
        bootstrap_test_instance.test_r2_score()
        bootstrap_test_instance.test_mae()
        bootstrap_test_instance.test_rmse()
        bootstrap_test_instance.test_confidence_intervals()

    def test_plotting(self):
        plotting_test_instance = TestPlots()
        plotting_test_instance.setup_class()
        plotting_test_instance.test_plot()
