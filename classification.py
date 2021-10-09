#!/usr/bin/env python3
import logging
import math
import math
import sys
from collections import defaultdict
from collections import defaultdict
from itertools import product

import numba
import numpy as np
import pandas as pd
from numba import prange, jit
from numba import types
from numba.typed import List
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target


estimates = defaultdict(lambda: defaultdict(list))
intervals = defaultdict(lambda: defaultdict(list))

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


@jit(nopython=True, cache=True)
def confusion_matrix(y_true, y_pred, labels, n_labels):
    result = np.zeros((n_labels, n_labels), dtype=np.int32)

    for i in range(len(y_true)):
        result[y_true[i]][y_pred[i]] += 1

    return result


@jit(nopython=True, cache=True)
def metrics_from_confusion_matrix(confusion_matrix, classes):
    metrics = List()
    zero_matrix = np.zeros(confusion_matrix.shape, dtype=numba.int32)

    for i in range(len(classes)):
        tp = confusion_matrix[i, i]

        # FN = row i - TP
        fn_mask = np.copy(zero_matrix)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion_matrix, fn_mask))

        # FP = column i - TP
        fp_mask = np.copy(zero_matrix)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion_matrix, fp_mask))

        # TN = everything - TP - FN - FP
        # tn_mask = 1 - (fn_mask + fp_mask)
        # tn_mask[i, i] = 0  # for TP
        # tn = np.sum(np.multiply(confusion_matrix, tn_mask))

        # TPR = TP/(TP+FN)
        recall = precision = f1_score = 0
        sum_1 = tp + fn
        if sum_1 > 0:
            recall = tp / sum_1

        # FPR = FP/(FP+TN)
        # fpr = fp / (fp + tn)

        sum_2 = tp + fp
        if sum_2 > 0:
            precision = tp / sum_2

        if precision > 0 and recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)

        metrics.append(precision)
        metrics.append(recall)
        metrics.append(f1_score)

    return metrics


@jit(nopython=True, cache=True)
def reporter(df, df_length, iteration, labels, n_labels, all_results):
    # bootstrap samples
    df_local = df[np.random.choice(df_length, size=df_length, replace=True), :]
    matrix = confusion_matrix(
        df_local[:, 0], df_local[:, 1], labels=labels, n_labels=n_labels
    )
    metrics = metrics_from_confusion_matrix(matrix, labels)
    col = 0
    for value in metrics:
        all_results[iteration][col] = value
        col += 1


@jit(nopython=True, cache=True)
def get_actual_metrics(df, df_length, labels, n_labels, true_results):
    # bootstrap samples
    matrix = confusion_matrix(df[:, 0], df[:, 1], labels=labels, n_labels=n_labels)
    metrics = metrics_from_confusion_matrix(matrix, labels)
    col = 0
    for value in metrics:
        true_results[col] = value
        col += 1


@jit(nopython=True, parallel=True, cache=True)
def run(df, df_length, n, labels, n_labels, headers, start_index, end_index):
    """
    fastest way to do this is to just create a numpy matrix of results ie a dataframe
    append every row of results to the dataframe, as each row will have same headers
    just define class headers:
    classA_precision, classA_recall, classB_precision, etc.
    then calculating inervals is just column-by-column sort, can be done in parallel
    """
    # we know ahead of time the shape of the final matrix
    all_results = np.zeros((n, len(headers)), dtype=types.float64)
    true_results = np.zeros(len(headers), dtype=types.float64)
    for i in prange(n):
        reporter(df, df_length, i, labels, n_labels, all_results)
    get_actual_metrics(df, df_length, labels, n_labels, true_results)
    # we can't do list appends or dictionary assignments in parallel safetly in numba (due to thread safety issues)
    # instead, we pre-allocate a matrix of len(headers)
    intervals = np.zeros((len(headers), 3), dtype=types.float64)
    for i in prange(len(headers)):
        column = all_results[:, i]
        column.sort()  # in-place
        intervals[i][0] = column[start_index]
        intervals[i][1] = column[end_index]
        intervals[i][2] = true_results[i]

    return intervals


def create_numba_list(py_list):
    numba_list = List()
    for item in py_list:
        numba_list.append(item)
    return numba_list


def label_counts(labels):
    """
    return a dictioary of label: count pairs

    we're running an old version of numpy, so we can't do:
    np.array(np.unique(labels, return_counts=True)).T
    when it's available, add back & add the numba decorator
    """
    data = pd.value_counts(labels)
    return data.to_dict()


def classification_report(
    label,
    predicted,
    n=1000,
    confidence_level=95,
    as_dict=False,
    sort_by_sample_size=False,
):
    """
    n: number of bootstrap iterations
    confidence_level: 90, 95, etc., between 1 & 100
    as_dict: return a nested dictionary
    sorted: return the Pandas dataframe, sorted in descending order of class sample size
    """
    assert confidence_level > 0 and confidence_level < 100
    assert (type_of_target(label) == "binary") or (
        type_of_target(label) == "multiclass"
    )
    assert (type_of_target(predicted) == "binary") or (
        type_of_target(predicted) == "multiclass"
    )
    check_consistent_length(label, predicted)

    df = pd.DataFrame({"label": label, "predicted": predicted})
    df["id"] = np.arange(len(df))
    df_length = len(df)

    labels = create_numba_list(np.unique(df["label"]))
    n_labels = len(labels)
    mapping_df = pd.DataFrame({"class": labels, "value": range(len(labels))})
    df_label_encoded = df.merge(mapping_df, left_on="label", right_on="class")
    df_predicted_encoded = df.merge(mapping_df, left_on="predicted", right_on="class")
    df_encoded = df_label_encoded.merge(
        df_predicted_encoded, on="id", suffixes=["_label", "_predicted"]
    )
    df_encoded_np = df_encoded[["value_label", "value_predicted"]].to_numpy()

    # we use the caret character as a separator, so we first verify none of labels contain it
    for label in labels:
        assert "^" not in str(
            label
        ), "please remove the '^' character from your labels, it's used as a delimiter in fronni"
    headers = [
        str(item[0]) + "^" + str(item[1])
        for item in product(labels, ["precision", "recall", "f1_score"])
    ]
    delta = (1 - (confidence_level / 100)) / 2
    start_index = math.floor(n * delta)
    end_index = math.floor(n * (1 - delta))

    results = run(
        df_encoded_np,
        df_length,
        n,
        labels,
        n_labels,
        create_numba_list(headers),
        start_index,
        end_index,
    )

    label_count_dict = label_counts(df["label"].to_numpy())
    dict_results = defaultdict(dict)
    for index, row in enumerate(results):
        header_parts = headers[index].split("^")
        class_name = header_parts[0]
        metric_name = header_parts[1]
        dict_results[class_name][metric_name] = {
            "true": row[2],
            "lower": row[0],
            "upper": row[1],
            "sample_size": label_count_dict.get(class_name),
        }
    if sort_by_sample_size:
        dict_results = dict(
            sorted(
                dict_results.items(),
                key=lambda kv: kv[1]["precision"]["sample_size"],
                reverse=True,
            )
        )
    if as_dict:
        return dict_results
    # otherwise, return the results as a pandas dataframe that can be printed out
    formatted_results = []
    for class_name, metrics in dict_results.items():
        for metric_name, metric_values in metrics.items():
            class_metric_dict = {"class": class_name, "metric": metric_name}
            formatted_results.append(
                {**class_metric_dict, **metric_values}  # combine dicts
            )
    df = pd.DataFrame(formatted_results)
    return df


def plot_classification_report(report, save_to_filename=None):
    """
    save_to_filename: "image.png", "image.pdf", etc.
    documentation here: https://plotly.com/python/error-bars/
    """
    import plotly.graph_objects as go

    report = report.iloc[::-1]  # reverse order, in case 'sort_by_sample_size' was true

    precision_df = report.query("metric=='precision'")
    recall_df = report.query("metric=='recall'")
    f1_df = report.query("metric=='f1_score'")
    classes = f1_df["class"].tolist()
    error_width = (f1_df["upper"] - f1_df["lower"]).tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(f1_df["true"].to_numpy()),
            y=classes,
            marker_color="blue",
            text=list(f1_df["true"].to_numpy()),
            mode="markers",  # no connecting lines
            error_x=dict(
                type="data",  # value of error bar given in data coordinates
                array=error_width,
                color="blue",
                width=8,
                visible=True,
            ),
            name="F1 score",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(precision_df["true"].to_numpy()),
            y=classes,
            marker_color="green",
            text=list(precision_df["true"].to_numpy()),
            mode="markers",
            name="precision",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(recall_df["true"].to_numpy()),
            y=classes,
            marker_color="red",
            text=list(recall_df["true"].to_numpy()),
            mode="markers",
            name="recall",
        )
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(title_text="Class")
    fig.update_yaxes(
        type="category"
    )  # force it to be categorical so we don't have tick marks

    fig.update_layout(
        autosize=False,
        width=1200,
        height=len(classes) * 20,
        margin=dict(l=50, r=50, b=5, t=5, pad=4),
        paper_bgcolor="white",
    )

    fig.show()
    if save_to_filename:  # requires orca or kaleido installed
        fig.write_image(save_to_filename)
