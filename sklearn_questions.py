"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `check_*` functions imported in the file.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.

Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 5 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
# import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

         Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            training data.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X_val, y_val = check_X_y(X, y)
        check_classification_targets(y_val)
        self.classes_ = unique_labels(y_val)
        self.X_train_ = X_val
        self.y_train_ = y_val
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Test data to predict on.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Class labels for each test data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.zeros(X.shape[0])
        y_pred = np.array(y_pred, dtype=type(self.y_train_[0]))
        K = self.n_neighbors
        for i in range(X.shape[0]):
            dist_neigh = []
            tab = np.concatenate(([X[i]], self.X_train_), axis=0)
            dist_neigh.append(pairwise_distances(tab)[0, :])
            id_nn = np.array(dist_neigh)[0, 1:].argsort()[:K]
            id_nn = np.array(id_nn)
            unique_vals = np.unique(self.y_train_[id_nn])
            idx_max = unique_vals[0]
            maxi = 0
            for elem in unique_vals:
                if (self.y_train_[id_nn] == elem).sum() >= maxi:
                    maxi = (self.y_train_[id_nn] == elem).sum()
                    idx_max = elem
            y_pred[i] = idx_max
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            training data.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        score = np.mean(self.predict(X) == y)
        return score


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if self.time_col == 'index':
            x = X.reset_index()
            time_col = self.time_col
            n_splits = x[time_col].dt.to_period('M').nunique()-1
        else:
            time_col = self.time_col
            n_splits = X[time_col].dt.to_period('M').nunique()-1
        return n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        """
        n_samples = X.shape[0]
        n_splits = self.get_n_splits(X, y, groups)
        for i in range(n_splits):
            idx_train = range(n_samples)
            idx_test = range(n_samples)
            yield (
                idx_train, idx_test
            )
        """
        if not(is_datetime64_ns_dtype(X.reset_index()[self.time_col])):
            raise ValueError('datetime')
        n_splits = self.get_n_splits(X, y, groups)
        for i in range(n_splits):
            idx_train = range(i+1)
            idx_test = range(i+1, i+2)
            yield (
                idx_train, idx_test
            )
