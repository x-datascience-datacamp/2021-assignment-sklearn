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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
import copy


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
        # the target must be valid
        check_classification_targets(y)
        # Checks X and y for consistent length, enforces X to be 2D and y 1D
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.classes_ = np.unique(self.y_)
        self.n_features_in_ = self.X_.shape[1]

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
        # Checks if the estimator is fitted by verifying the
        # presence of fitted attributes
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        # Input validation on an array
        X = check_array(X)

        distance = pairwise_distances(X, self.X_)
        y = []

        for k in range(len(X)):
            distances = distance[k][:]
            ind = distances.argsort()
            neighbors = ind[:self.n_neighbors]
            y_neighbors = self.y_[neighbors]
            y.append(max(list(y_neighbors), key=list(y_neighbors).count))
        return np.array(y)

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
        y_hat = self.predict(X)
        n_samples = y_hat.shape[0]
        right = (y_hat == y).sum()
        return right/n_samples


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
        Xt = X.reset_index()
        if self.time_col == 'index':
            time = X.index
        else:
            time = X[self.time_col]

        if time.dtype != "datetime64[ns]":
            raise ValueError("Type of column {self.time_col} must be datetime")

        n_split = Xt[self.time_col].dt.strftime("%m/%y").nunique() - 1
        return n_split

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
        n_splits = self.get_n_splits(X, y, groups)
        Xr = copy.deepcopy(X.reset_index())
        Xr["formated_date"] = Xr[self.time_col].dt.strftime("%Y/%m")
        date = Xr["formated_date"].unique()
        date.sort()
        train = Xr["formated_date"] == date[0]
        for i in range(n_splits):
            test = Xr["formated_date"] == date[i + 1]
            idx_train = Xr.loc[train].index
            idx_test = Xr.loc[test].index
            yield (idx_train, idx_test)
