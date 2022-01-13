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
november 2020 to march 2021, you have have 4 splits. The first split
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
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances

from collections import Counter


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  
        if (isinstance(n_neighbors, int) is False):
            raise ValueError("provide an integer please")

        self.n_neighbors = n_neighbors

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = dict(zip(range(len(y)), y))
        self.classes_ = np.unique(y).tolist()

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        y_predict = np.zeros(X.shape[0])
        distance = pairwise_distances(X, self.X_, metric="euclidean")
        if distance.shape[1] >= self.n_neighbors:
            n_neighbors = self.n_neighbors
        else:
            n_neighbors = (distance.shape[0]-1)

        argmins = np.argpartition(distance,n_neighbors, axis=1)[:, :n_neighbors]

        labels = np.vectorize(self.y_.get)(argmins)
        y_predict = np.apply_along_axis(lambda x: max(Counter(x),
                                     key=np.where), 1, labels)

        return y_predict

    def score(self, X, y):

        X, y = check_X_y(X, y)
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        check_classification_targets(y)

        y_ = self.predict(X)
        score = np.mean(y == y_)

        return score

class MonthlySplit(BaseCrossValidator):

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def _get_date_column(self, X):

        if (self.time_col != 'index') and (self.time_col in X.columns):
            date_column = X[self.time_col]
        elif self.time_col == 'index':
            date_column = X.reset_index()['index']
        else:
            raise Exception

        return date_column

    def _get_combinations(self, X):

        date_column = self._get_date_column(X)

        month_year = list(
            zip(date_column.dt.year.values.tolist(),
                date_column.dt.month.values.tolist()))
        month_year_unique = list(set(month_year))
        month_year_unique.sort()

        return month_year_unique

    def get_n_splits(self, X, y=None, groups=None):
 
        month_year_unique = self._get_combinations(X)

        n_splits = len(month_year_unique)-1

        return n_splits

    def split(self, X, y, groups=None):

        n_splits = self.get_n_splits(X, y, groups)
        combinations = self._get_combinations(X)

        date_column = self._get_date_column(X)
        year_col = date_column.dt.year
        month_col = date_column.dt.month

        for i in range(n_splits):
            year, month = combinations[i]
            next_year, next_month = combinations[i+1]
            idx_train = np.where((month_col == month) & (year_col == year))
            idx_test = np.where((month_col == next_month) & (year_col == next_year))
            yield (idx_train, idx_test)