from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The * by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        th_values = []
        er = []
        signs = []

        for j in range(X.shape[1]):
            pos_th, pos_er = self._find_threshold(X[:, j], y, 1)
            neg_th, neg_er = self._find_threshold(X[:, j], y, -1)
            if pos_er < neg_er:
                th_values.append(pos_th)
                er.append(pos_er)
                signs.append(1)
            else:
                th_values.append(neg_th)
                er.append(neg_er)
                signs.append(-1)

        j = np.argmin(er)
        self.j_ = j
        self.sign_ = signs[j]
        self.threshold_ = th_values[j]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        sorted_values = np.array(sorted(values))
        v_prev = values[0]
        labels_sign = np.sign(labels)
        th_values = [float("-inf")]
        false = (labels_sign != sign)
        not_false = ~ false
        losses = [np.sum(np.abs(false * labels))]

        for v_curr in sorted_values[1:]:
            th = (v_prev + v_curr) / 2
            higher = values >= th
            higher_false = higher & false
            lower_false = (~ higher) & not_false
            error = np.sum(np.abs(labels[higher_false])) + np.sum(np.abs(labels[lower_false]))
            losses.append(error)
            th_values.append(th)
            v_prev = v_curr

        th_values.append(float("inf"))
        losses.append(np.sum(np.abs(not_false * labels)))
        ind = np.argmin(losses)

        return th_values[ind], losses[ind]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
