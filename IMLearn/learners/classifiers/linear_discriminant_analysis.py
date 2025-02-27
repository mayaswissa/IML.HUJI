from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ = np.unique(y)
        self.mu_ = np.zeros([X.shape[1], len(self.classes_)])
        self.pi_ = np.zeros(len(self.classes_))
        self.cov_ = np.zeros([X.shape[1], X.shape[1]])
        for k, class_ in enumerate(self.classes_):
            self.mu_[:, k] = np.mean(X[class_ == y], axis=0)
            self.pi_[k] = len(y[class_ == y])
            self.cov_ += np.transpose(X[class_ == y] - self.mu_[:, class_]) @ (X[class_ == y] - self.mu_[:, class_])

        self.pi_ = self.pi_ / len(y)
        self.mu_ = np.transpose(self.mu_)
        self.cov_ = np.transpose(self.cov_) / (len(y) - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros([X.shape[0], len(self.classes_)])
        log_pi = np.log(self.pi_)
        for index, x in enumerate(X):
            for k, class_ in enumerate(self.classes_):
                likelihoods[index, k] = np.transpose(self._cov_inv @ self.mu_[k]) @ x + log_pi[k] -\
                                        (1/2) * self.mu_[k] @ self._cov_inv @ self.mu_[k]
        return likelihoods

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
        from ...metrics import misclassification_error

        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)