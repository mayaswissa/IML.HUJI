from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros([X.shape[1], len(self.classes_)])
        for k, class_ in enumerate(self.classes_):
            self.mu_[:, k] = np.mean(X[class_ == y], axis=0)
            self.vars_[:, k] = np.var(X[class_ == y], axis=0, ddof=1)
            self.pi_[k] = len(y[class_ == y])
        self.pi_ = self.pi_ / len(y)
        self.mu_ = np.transpose(self.mu_)
        self.vars_ = np.transpose(self.vars_)

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

        l = np.zeros((X.shape[0], self.classes_.shape[0]))
        for k, class_ in enumerate(self.classes_):
            cov = np.diag(self.vars_[k])
            scalar = np.sqrt(np.linalg.det(cov) * np.power(2 * np.pi, X.shape[1]))
            x = X - self.mu_[k]
            l[:, k] = self.pi_[k] * (1 / scalar) *\
                      np.squeeze(np.exp((-0.5) * x[..., None, :] @ np.linalg.inv(cov) @ x[..., None]))

        return l



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
