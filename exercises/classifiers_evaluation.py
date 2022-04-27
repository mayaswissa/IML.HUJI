from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    samples, labels, = [], []
    losses = []

    def callback(fit: Perceptron, x: np.ndarray, y: int):
        losses.append((fit.loss(samples, np.array(labels))))

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        [samples, labels] = load_dataset("../datasets/" + f)
        losses = []

        # Fit Perceptron and record loss in each fit iteration
        perceptron = Perceptron(callback=callback)
        perceptron.fit(samples, labels)

        # Plot figure of loss as function of fitting iteration
        plt.title(f"Loss as function of fitting iteration - {n}")
        plt.xlabel("Training iterations")
        plt.ylabel("Training loss")
        plot_range = range(len(losses))
        plt.plot(plot_range, losses, color="darkturquoise")
        plt.savefig(f"Loss as function of fitting iteration - {n}.png")
        plt.show()
        plt.clf()
        plt.cla()




def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        [samples, labels] = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        # LDA
        lda = LDA()
        lda.fit(samples, labels)
        lda_prediction = lda.predict(samples)
        # Gaussian Naive Bayes
        gnb = GaussianNaiveBayes()
        gnb.fit(samples, labels)
        gnb_prediction = gnb.predict(samples)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        accur_lda = str(accuracy(labels, lda_prediction))
        accur_gnb = str(accuracy(labels, gnb_prediction))
        titles = [f"{f} LDA accuracy: {accur_lda}",
                  f"{f} GNB accuracy:  {accur_gnb}"]

        models = [lda, gnb]
        predictions = [lda_prediction, gnb_prediction]

        fig = make_subplots(subplot_titles=titles, cols=2)

        # Add traces for data-points setting symbols and colors
        # Add `X` dots specifying fitted Gaussians' means

        for i in range(len(models)):
            fig.add_traces(go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers", showlegend=False,
                                      marker=dict(symbol=np.uint32(labels).tolist(), color=np.uint32(predictions[i]),
                                                  line=dict(color=labels, width=1)),
                                      text=[f"True class: {str(labels[i])}, Predicted class: {str(predictions[0][i])}"
                                            for i in range(samples.shape[0])], hovertemplate='%{text}'), rows=1,
                           cols=i + 1)
            fig.add_traces(go.Scatter(x=models[i].mu_[:, 0], y=models[i].mu_[:, 1],
                                      marker=dict(symbol='x', color='black', size=10),
                                      mode='markers', showlegend=False), rows=1, cols=i + 1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            elipse_lda = get_ellipse(lda.mu_[i, :], lda.cov_)
            elipse_gnb = get_ellipse(gnb.mu_[i, :], np.diag(gnb.vars_[i]))
            fig.add_traces(elipse_lda, rows=1, cols=1)
            fig.add_traces(elipse_gnb, rows=1, cols=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
