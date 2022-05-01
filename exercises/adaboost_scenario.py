import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adb = AdaBoost(DecisionStump, n_learners)
    adb.fit(train_X, train_y)
    train_losses, test_losses = np.zeros(n_learners), np.zeros(n_learners)

    for i in range(1, n_learners + 1):
        train_losses[i - 1] = adb.partial_loss(train_X, train_y, i)
        test_losses[i - 1] = adb.partial_loss(test_X, test_y, i)
    plt.plot(range(n_learners), train_losses, label="Train")
    plt.plot(range(n_learners), test_losses, label="Test")
    plt.xlabel("Number of fitted learners")
    plt.ylabel("Loss")
    plt.title(f"AdaBoost Train & test errors as a function of number of fitted learners\n noise = {noise}")
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    l = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} Learners" for i in T],
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    m = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, name="Label 1",
                   marker=dict(color=(test_y == 1).astype(int), symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda x: adb.partial_predict(x, t), l[0], l[1], showscale=False), m],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(width=800, height=900,
                      title=f"AdaBoost Decision boundaries based on number of learners\n noise={noise}",
                      margin=dict(t=100))
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.write_image(f"AdaBoostDecisionBoundaries {noise}.png")

    # Question 3: Decision surface of best performing ensemble
    t_min = np.argmin(test_losses) + 1
    acc = np.round(1 - test_losses[t_min], 3)
    fig = go.Figure([decision_surface(lambda x: adb.partial_predict(x, t_min), l[0], l[1], showscale=False), m])
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(title_text=f"Ensemble with lowest test error \n ensemble={t_min}, maximal accuracy= {acc}")
    fig.write_image(f"AdaBoostLowestBoundaryNoise {noise}.png")

    # Question 4: Decision surface with weighted samples
    m = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=(train_y == 1).astype(int),
                               size=adb.D_ / np.max(adb.D_) * 5,
                               symbol=class_symbols[train_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))

    fig = go.Figure([decision_surface(adb.predict, l[0], l[1], showscale=False), m])
    fig.update_xaxes(range=[-1, 1], constrain="domain")
    fig.update_yaxes(range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(dict1=dict(width=600, height=600,
                                 title=f"Adaboost decision surface with weighted samples \n noise= {noise}"))
    fig.write_image(f"AdaBoostWeightedSamplesNoise {noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)