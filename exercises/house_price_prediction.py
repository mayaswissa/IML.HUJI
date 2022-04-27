from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # load the data set:
    df = pd.read_csv(filename).dropna().drop_duplicates()

    # drop insignificant information
    drop_list = ["zipcode", "id", "long", "date", "sqft_lot15"]
    for f in drop_list:
        df = df.drop([f], axis=1)

    # convert strings to nums:
    pd.get_dummies(df)

    # handle Nan:
    df.fillna(df.mean())

    df = df.drop(df[df.price <= 0].index)

    # drop houses with valid information:
    drop_list = ["price", "floors", "sqft_living", "sqft_living15", "sqft_lot"]
    for f in drop_list:
        df = df[df[f] > 0]

    drop_list = ["bedrooms", "bathrooms", "view", "sqft_basement", "sqft_above"]
    for f in drop_list:
        df = df[df[f] >= 0]

    df = df[df['view'].isin(np.arange(0, 5, 1))]
    df = df[df["condition"].isin(np.arange(0, 6, 1))]
    df.drop(df[(df["sqft_above"] <= 0) & (df["sqft_basement"] <= 0)].index)

    df = df[df['grade'].isin(np.arange(0, 14, 1))]

    labels = df["price"]
    features = df.drop(columns="price")
    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
        Create scatter plot between each feature and the response.
            - Plot title specifies feature name
            - Plot title specifies Pearson Correlation between feature and response
            - Plot saved under given folder with file name including feature name
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Design matrix of regression problem

        y : array-like of shape (n_samples, )
            Response vector to evaluate against

        output_path: str (default ".")
            Path to folder in which plots are saved
        """

    for (feature, feature_data) in X.iteritems():
        div = np.std(X[feature]) * np.std(y)
        cov = np.cov(X[feature], y)
        pc = (cov / div)[1][0]

        plt.title(f"Pearson's Correlation between them is: {pc}", fontsize=10)
        plt.suptitle(f"Pearson's Correlation between {feature} values and response", fontsize=12)
        plt.xlabel(f"Feature: {feature}")
        plt.ylabel("Response")
        plt.scatter(feature_data, y, color="darkturquoise")
        plt.savefig(f"{output_path} cor_btw_res_and_{feature}.png")
        plt.show()
        plt.clf()
        plt.cla()




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    confidence_low = []
    confidence_high = []
    means = []
    train_X["price"] = train_y
    Ps = list(range(10, 101))
    for p in Ps:

        losses = []
        for i in range(10):
            train_X_sample = train_X.sample(frac=p / 100.0)

            linear_regression = LinearRegression(include_intercept=True)

            linear_regression.fit(train_X_sample.drop(columns="price"), train_X_sample.price)

            loss = linear_regression.loss(test_X.to_numpy(), test_y.to_numpy())

            losses.append(loss)
        std = np.std(losses)
        confidence_low.append(np.mean(losses) - std * 2)
        confidence_high.append(np.mean(losses) + std * 2)
        means.append(np.mean(losses))

    plt.plot(Ps, means, c="aquamarine")
    plt.plot(Ps, confidence_low, c="darkturquoise")
    plt.plot(Ps, confidence_high, c="darkturquoise")
    plt.fill_between(Ps, confidence_low, confidence_high, color="aliceblue")
    plt.xlabel("% of training data used")
    plt.ylabel("test loss")
    plt.title("test loss as function of training data size")
    plt.show()
