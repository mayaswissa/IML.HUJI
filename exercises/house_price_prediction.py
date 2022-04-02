from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

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
    df = pd.read_csv(filename)

    df = df.dropna().drop_duplicates()

    # convert zipcode data to from string to int:
    df["zipcode"] = df["zipcode"].astype(int)

    # drop id (uniques) - not meaningful data:
    df = df.drop(["id"], asix=1)

    # handle Nan:
    pd.get_dummies(df)

    # handle empty cells:
    df.fillna(df.mean())  # todo

    # todo handle date representation

    # todo handle negative price? Can a living room size be too small?
    df = df.drop(df[df.price <= 0].index)  # todo how to manipulate the price if this is what we need to predict

    # valid information:
    df = df.drop(df[df["price"] <= 0])
    df = df.drop(df[df["floors"] <= 0])
    df = df.drop(df[df["sqft_liv"] <= 0])
    df = df.drop(df[df["sqft_lot"] <= 0])

    df = df.drop(df[df["bedrooms"] < 0])
    df = df.drop(df[df["bathrooms"] < 0])
    df = df.drop(df[df["sqft_basement"] < 0])
    df = df.drop(df[~df["view"].isin([0, 1, 2, 3, 4])])
    df = df.drop(df[~df["condition"].isin([1, 2, 3, 4, 5])])
    df = df.drop(df[df["view"] < 0])

    df = df.drop(df[~df["sqft_lot"].isin([0, 1])])

    df.insert(0, 'intercept', 1, True)
    return df.drop("price", 1), df.price


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
    for feature in X:
        div = np.std(X[feature]) * np.std(y)
        cov = np.cov(X[feature], y)[0, 1]
        pc = cov / div
        plt = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}),
                         x="x", y="y", trendline="ols",
                         title="The correlation between " + feature + "values and their Pearson Correlation " + pc,
                         labels={"x": feature + " Values", "y": "Response values"})
        plt.write_image(output_path % feature)  # todo what!??!?!?!?!?!??!?!?!?


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
