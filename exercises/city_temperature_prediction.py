import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # load the data set:
    data_frame = pd.read_csv(filename, parse_dates=[2]).dropna().drop_duplicates()

    # deal with invalid data:
    data_frame = data_frame[data_frame["Month"].isin(np.arange(1, 13, 1))]
    data_frame = data_frame[data_frame["Day"].isin(np.arange(1, 32, 1))]
    data_frame = data_frame[data_frame["Temp"] < 50]
    data_frame = data_frame[data_frame["Temp"] > -40]

    # Add a `DayOfYear` column based on the `Date` column:
    data_frame["DayOfYear"] = [date.day_of_year for date in data_frame["Date"]]

    return data_frame.drop(columns="Date")

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    subset = df[df["Country"] == "Israel"]
    colors = ['aliceblue', 'aqua', 'aquamarine', 'darkturquoise', 'ivory', 'khaki', 'lavender', 'lavenderblush',
              'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan']
    i = 0
    for year in range(1995, 2008):
        year_d = subset[subset["Year"] == year]
        plt.scatter(year_d.DayOfYear, year_d.Temp, c=colors[i], label=year)
        plt.title("Israel Temperature")
        plt.xlabel("Day Of Year")
        plt.ylabel("Temperature")
        i += 1

    plt.legend(bbox_to_anchor=(0.95, 1))
    plt.show()

    by_month = subset.groupby(by="Month").Temp.std()
    plt.bar(range(1,13), by_month)
    plt.ylabel("std")
    plt.xlabel("Month")
    plt.title("std as a function of month")

    plt.show()

    # Question 3 - Exploring differences between countries
    country_std = df.groupby(by=["Month", "Country"]).Temp.agg(['mean', 'std']).reset_index()
    fig = px.line(country_std, x="Month", y="mean", error_y="std", color="Country", labels="mean with std error",
                  title="Average monthly temperature")
    fig.show()


    # Question 4 - Fitting model for different values of `k`

    train_X, train_y, test_X, test_y = split_train_test(subset["DayOfYear"], subset["Temp"])
    losses = []
    Ks = list(range(1, 11))
    for k in Ks:
        polynomial = PolynomialFitting(k)
        polynomial.fit(train_X, train_y)
        loss = polynomial.loss(test_X, test_y)
        losses.append(loss)
        loss = round(loss, 2)
        form = "{:.2f}"
        print(f"Loss for k = {str(k)} is: {form.format(loss)}")

    plt.bar(Ks, losses, color="darkturquoise")
    plt.ylabel("test loss")
    plt.xlabel("polynomial degree")
    plt.title("loss as function of polynomial degree")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    polynomial = PolynomialFitting(5)  # according to Q4 - best test loss when k = 5
    polynomial.fit(subset["DayOfYear"], subset["Temp"])
    losses = []
    countries = list(set(df["Country"].values))
    for country in countries:
        cur_country = df[df["Country"] == country]
        loss = polynomial.loss(cur_country["DayOfYear"], cur_country["Temp"])
        losses.append(loss)

    plt.bar(countries, losses, color="darkturquoise")
    plt.ylabel("test loss")
    plt.xlabel("polynomial degree")
    plt.title("loss with degree 5")
    plt.show()
