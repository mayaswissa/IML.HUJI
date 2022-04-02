from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    vg = UnivariateGaussian()
    fit = vg.fit(samples)
    print('(' + str(fit.mu_) + ', ' + str(fit.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent

    distances = np.zeros(100)
    index = 0
    for i in range(10, 1000, 10):
        distances[index] = np.abs(np.mean(samples[0:i]) - vg.mu_)
        index += 1

    plt.scatter(range(10, 1001, 10), distances)
    plt.xlabel("samples")
    plt.ylabel("differences")
    plt.title("Distance between the Estimated and True Expectation values")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    plt.scatter(samples, vg.pdf(samples))
    plt.ylabel("PDF")
    plt.xlabel("SAMPLES")
    plt.title("Empirical PDF of samples")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    samples = np.random.multivariate_normal(mu, sigma, 1000)
    vg = MultivariateGaussian()
    fit = vg.fit(samples)
    print("estimated expectation: \n" + str(fit.mu_))
    print("covariance matrix: \n" + str(fit.cov_))

    # Question 5 - Likelihood evaluation
    likelihood_matrix = np.zeros((200, 200))
    f_vals = []
    form = "{:.3f}"
    for i, f1 in enumerate(np.linspace(-10, 10, 200)):
        for j, f3 in enumerate(np.linspace(-10, 10, 200)):
            l = MultivariateGaussian.log_likelihood(mu=np.transpose(np.array([f1, 0, f3, 0])),
                                                    cov=sigma, X=samples)
            likelihood_matrix[i, j] = l

            f_vals.append((form.format(f1), form.format(f3)))

    import plotly.express as px

    plot = px.imshow(likelihood_matrix, x=np.linspace(-10, 10, 200),
                     labels=dict(x="f3", y="f1", color="Log-likelihood"),
                     y=np.linspace(-10, 10, 200))
    plot.update_layout(title_text="The Log-likelihood with expectation [f1, 0, f3, 0]")
    plot.show()

    # Question 6 - Maximum likelihood
    ind = np.argmax(likelihood_matrix)
    linspace = np.linspace(-10, 10, 200)
    print("maximum log-likelihood value: " + str(form.format(np.max(likelihood_matrix))))
    print("f1, f3 of this value: (" + str(form.format(linspace[int(ind/200)])) + ', ' + str(form.format(linspace[int(ind % 200)])) + ')')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


