import numpy as np
from scipy.stats.distributions import norm


def generate_logistic():

    # Number of clusters
    nclust = 100

    # Regression coefficients
    beta = np.array([1, -2, 1], dtype=np.float64)

    # Covariate correlations
    r = 0.4

    # Cluster effects of covariates
    rx = 0.5

    # Within-cluster outcome dependence
    re = 0.3

    p = len(beta)

    OUT = open("gee_logistic_1.csv", "w", encoding="utf-8")

    for i in range(nclust):

        n = np.random.randint(3, 6)  # Cluster size

        x = np.random.normal(size=(n, p))
        x = rx*np.random.normal() + np.sqrt(1-rx**2)*x
        x[:, 2] = r*x[:, 1] + np.sqrt(1-r**2)*x[:, 2]
        pr = 1/(1+np.exp(-np.dot(x, beta)))
        z = re*np.random.normal() +\
            np.sqrt(1-re**2)*np.random.normal(size=n)
        u = norm.cdf(z)
        y = 1*(u < pr)

        for j in range(n):
            OUT.write("%d, %d," % (i, y[j]))
            OUT.write(",".join(["%.3f" % b for b in x[j, :]]) + "\n")

    OUT.close()


def generate_linear():

    # Number of clusters
    nclust = 100

    # Regression coefficients
    beta = np.array([1, -2, 1], dtype=np.float64)

    # Within cluster covariate correlations
    r = 0.4

    # Between cluster covariate effects
    rx = 0.5

    # Within-cluster outcome dependence
    # re = 0.3

    p = len(beta)

    OUT = open("gee_linear_1.csv", "w", encoding="utf-8")

    for i in range(nclust):

        n = np.random.randint(3, 6)  # Cluster size

        x = np.random.normal(size=(n, p))
        x = rx*np.random.normal() + np.sqrt(1-rx**2)*x
        x[:, 2] = r*x[:, 1] + np.sqrt(1-r**2)*x[:, 2]
        # TODO: should `e` be used somewhere?
        # e = np.sqrt(1-re**2)*np.random.normal(size=n) + re*np.random.normal()
        y = np.dot(x, beta) + np.random.normal(size=n)

        for j in range(n):
            OUT.write("%d, %d," % (i, y[j]))
            OUT.write(",".join(["%.3f" % b for b in x[j, :]]) + "\n")

    OUT.close()


def generate_nested_linear():

    # Number of clusters (clusters have 10 values, partitioned into 2
    # subclusters of size 5).
    nclust = 200

    # Regression coefficients
    beta = np.array([1, -2, 1], dtype=np.float64)

    # Top level cluster variance component
    v1 = 1

    # Subcluster variance component
    v2 = 0.5

    # Error variance component
    v3 = 1.5

    p = len(beta)

    OUT = open("gee_nested_linear_1.csv", "w", encoding="utf-8")

    for i in range(nclust):

        x = np.random.normal(size=(10, p))
        y = np.dot(x, beta)

        y += np.sqrt(v1)*np.random.normal()
        y[0:5] += np.sqrt(v2)*np.random.normal()
        y[5:10] += np.sqrt(v2)*np.random.normal()
        y += np.sqrt(v3)*np.random.normal(size=10)

        for j in range(10):
            OUT.write("%d, %.3f," % (i, y[j]))
            OUT.write(",".join(["%.3f" % b for b in x[j, :]]) + "\n")

    OUT.close()


def generate_ordinal():

    # Regression coefficients
    beta = np.zeros(5, dtype=np.float64)
    beta[2] = 1
    beta[4] = -1

    rz = 0.5

    OUT = open("gee_ordinal_1.csv", "w", encoding="utf-8")

    for i in range(200):

        n = np.random.randint(3, 6)  # Cluster size

        x = np.random.normal(size=(n, 5))
        for j in range(5):
            x[:, j] += np.random.normal()
        pr = np.dot(x, beta)
        pr = np.array([1, 0, -0.5]) + pr[:, None]
        pr = 1 / (1 + np.exp(-pr))

        z = rz*np.random.normal() +\
            np.sqrt(1-rz**2)*np.random.normal(size=n)
        u = norm.cdf(z)

        y = (u[:, None] > pr).sum(1)

        for j in range(n):
            OUT.write("%d, %d," % (i, y[j]))
            OUT.write(",".join(["%.3f" % b for b in x[j, :]]) + "\n")

    OUT.close()


def generate_nominal():

    # Regression coefficients
    beta1 = np.r_[0.5, 0.5]
    beta2 = np.r_[-1, -0.5]
    p = len(beta1)

    rz = 0.5

    OUT = open("gee_nominal_1.csv", "w", encoding="utf-8")

    for i in range(200):

        n = np.random.randint(3, 6)  # Cluster size

        x = np.random.normal(size=(n, p))
        x[:, 0] = 1
        for j in range(1, x.shape[1]):
            x[:, j] += np.random.normal()
        pr1 = np.exp(np.dot(x, beta1))[:, None]
        pr2 = np.exp(np.dot(x, beta2))[:, None]
        den = 1 + pr1 + pr2
        pr = np.hstack((pr1/den, pr2/den, 1/den))
        cpr = np.cumsum(pr, 1)

        z = rz*np.random.normal() +\
            np.sqrt(1-rz**2)*np.random.normal(size=n)
        u = norm.cdf(z)

        y = (u[:, None] > cpr).sum(1)

        for j in range(n):
            OUT.write("%d, %d," % (i, y[j]))
            OUT.write(",".join(["%.3f" % b for b in x[j, :]]) + "\n")

    OUT.close()


def generate_poisson():

    # Regression coefficients
    beta = np.zeros(5, dtype=np.float64)
    beta[2] = 0.5
    beta[4] = -0.5

    nclust = 100

    OUT = open("gee_poisson_1.csv", "w", encoding="utf-8")

    for i in range(nclust):

        n = np.random.randint(3, 6)  # Cluster size

        x = np.random.normal(size=(n, 5))
        for j in range(5):
            x[:, j] += np.random.normal()
        lp = np.dot(x, beta)
        E = np.exp(lp)
        y = [np.random.poisson(e) for e in E]
        y = np.array(y)

        for j in range(n):
            OUT.write("%d, %d," % (i, y[j]))
            OUT.write(",".join(["%.3f" % b for b in x[j, :]]) + "\n")

    OUT.close()
