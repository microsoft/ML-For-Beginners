"""
Assesment of Generalized Estimating Equations using simulation.

This script checks Poisson models.

See the generated file "gee_poisson_simulation_check.txt" for results.
"""

import numpy as np
from statsmodels.genmod.families import Poisson
from .gee_gaussian_simulation_check import GEE_simulator
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable,Independence


class Exchangeable_simulator(GEE_simulator):
    """
    Simulate exchangeable Poisson data.

    The data within a cluster are simulated as y_i = z_c + z_i.  The
    z_c, and {z_i} are independent Poisson random variables with
    expected values e_c and {e_i}, respectively.  In order for the
    pairwise correlation to be equal to `f` for all pairs, we need

         e_c / sqrt((e_c + e_i) * (e_c + e_j)) = f for all i, j.

    By setting all e_i = e within a cluster, these equations can be
    satisfied.  We thus need

         e_c * (1 - f) = f * e,

    which can be solved (non-uniquely) for e and e_c.
    """

    scale_inv = 1.

    def print_dparams(self, dparams_est):
        OUT.write("Estimated common pairwise correlation:   %8.4f\n" %
                  dparams_est[0])
        OUT.write("True common pairwise correlation:        %8.4f\n" %
                  self.dparams[0])
        OUT.write("Estimated inverse scale parameter:       %8.4f\n" %
                  dparams_est[1])
        OUT.write("True inverse scale parameter:            %8.4f\n" %
                  self.scale_inv)
        OUT.write("\n")


    def simulate(self):

        endog, exog, group, time = [], [], [], []

        # Get a basis for the orthogonal complement to params.
        f = np.sum(self.params**2)
        u,s,vt = np.linalg.svd(np.eye(len(self.params)) -
                               np.outer(self.params, self.params) / f)
        params0 = u[:,np.flatnonzero(s > 1e-6)]

        for i in range(self.ngroups):

            gsize = np.random.randint(self.group_size_range[0],
                                      self.group_size_range[1])

            group.append([i,] * gsize)

            time1 = np.random.normal(size=(gsize, 2))
            time.append(time1)

            e_c = np.random.uniform(low=1, high=10)
            e = e_c * (1 - self.dparams[0]) / self.dparams[0]

            common = np.random.poisson(e_c)
            unique = np.random.poisson(e, gsize)
            endog1 = common + unique
            endog.append(endog1)

            lpr = np.log(e_c + e) * np.ones(gsize)

            # Create an exog matrix so that E[Y] = log(dot(exog1, params))
            exog1 = np.outer(lpr, self.params) / np.sum(self.params**2)
            emat = np.random.normal(size=(len(lpr), params0.shape[1]))
            exog1 += np.dot(emat, params0.T)

            exog.append(exog1)

        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)


class Overdispersed_simulator(GEE_simulator):
    """
    Use the negative binomial distribution to check GEE estimation
    using the overdispered Poisson model with independent dependence.

    Simulating
        X = np.random.negative_binomial(n, p, size)
    then EX = (1 - p) * n / p
         Var(X) = (1 - p) * n / p**2

    These equations can be inverted as follows:

        p = E / V
        n = E * p / (1 - p)

    dparams[0] is the common correlation coefficient
    """


    def print_dparams(self, dparams_est):
        OUT.write("Estimated inverse scale parameter:       %8.4f\n" %
                  dparams_est[0])
        OUT.write("True inverse scale parameter:            %8.4f\n" %
                  self.scale_inv)
        OUT.write("\n")


    def simulate(self):

        endog, exog, group, time = [], [], [], []

        # Get a basis for the orthogonal complement to params.
        f = np.sum(self.params**2)
        u,s,vt = np.linalg.svd(np.eye(len(self.params)) -
                               np.outer(self.params, self.params) / f)
        params0 = u[:,np.flatnonzero(s > 1e-6)]

        for i in range(self.ngroups):

            gsize = np.random.randint(self.group_size_range[0],
                                      self.group_size_range[1])

            group.append([i,] * gsize)

            time1 = np.random.normal(size=(gsize, 2))
            time.append(time1)

            exog1 = np.random.normal(size=(gsize, len(self.params)))
            exog.append(exog1)

            E = np.exp(np.dot(exog1, self.params))
            V = E * self.scale_inv

            p = E / V
            n = E * p / (1 - p)

            endog1 = np.random.negative_binomial(n, p, gsize)
            endog.append(endog1)

        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)



def gendat_exchangeable():
    exs = Exchangeable_simulator()
    exs.params = np.r_[2., 0.2, 0.2, -0.1, -0.2]
    exs.ngroups = 200
    exs.dparams = [0.3,]
    exs.simulate()
    return exs, Exchangeable()

def gendat_overdispersed():
    exs = Overdispersed_simulator()
    exs.params = np.r_[2., 0.2, 0.2, -0.1, -0.2]
    exs.ngroups = 200
    exs.scale_inv = 2.
    exs.dparams = []
    exs.simulate()
    return exs, Independence()


if __name__ == "__main__":

    np.set_printoptions(formatter={'all': lambda x: "%8.3f" % x},
                        suppress=True)

    OUT = open("gee_poisson_simulation_check.txt", "w", encoding="utf-8")

    nrep = 100

    gendats = [gendat_exchangeable, gendat_overdispersed]

    lhs = np.array([[0., 1, -1, 0, 0],])
    rhs = np.r_[0.0,]

    # Loop over data generating models
    for gendat in gendats:

        pvalues = []
        params = []
        std_errors = []
        dparams = []

        for j in range(nrep):

            da, va = gendat()
            ga = Poisson()

            # Poisson seems to be more sensitive to starting values,
            # so we run the independence model first.
            md = GEE(da.endog, da.exog, da.group, da.time, ga,
                     Independence())
            mdf = md.fit()

            md = GEE(da.endog, da.exog, da.group, da.time, ga, va)
            mdf = md.fit(start_params = mdf.params)
            if mdf is None or (not mdf.converged):
                print("Failed to converge")
                continue

            scale_inv = 1. / md.estimate_scale()
            dparams.append(np.r_[va.dparams, scale_inv])
            params.append(np.asarray(mdf.params))
            std_errors.append(np.asarray(mdf.standard_errors))

            da,va = gendat()
            ga = Poisson()

            md = GEE(da.endog, da.exog, da.group, da.time, ga, va,
                     constraint=(lhs, rhs))
            mdf = md.fit()
            if mdf is None or (not mdf.converged):
                print("Failed to converge")
                continue

            score = md.score_test_results
            pvalue = score["p-value"]
            pvalues.append(pvalue)

        dparams_mean = np.array(sum(dparams) / len(dparams))
        OUT.write("Results based on %d successful fits out of %d data sets.\n\n"
                  % (len(dparams), nrep))
        OUT.write("Checking dependence parameters:\n")
        da.print_dparams(dparams_mean)

        params = np.array(params)
        eparams = params.mean(0)
        sdparams = params.std(0)
        std_errors = np.array(std_errors)
        std_errors = std_errors.mean(0)

        OUT.write("Checking parameter values:\n")
        OUT.write("Observed:            ")
        OUT.write(np.array_str(eparams) + "\n")
        OUT.write("Expected:            ")
        OUT.write(np.array_str(da.params) + "\n")
        OUT.write("Absolute difference: ")
        OUT.write(np.array_str(eparams - da.params) + "\n")
        OUT.write("Relative difference: ")
        OUT.write(np.array_str((eparams - da.params) / da.params)
                  + "\n")
        OUT.write("\n")

        OUT.write("Checking standard errors\n")
        OUT.write("Observed:            ")
        OUT.write(np.array_str(sdparams) + "\n")
        OUT.write("Expected:            ")
        OUT.write(np.array_str(std_errors) + "\n")
        OUT.write("Absolute difference: ")
        OUT.write(np.array_str(sdparams - std_errors) + "\n")
        OUT.write("Relative difference: ")
        OUT.write(np.array_str((sdparams - std_errors) / std_errors)
                  + "\n")
        OUT.write("\n")

        pvalues.sort()
        OUT.write("Checking constrained estimation:\n")
        OUT.write("Left hand side:\n")
        OUT.write(np.array_str(lhs) + "\n")
        OUT.write("Right hand side:\n")
        OUT.write(np.array_str(rhs) + "\n")
        OUT.write("Observed p-values   Expected Null p-values\n")
        for q in np.arange(0.1, 0.91, 0.1):
            OUT.write("%20.3f %20.3f\n" %
                      (pvalues[int(q*len(pvalues))], q))

        OUT.write("=" * 80 + "\n\n")

    OUT.close()
