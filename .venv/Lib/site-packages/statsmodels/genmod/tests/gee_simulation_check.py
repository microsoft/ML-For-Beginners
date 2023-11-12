"""
Assesment of Generalized Estimating Equations using simulation.

Only Gaussian models are currently checked.

See the generated file "gee_simulation_check.txt" for results.
"""
from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested

np.set_printoptions(formatter={'all': lambda x: "%8.3f" % x},
                    suppress=True)


OUT = open("gee_simulation_check.txt", "w", encoding="utf-8")

class GEE_simulator:

    #
    # Parameters that must be defined
    #

    # Number of groups
    ngroups = None

    # Standard deviation of the pure errors
    error_sd = None

    # The regression coefficients
    params = None

    # The parameters defining the dependence structure
    dparams = None


    #
    # Output parameters
    #

    # Matrix of exogeneous data (rows are cases, columns are
    # variables)
    exog = None

    # Matrix of endogeneous data (len(endog) = exog.shape[0])
    endog = None

    # Matrix of time information (time.shape[0] = len(endog))
    time = None

    # Group labels (len(groups) = len(endog))
    group = None

    # Group sizes are random within this range
    group_size_range = [4, 11]

    # dparams_est is dparams with scale_inv appended
    def print_dparams(self, dparams_est):
        raise NotImplementedError


class AR_simulator(GEE_simulator):

    # The distance function for determining AR correlations.
    distfun = [lambda x, y: np.sqrt(np.sum((x-y)**2)),]


    def print_dparams(self, dparams_est):
        OUT.write("AR coefficient estimate:   %8.4f\n" %
                  dparams_est[0])
        OUT.write("AR coefficient truth:      %8.4f\n" %
                  self.dparams[0])
        OUT.write("Error variance estimate:   %8.4f\n" %
                  dparams_est[1])
        OUT.write("Error variance truth:      %8.4f\n" %
                  self.error_sd**2)
        OUT.write("\n")

    def simulate(self):

        endog, exog, group, time = [], [], [], []

        for i in range(self.ngroups):

            gsize = np.random.randint(self.group_size_range[0],
                                      self.group_size_range[1])

            group.append([i,] * gsize)

            time1 = np.random.normal(size=(gsize,2))
            time.append(time1)

            exog1 = np.random.normal(size=(gsize, 5))
            exog1[:,0] = 1
            exog.append(exog1)

            # Pairwise distances within the cluster
            distances = scipy.spatial.distance.cdist(time1, time1,
                                                     self.distfun[0])

            # Pairwise correlations within the cluster
            correlations = self.dparams[0]**distances
            correlations_sr = np.linalg.cholesky(correlations)

            errors = np.dot(correlations_sr, np.random.normal(size=gsize))

            endog1 = np.dot(exog1, self.params) + errors * self.error_sd
            endog.append(endog1)

        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)



class Nested_simulator(GEE_simulator):

    # Vector containing list of nest sizes (used instead of
    # group_size_range).
    nest_sizes = None

    # Matrix of nest id's (an output parameter)
    id_matrix = None


    def print_dparams(self, dparams_est):
        for j in range(len(self.nest_sizes)):
            OUT.write("Nest %d variance estimate:  %8.4f\n" % \
                          (j+1, dparams_est[j]))
            OUT.write("Nest %d variance truth:     %8.4f\n" % \
                          (j+1, self.dparams[j]))

        OUT.write("Error variance estimate:   %8.4f\n" % \
            (dparams_est[-1] - sum(dparams_est[0:-1])))
        OUT.write("Error variance truth:      %8.4f\n" %
                  self.error_sd**2)
        OUT.write("\n")


    def simulate(self):

        group_effect_var = self.dparams[0]

        vcomp = self.dparams[1:]
        vcomp.append(0)

        endog, exog, group, id_matrix = [], [], [], []

        for i in range(self.ngroups):

            iterators = [lrange(n) for n in self.nest_sizes]

            # The random effects
            variances = [np.sqrt(v)*np.random.normal(size=n)
                         for v,n in zip(vcomp, self.nest_sizes)]

            gpe = np.random.normal() * np.sqrt(group_effect_var)

            nest_all = []
            for j in self.nest_sizes:
                nest_all.append(set())

            for nest in product(*iterators):

                group.append(i)

                # The sum of all random effects that apply to this
                # unit
                ref = gpe + sum([v[j] for v,j in zip(variances, nest)])

                exog1 = np.random.normal(size=5)
                exog1[0] = 1
                exog.append(exog1)

                error = ref + self.error_sd * np.random.normal()

                endog1 = np.dot(exog1, self.params) + error
                endog.append(endog1)

                for j in range(len(nest)):
                    nest_all[j].add(tuple(nest[0:j+1]))

                nest1 = [len(x)-1 for x in nest_all]
                id_matrix.append(nest1[0:-1])

        self.exog = np.array(exog)
        self.endog = np.array(endog)
        self.group = np.array(group)
        self.id_matrix = np.array(id_matrix)
        self.time = np.zeros_like(self.endog)







def check_constraint(da, va, ga):
    """
    Check the score testing of the parameter constraints.
    """





def gen_gendat_ar0(ar):
    def gendat_ar0(msg = False):
        ars = AR_simulator()
        ars.ngroups = 200
        ars.params = np.r_[0, -1, 1, 0, 0.5]
        ars.error_sd = 2
        ars.dparams = [ar,]
        ars.simulate()
        return ars, Autoregressive()
    return gendat_ar0

def gen_gendat_ar1(ar):
    def gendat_ar1():
        ars = AR_simulator()
        ars.ngroups = 200
        ars.params = np.r_[0, -0.8, 1.2, 0, 0.5]
        ars.error_sd = 2
        ars.dparams = [ar,]
        ars.simulate()
        return ars, Autoregressive()
    return gendat_ar1

def gendat_nested0():
    ns = Nested_simulator()
    ns.error_sd = 1.
    ns.params = np.r_[0., 1, 1, -1, -1]
    ns.ngroups = 50
    ns.nest_sizes = [10, 5]
    ns.dparams = [2., 1.]
    ns.simulate()
    return ns, Nested(ns.id_matrix)

def gendat_nested1():
    ns = Nested_simulator()
    ns.error_sd = 2.
    ns.params = np.r_[0, 1, 1.3, -0.8, -1.2]
    ns.ngroups = 50
    ns.nest_sizes = [10, 5]
    ns.dparams = [1., 3.]
    ns.simulate()
    return ns, Nested(ns.id_matrix)


nrep = 100

gendats = [gen_gendat_ar0(ar) for ar in (0, 0.3, 0.6)]
gendats.extend([gen_gendat_ar1(ar) for ar in (0, 0.3, 0.6)])
gendats.extend([gendat_nested0, gendat_nested1])

lhs = np.array([[0., 1, 1, 0, 0],])
rhs = np.r_[0.,]

# Loop over data generating models
for gendat in gendats:

    pvalues = []
    params = []
    std_errors = []
    dparams = []

    for j in range(nrep):

        da,va = gendat()
        ga = Gaussian()

        md = GEE(da.endog, da.exog, da.group, da.time, ga, va)
        mdf = md.fit()

        scale_inv = 1 / md.estimate_scale()
        dparams.append(np.r_[va.dparams, scale_inv])
        params.append(np.asarray(mdf.params))
        std_errors.append(np.asarray(mdf.standard_errors))

        da,va = gendat()
        ga = Gaussian()

        md = GEE(da.endog, da.exog, da.group, da.time, ga, va,
                 constraint=(lhs, rhs))
        mdf = md.fit()
        score = md.score_test_results
        pvalue = score["p-value"]
        pvalues.append(pvalue)

    dparams_mean = np.array(sum(dparams) / len(dparams))
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
    OUT.write(np.array_str((eparams - da.params) / da.params) + "\n")
    OUT.write("\n")

    OUT.write("Checking standard errors\n")
    OUT.write("Observed:            ")
    OUT.write(np.array_str(sdparams) + "\n")
    OUT.write("Expected:            ")
    OUT.write(np.array_str(std_errors) + "\n")
    OUT.write("Absolute difference: ")
    OUT.write(np.array_str(sdparams - std_errors) + "\n")
    OUT.write("Relative difference: ")
    OUT.write(np.array_str((sdparams - std_errors) / std_errors) + "\n")
    OUT.write("\n")

    pvalues.sort()
    OUT.write("Checking constrained estimation:\n")
    OUT.write("Left hand side:\n")
    OUT.write(np.array_str(lhs) + "\n")
    OUT.write("Right hand side:\n")
    OUT.write(np.array_str(rhs) + "\n")
    OUT.write("Observed p-values   Expected Null p-values\n")
    for q in np.arange(0.1, 0.91, 0.1):
        OUT.write("%20.3f %20.3f\n" % (pvalues[int(q*len(pvalues))], q))

    OUT.write("=" * 80 + "\n\n")

OUT.close()
