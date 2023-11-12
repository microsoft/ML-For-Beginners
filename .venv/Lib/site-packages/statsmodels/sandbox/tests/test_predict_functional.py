from statsmodels.sandbox.predict_functional import predict_functional
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

# If true, the output is written to a multi-page pdf file.
pdf_output = False

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

def pctl(q):
    return lambda x : np.percentile(x, 100 *q)


class TestPredFunc:

    @classmethod
    def setup_class(cls):
        if pdf_output:
            from matplotlib.backends.backend_pdf import PdfPages
            cls.pdf = PdfPages("predict_functional.pdf")

    @classmethod
    def teardown_class(cls):
        if pdf_output:
            cls.pdf.close()

    def close_or_save(self, fig):
        if pdf_output:
            self.pdf.savefig(fig)

    @pytest.mark.matplotlib
    def test_formula(self, close_figures):

        np.random.seed(542)
        n = 500
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        x3 = np.random.normal(size=n)
        x4 = np.random.randint(0, 5, size=n)
        x4 = np.asarray(["ABCDE"[i] for i in x4])
        x5 = np.random.normal(size=n)
        y = 0.3*x2**2 + (x4 == "B") + 0.1*(x4 == "B")*x2**2 + x5 + np.random.normal(size=n)

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})

        fml = "y ~ x1 + bs(x2, df=4) + x3 + x2*x3 + I(x1**2) + C(x4) + C(x4)*bs(x2, df=4) + x5"
        model = sm.OLS.from_formula(fml, data=df)
        result = model.fit()

        summaries = {"x1": np.mean, "x3": pctl(0.75), "x5": np.mean}

        values = {"x4": "B"}
        pr1, ci1, fvals1 = predict_functional(result, "x2", summaries, values)

        values = {"x4": "C"}
        pr2, ci2, fvals2 = predict_functional(result, "x2", summaries, values)

        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.plot(fvals1, pr1, '-', label='x4=B')
        plt.plot(fvals2, pr2, '-', label='x4=C')
        ha, lb = ax.get_legend_handles_labels()
        plt.figlegend(ha, lb, loc="center right")
        plt.xlabel("Focus variable", size=15)
        plt.ylabel("Fitted mean", size=15)
        plt.title("Linear model prediction")
        self.close_or_save(fig)

        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.plot(fvals1, pr1, '-', label='x4=B')
        plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
        plt.plot(fvals2, pr2, '-', label='x4=C')
        plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
        ha, lb = ax.get_legend_handles_labels()
        plt.figlegend(ha, lb, loc="center right")
        plt.xlabel("Focus variable", size=15)
        plt.ylabel("Fitted mean", size=15)
        plt.title("Linear model prediction")
        self.close_or_save(fig)

    @pytest.mark.matplotlib
    def test_lm_contrast(self, close_figures):

        np.random.seed(542)
        n = 200
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        x3 = np.random.normal(size=n)
        y = x1 + 2*x2 + x3 - x1*x2 + x2*x3 + np.random.normal(size=n)

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        fml = "y ~ x1 + x2 + x3 + x1*x2 + x2*x3"
        model = sm.OLS.from_formula(fml, data=df)
        result = model.fit()

        values = {"x2": 1, "x3": 1} # y = 4
        values2 = {"x2": 0, "x3": 0} # y = x1
        pr, cb, fvals = predict_functional(result, "x1", values=values,
                                           values2=values2, ci_method='scheffe')

        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.67, 0.8])
        plt.plot(fvals, pr, '-', label="Estimate", color='orange', lw=4)
        plt.plot(fvals, 4 - fvals, '-', label="Truth", color='lime', lw=4)
        plt.fill_between(fvals, cb[:, 0], cb[:, 1], color='grey')
        ha, lb = ax.get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, loc="center right")
        leg.draw_frame(False)
        plt.xlabel("Focus variable", size=15)
        plt.ylabel("Mean contrast", size=15)
        plt.title("Linear model contrast")
        self.close_or_save(fig)

    @pytest.mark.matplotlib
    def test_glm_formula_contrast(self, close_figures):

        np.random.seed(542)
        n = 50
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        x3 = np.random.normal(size=n)
        mn = 5 + 0.1*x1 + 0.1*x2 + 0.1*x3 - 0.1*x1*x2
        y = np.random.poisson(np.exp(mn), size=len(mn))

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        fml = "y ~ x1 + x2 + x3 + x1*x2"
        model = sm.GLM.from_formula(fml, data=df, family=sm.families.Poisson())
        result = model.fit()

        values = {"x2": 1, "x3": 1} # y = 5.2
        values2 = {"x2": 0, "x3": 0} # y = 5 + 0.1*x1
        pr, cb, fvals = predict_functional(result, "x1", values=values,
                                           values2=values2, ci_method='simultaneous')

        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.67, 0.8])
        plt.plot(fvals, pr, '-', label="Estimate", color='orange', lw=4)
        plt.plot(fvals, 0.2 - 0.1*fvals, '-', label="Truth", color='lime', lw=4)
        plt.fill_between(fvals, cb[:, 0], cb[:, 1], color='grey')
        ha, lb = ax.get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, loc="center right")
        leg.draw_frame(False)
        plt.xlabel("Focus variable", size=15)
        plt.ylabel("Linear predictor contrast", size=15)
        plt.title("Poisson regression contrast")
        self.close_or_save(fig)

    @pytest.mark.matplotlib
    def test_scb(self, close_figures):

        np.random.seed(473)
        n = 100
        x = np.random.normal(size=(n,4))
        x[:, 0] = 1

        for fam_name in "poisson", "binomial", "gaussian":

            if fam_name == "poisson":
                y = np.random.poisson(20, size=n)
                fam = sm.families.Poisson()
                true_mean = 20
                true_lp = np.log(20)
            elif fam_name == "binomial":
                y = 1 * (np.random.uniform(size=n) < 0.5)
                fam = sm.families.Binomial()
                true_mean = 0.5
                true_lp = 0
            elif fam_name == "gaussian":
                y = np.random.normal(size=n)
                fam = sm.families.Gaussian()
                true_mean = 0
                true_lp = 0

            model = sm.GLM(y, x, family=fam)
            result = model.fit()

            # CB is for linear predictor or mean response
            for linear in False, True:

                true = true_lp if linear else true_mean

                values = {'const': 1, "x2": 0}
                summaries = {"x3": np.mean}
                pred1, cb1, fvals1 = predict_functional(result, "x1",
                            values=values, summaries=summaries, linear=linear)
                pred2, cb2, fvals2 = predict_functional(result, "x1",
                            values=values, summaries=summaries,
                            ci_method='simultaneous', linear=linear)

                plt.clf()
                fig = plt.figure()
                ax = plt.axes([0.1, 0.1, 0.58, 0.8])
                plt.plot(fvals1, pred1, '-', color='black', label='Estimate')
                plt.plot(fvals1, true * np.ones(len(pred1)), '-', color='purple',
                         label='Truth')
                plt.plot(fvals1, cb1[:, 0], color='blue', label='Pointwise CB')
                plt.plot(fvals1, cb1[:, 1], color='blue')
                plt.plot(fvals2, cb2[:, 0], color='green', label='Simultaneous CB')
                plt.plot(fvals2, cb2[:, 1], color='green')
                ha, lb = ax.get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, loc="center right")
                leg.draw_frame(False)
                plt.xlabel("Focus variable", size=15)
                if linear:
                    plt.ylabel("Linear predictor", size=15)
                else:
                    plt.ylabel("Fitted mean", size=15)
                plt.title("%s family prediction" % fam_name.capitalize())

                self.close_or_save(fig)

    @pytest.mark.matplotlib
    def test_glm_formula(self, close_figures):

        np.random.seed(542)
        n = 500
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        x3 = np.random.randint(0, 3, size=n)
        x3 = np.asarray(["ABC"[i] for i in x3])
        lin_pred = -1 + 0.5*x1**2 + (x3 == "B")
        prob = 1 / (1 + np.exp(-lin_pred))
        y = 1 * (np.random.uniform(size=n) < prob)

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        fml = "y ~ x1 + I(x1**2) + x2 + C(x3)"
        model = sm.GLM.from_formula(fml, family=sm.families.Binomial(), data=df)
        result = model.fit()
        summaries = {"x2": np.mean}

        for linear in False, True:

            values = {"x3": "B"}
            pr1, ci1, fvals1 = predict_functional(result, "x1", summaries, values, linear=linear)

            values = {"x3": "C"}
            pr2, ci2, fvals2 = predict_functional(result, "x1", summaries, values, linear=linear)

            exact1 = -1 + 0.5*fvals1**2 + 1
            exact2 = -1 + 0.5*fvals2**2

            if not linear:
                exact1 = 1 / (1 + np.exp(-exact1))
                exact2 = 1 / (1 + np.exp(-exact2))

            plt.clf()
            fig = plt.figure()
            ax = plt.axes([0.1, 0.1, 0.7, 0.8])
            plt.plot(fvals1, pr1, '-', label='x3=B')
            plt.plot(fvals2, pr2, '-', label='x3=C')
            plt.plot(fvals1, exact1, '-', label='x3=B (exact)')
            plt.plot(fvals2, exact2, '-', label='x3=C (exact)')
            ha, lb = ax.get_legend_handles_labels()
            plt.figlegend(ha, lb, loc="center right")
            plt.xlabel("Focus variable", size=15)
            if linear:
                plt.ylabel("Fitted linear predictor", size=15)
            else:
                plt.ylabel("Fitted probability", size=15)
            plt.title("Binomial GLM prediction")
            self.close_or_save(fig)

            plt.clf()
            fig = plt.figure()
            ax = plt.axes([0.1, 0.1, 0.7, 0.8])
            plt.plot(fvals1, pr1, '-', label='x3=B', color='orange')
            plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
            plt.plot(fvals2, pr2, '-', label='x3=C', color='lime')
            plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
            ha, lb = ax.get_legend_handles_labels()
            plt.figlegend(ha, lb, loc="center right")
            plt.xlabel("Focus variable", size=15)
            if linear:
                plt.ylabel("Fitted linear predictor", size=15)
            else:
                plt.ylabel("Fitted probability", size=15)
            plt.title("Binomial GLM prediction")
            self.close_or_save(fig)

    @pytest.mark.matplotlib
    def test_noformula_prediction(self, close_figures):

        np.random.seed(6434)
        n = 200
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        x3 = np.random.normal(size=n)
        y = x1 - x2 + np.random.normal(size=n)

        exog = np.vstack((x1, x2, x3)).T

        model = sm.OLS(y, exog)
        result = model.fit()

        summaries = {"x3": pctl(0.75)}
        values = {"x2": 1}
        pr1, ci1, fvals1 = predict_functional(result, "x1", summaries, values)

        values = {"x2": -1}
        pr2, ci2, fvals2 = predict_functional(result, "x1", summaries, values)

        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.plot(fvals1, pr1, '-', label='x2=1', lw=4, alpha=0.6, color='orange')
        plt.plot(fvals2, pr2, '-', label='x2=-1', lw=4, alpha=0.6, color='lime')
        ha, lb = ax.get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, loc="center right")
        leg.draw_frame(False)
        plt.xlabel("Focus variable", size=15)
        plt.ylabel("Fitted mean", size=15)
        plt.title("Linear model prediction")
        self.close_or_save(fig)

        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.plot(fvals1, pr1, '-', label='x2=1', lw=4, alpha=0.6, color='orange')
        plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
        plt.plot(fvals1, pr2, '-', label='x2=1', lw=4, alpha=0.6, color='lime')
        plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
        ha, lb = ax.get_legend_handles_labels()
        plt.figlegend(ha, lb, loc="center right")
        plt.xlabel("Focus variable", size=15)
        plt.ylabel("Fitted mean", size=15)
        plt.title("Linear model prediction")
        self.close_or_save(fig)
