from statsmodels.compat.python import lzip

from io import StringIO

import numpy as np

from statsmodels.iolib import SimpleTable

mat = np.array

_default_table_fmt = dict(
    empty_cell = '',
    colsep='  ',
    row_pre = '',
    row_post = '',
    table_dec_above='=',
    table_dec_below='=',
    header_dec_below='-',
    header_fmt = '%s',
    stub_fmt = '%s',
    title_align='c',
    header_align = 'r',
    data_aligns = 'r',
    stubs_align = 'l',
    fmt = 'txt'
)


class VARSummary:
    default_fmt = dict(
        #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
        #data_fmts = ["%#10.4g","%#10.4g","%#10.4g","%#6.4g"],
        data_fmts = ["%#15.6F","%#15.6F","%#15.3F","%#14.3F"],
        empty_cell = '',
        #colwidths = 10,
        colsep='  ',
        row_pre = '',
        row_post = '',
        table_dec_above='=',
        table_dec_below='=',
        header_dec_below='-',
        header_fmt = '%s',
        stub_fmt = '%s',
        title_align='c',
        header_align = 'r',
        data_aligns = 'r',
        stubs_align = 'l',
        fmt = 'txt'
    )

    part1_fmt = dict(
        default_fmt,
        data_fmts = ["%s"],
        colwidths = 15,
        colsep=' ',
        table_dec_below='',
        header_dec_below=None,
    )
    part2_fmt = dict(
        default_fmt,
        data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
        colwidths = None,
        colsep='    ',
        table_dec_above='-',
        table_dec_below='-',
        header_dec_below=None,
    )

    def __init__(self, estimator):
        self.model = estimator
        self.summary = self.make()

    def __repr__(self):
        return self.summary

    def make(self, endog_names=None, exog_names=None):
        """
        Summary of VAR model
        """
        buf = StringIO()

        buf.write(self._header_table() + '\n')
        buf.write(self._stats_table() + '\n')
        buf.write(self._coef_table() + '\n')
        buf.write(self._resid_info() + '\n')

        return buf.getvalue()

    def _header_table(self):
        import time

        model = self.model

        t = time.localtime()

        # TODO: change when we allow coef restrictions
        # ncoefs = len(model.beta)

        # Header information
        part1title = "Summary of Regression Results"
        part1data = [[model._model_type],
                     ["OLS"], #TODO: change when fit methods change
                     [time.strftime("%a, %d, %b, %Y", t)],
                     [time.strftime("%H:%M:%S", t)]]
        part1header = None
        part1stubs = ('Model:',
                      'Method:',
                      'Date:',
                      'Time:')
        part1 = SimpleTable(part1data, part1header, part1stubs,
                            title=part1title, txt_fmt=self.part1_fmt)

        return str(part1)

    def _stats_table(self):
        # TODO: do we want individual statistics or should users just
        # use results if wanted?
        # Handle overall fit statistics

        model = self.model

        part2Lstubs = ('No. of Equations:',
                       'Nobs:',
                       'Log likelihood:',
                       'AIC:')
        part2Rstubs = ('BIC:',
                       'HQIC:',
                       'FPE:',
                       'Det(Omega_mle):')
        part2Ldata = [[model.neqs], [model.nobs], [model.llf], [model.aic]]
        part2Rdata = [[model.bic], [model.hqic], [model.fpe], [model.detomega]]
        part2Lheader = None
        part2L = SimpleTable(part2Ldata, part2Lheader, part2Lstubs,
                             txt_fmt = self.part2_fmt)
        part2R = SimpleTable(part2Rdata, part2Lheader, part2Rstubs,
                             txt_fmt = self.part2_fmt)
        part2L.extend_right(part2R)

        return str(part2L)

    def _coef_table(self):
        model = self.model
        k = model.neqs

        Xnames = self.model.exog_names

        data = lzip(model.params.T.ravel(),
                    model.stderr.T.ravel(),
                    model.tvalues.T.ravel(),
                    model.pvalues.T.ravel())

        header = ('coefficient','std. error','t-stat','prob')

        buf = StringIO()
        dim = k * model.k_ar + model.k_trend + model.k_exog_user
        for i in range(k):
            section = "Results for equation %s" % model.names[i]
            buf.write(section + '\n')

            table = SimpleTable(data[dim * i : dim * (i + 1)], header,
                                Xnames, title=None, txt_fmt = self.default_fmt)
            buf.write(str(table) + '\n')

            if i < k - 1:
                buf.write('\n')

        return buf.getvalue()

    def _resid_info(self):
        buf = StringIO()
        names = self.model.names

        buf.write("Correlation matrix of residuals" + '\n')
        buf.write(pprint_matrix(self.model.resid_corr, names, names) + '\n')

        return buf.getvalue()


def normality_summary(results):
    title = "Normality skew/kurtosis Chi^2-test"
    null_hyp = 'H_0: data generated by normally-distributed process'
    return hypothesis_test_table(results, title, null_hyp)


def hypothesis_test_table(results, title, null_hyp):
    fmt = dict(_default_table_fmt,
               data_fmts=["%#15.6F","%#15.6F","%#15.3F", "%s"])

    buf = StringIO()
    table = SimpleTable([[results['statistic'],
                          results['crit_value'],
                          results['pvalue'],
                          str(results['df'])]],
                        ['Test statistic', 'Critical Value', 'p-value',
                         'df'], [''], title=None, txt_fmt=fmt)

    buf.write(title + '\n')
    buf.write(str(table) + '\n')

    buf.write(null_hyp + '\n')

    buf.write("Conclusion: %s H_0" % results['conclusion'])
    buf.write(" at %.2f%% significance level" % (results['signif'] * 100))

    return buf.getvalue()


def pprint_matrix(values, rlabels, clabels, col_space=None):
    buf = StringIO()

    T, K = len(rlabels), len(clabels)

    if col_space is None:
        min_space = 10
        col_space = [max(len(str(c)) + 2, min_space) for c in clabels]
    else:
        col_space = (col_space,) * K

    row_space = max([len(str(x)) for x in rlabels]) + 2

    head = _pfixed('', row_space)

    for j, h in enumerate(clabels):
        head += _pfixed(h, col_space[j])

    buf.write(head + '\n')

    for i, rlab in enumerate(rlabels):
        line = ('%s' % rlab).ljust(row_space)

        for j in range(K):
            line += _pfixed(values[i,j], col_space[j])

        buf.write(line + '\n')

    return buf.getvalue()


def _pfixed(s, space, nanRep=None, float_format=None):
    if isinstance(s, float):
        if float_format:
            formatted = float_format(s)
        else:
            formatted = "%#8.6F" % s

        return formatted.rjust(space)
    else:
        return ('%s' % s)[:space].rjust(space)
