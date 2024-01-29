from statsmodels.compat.python import lmap, lrange, lzip

import copy
from itertools import zip_longest
import time

import numpy as np

from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
    fmt_2,
    fmt_2cols,
    fmt_params,
    gen_fmt,
)

from .summary2 import _model_types


def forg(x, prec=3):
    x = np.squeeze(x)
    if prec == 3:
        # for 3 decimals
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    else:
        raise ValueError("`prec` argument must be either 3 or 4, not {prec}"
                         .format(prec=prec))


def d_or_f(x, width=6):
    """convert number to string with either integer of float formatting

    This is used internally for nobs and degrees of freedom which are usually
    integers but can be float in some cases.

    Parameters
    ----------
    x : int or float
    width : int
        only used if x is nan

    Returns
    -------
    str : str
        number as formatted string
    """
    if np.isnan(x):
        return (width - 3) * ' ' + 'NaN'

    if x // 1 == x:
        return "%#6d" % x
    else:
        return "%#8.2f" % x


def summary(self, yname=None, xname=None, title=0, alpha=.05,
            returns='text', model_info=None):
    """
    Parameters
    ----------
    yname : str
            optional, Default is `Y`
    xname : list[str]
            optional, Default is `X.#` for # in p the number of regressors
    Confidance interval : (0,1) not implimented
    title : str
            optional, Defualt is 'Generalized linear model'
    returns : str
              'text', 'table', 'csv', 'latex', 'html'

    Returns
    -------
    Default :
    returns='print'
            Prints the summarirized results

    Option :
    returns='text'
            Prints the summarirized results

    Option :
    returns='table'
             SimpleTable instance : summarizing the fit of a linear model.

    Option :
    returns='csv'
            returns a string of csv of the results, to import into a spreadsheet

    Option :
    returns='latex'
    Not implimented yet

    Option :
    returns='HTML'
    Not implimented yet


    Examples (needs updating)
    --------
    >>> import statsmodels as sm
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> ols_results = sm.OLS(data.endog, data.exog).results
    >>> print ols_results.summary()
    ...

    Notes
    -----
    conf_int calculated from normal dist.
    """
    if title == 0:
        title = _model_types[self.model.__class__.__name__]

    if xname is not None and len(xname) != len(self.params):
        # GH 2298
        raise ValueError('User supplied xnames must have the same number of '
                         'entries as the number of model parameters '
                         '({0})'.format(len(self.params)))

    yname, xname = _getnames(self, yname, xname)

    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)
    modeltype = self.model.__class__.__name__
    nobs = self.nobs
    df_model = self.df_model
    df_resid = self.df_resid

    #General part of the summary table, Applicable to all? models
    #------------------------------------------------------------
    # TODO: define this generically, overwrite in model classes
    #replace definition of stubs data by single list
    #e.g.
    gen_left = [('Model type:', [modeltype]),
                ('Date:', [date]),
                ('Dependent Variable:', yname),  # TODO: What happens with multiple names?
                ('df model', [df_model])
                ]
    gen_stubs_left, gen_data_left = zip_longest(*gen_left) #transpose row col

    gen_title = title
    gen_header = None
    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title=gen_title,
                                 txt_fmt=gen_fmt
                                 )

    gen_stubs_right = ('Method:',
                       'Time:',
                       'Number of Obs:',
                       'df resid')
    gen_data_right = ([modeltype], #was dist family need to look at more
                      time_of_day,
                      [nobs],
                      [df_resid]
                      )
    gen_table_right = SimpleTable(gen_data_right,
                                  gen_header,
                                  gen_stubs_right,
                                  title=gen_title,
                                  txt_fmt=gen_fmt
                                  )
    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    # Parameters part of the summary table
    # ------------------------------------
    # Note: this is not necessary since we standardized names,
    #  only t versus normal
    tstats = {'OLS': self.t(),
              'GLS': self.t(),
              'GLSAR': self.t(),
              'WLS': self.t(),
              'RLM': self.t(),
              'GLM': self.t()}
    prob_stats = {'OLS': self.pvalues,
                  'GLS': self.pvalues,
                  'GLSAR': self.pvalues,
                  'WLS': self.pvalues,
                  'RLM': self.pvalues,
                  'GLM': self.pvalues
                  }
    # Dictionary to store the header names for the parameter part of the
    # summary table. look up by modeltype
    alp = str((1-alpha)*100)+'%'
    param_header = {
         'OLS'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'GLS'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'GLSAR' : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'WLS'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'GLM'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'], #glm uses t-distribution
         'RLM'   : ['coef', 'std err', 'z', 'P>|z|', alp + ' Conf. Interval']  #checke z
                   }
    params_stubs = xname
    params = self.params
    conf_int = self.conf_int(alpha)
    std_err = self.bse
    exog_len = lrange(len(xname))
    tstat = tstats[modeltype]
    prob_stat = prob_stats[modeltype]

    # Simpletable should be able to handle the formating
    params_data = lzip(["%#6.4g" % (params[i]) for i in exog_len],
                       ["%#6.4f" % (std_err[i]) for i in exog_len],
                       ["%#6.4f" % (tstat[i]) for i in exog_len],
                       ["%#6.4f" % (prob_stat[i]) for i in exog_len],
                       ["(%#5g, %#5g)" % tuple(conf_int[i]) for i in exog_len])
    parameter_table = SimpleTable(params_data,
                                  param_header[modeltype],
                                  params_stubs,
                                  title=None,
                                  txt_fmt=fmt_2
                                  )

    #special table
    #-------------
    #TODO: exists in linear_model, what about other models
    #residual diagnostics

    #output options
    #--------------
    #TODO: JP the rest needs to be fixed, similar to summary in linear_model

    def ols_printer():
        """
        print summary table for ols models
        """
        table = str(general_table)+'\n'+str(parameter_table)
        return table

    def glm_printer():
        table = str(general_table)+'\n'+str(parameter_table)
        return table

    printers = {'OLS': ols_printer, 'GLM': glm_printer}

    if returns == 'print':
        try:
            return printers[modeltype]()
        except KeyError:
            return printers['OLS']()


def _getnames(self, yname=None, xname=None):
    '''extract names from model or construct names
    '''
    if yname is None:
        if getattr(self.model, 'endog_names', None) is not None:
            yname = self.model.endog_names
        else:
            yname = 'y'

    if xname is None:
        if getattr(self.model, 'exog_names', None) is not None:
            xname = self.model.exog_names
        else:
            xname = ['var_%d' % i for i in range(len(self.params))]

    return yname, xname


def summary_top(results, title=None, gleft=None, gright=None, yname=None, xname=None):
    '''generate top table(s)


    TODO: this still uses predefined model_methods
    ? allow gleft, gright to be 1 element tuples instead of filling with None?

    '''
    #change of names ?
    gen_left, gen_right = gleft, gright

    # time and names are always included
    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)

    yname, xname = _getnames(results, yname=yname, xname=xname)

    # create dictionary with default
    # use lambdas because some values raise exception if they are not available
    default_items = dict([
          ('Dependent Variable:', lambda: [yname]),
          ('Dep. Variable:', lambda: [yname]),
          ('Model:', lambda: [results.model.__class__.__name__]),
          ('Date:', lambda: [date]),
          ('Time:', lambda: time_of_day),
          ('Number of Obs:', lambda: [results.nobs]),
          ('No. Observations:', lambda: [d_or_f(results.nobs)]),
          ('Df Model:', lambda: [d_or_f(results.df_model)]),
          ('Df Residuals:', lambda: [d_or_f(results.df_resid)]),
          ('Log-Likelihood:', lambda: ["%#8.5g" % results.llf])  # does not exist for RLM - exception
    ])

    if title is None:
        title = results.model.__class__.__name__ + 'Regression Results'

    if gen_left is None:
        # default: General part of the summary table, Applicable to all? models
        gen_left = [('Dep. Variable:', None),
                    ('Model type:', None),
                    ('Date:', None),
                    ('No. Observations:', None),
                    ('Df model:', None),
                    ('Df resid:', None)]

        try:
            llf = results.llf  # noqa: F841
            gen_left.append(('Log-Likelihood', None))
        except: # AttributeError, NotImplementedError
            pass

        gen_right = []

    gen_title = title
    gen_header = None

    # replace missing (None) values with default values
    gen_left_ = []
    for item, value in gen_left:
        if value is None:
            value = default_items[item]()  # let KeyErrors raise exception
        gen_left_.append((item, value))
    gen_left = gen_left_

    if gen_right:
        gen_right_ = []
        for item, value in gen_right:
            if value is None:
                value = default_items[item]()  # let KeyErrors raise exception
            gen_right_.append((item, value))
        gen_right = gen_right_

    # check nothing was missed
    missing_values = [k for k,v in gen_left + gen_right if v is None]
    assert missing_values == [], missing_values

    # pad both tables to equal number of rows
    if gen_right:
        if len(gen_right) < len(gen_left):
            # fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            # fill up with blank lines to same length, just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

        # padding in SimpleTable does not work like I want
        #force extra spacing and exact string length in right table
        gen_right = [('%-21s' % ('  '+k), v) for k,v in gen_right]
        gen_stubs_right, gen_data_right = zip_longest(*gen_right) #transpose row col
        gen_table_right = SimpleTable(gen_data_right,
                                      gen_header,
                                      gen_stubs_right,
                                      title=gen_title,
                                      txt_fmt=fmt_2cols
                                      )
    else:
        gen_table_right = []  #because .extend_right seems works with []

    #moved below so that we can pad if needed to match length of gen_right
    #transpose rows and columns, `unzip`
    gen_stubs_left, gen_data_left = zip_longest(*gen_left) #transpose row col

    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title=gen_title,
                                 txt_fmt=fmt_2cols
                                 )

    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table


def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, title=None):
    '''create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : {str, None}
        optional name for the endogenous variable, default is "y"
    xname : {list[str], None}
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    '''

    # Parameters part of the summary table
    # ------------------------------------
    # Note: this is not necessary since we standardized names,
    #   only t versus normal

    if isinstance(results, tuple):
        # for multivariate endog
        # TODO: check whether I do not want to refactor this
        #we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = np.asarray(results.params)
        std_err = np.asarray(results.bse)
        tvalues = np.asarray(results.tvalues)  # is this sometimes called zvalues
        pvalues = np.asarray(results.pvalues)
        conf_int = np.asarray(results.conf_int(alpha))
    if params.size == 0:
        return SimpleTable([['No Model Parameters']])
    # Dictionary to store the header names for the parameter part of the
    # summary table. look up by modeltype
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']

    if skip_header:
        param_header = None

    _, xname = _getnames(results, yname=yname, xname=xname)

    if len(xname) != len(params):
        raise ValueError('xnames and params do not have the same length')

    params_stubs = xname

    exog_idx = lrange(len(xname))
    params = np.asarray(params)
    std_err = np.asarray(std_err)
    tvalues = np.asarray(tvalues)
    pvalues = np.asarray(pvalues)
    conf_int = np.asarray(conf_int)
    params_data = lzip([forg(params[i], prec=4) for i in exog_idx],
                       [forg(std_err[i]) for i in exog_idx],
                       [forg(tvalues[i]) for i in exog_idx],
                       ["%#6.3f" % (pvalues[i]) for i in exog_idx],
                       [forg(conf_int[i,0]) for i in exog_idx],
                       [forg(conf_int[i,1]) for i in exog_idx])
    parameter_table = SimpleTable(params_data,
                                  param_header,
                                  params_stubs,
                                  title=title,
                                  txt_fmt=fmt_params
                                  )

    return parameter_table


def summary_params_frame(results, yname=None, xname=None, alpha=.05,
                         use_t=True):
    """
    Create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : {str, None}
        optional name for the endogenous variable, default is "y"
    xname : {list[str], None}
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    """

    # Parameters part of the summary table
    # ------------------------------------
    # Note: this is not necessary since we standardized names,
    #   only t versus normal

    if isinstance(results, tuple):
        # for multivariate endog
        # TODO: check whether I do not want to refactor this
        #we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        std_err = results.bse
        tvalues = results.tvalues  #is this sometimes called zvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)

    # Dictionary to store the header names for the parameter part of the
    # summary table. look up by modeltype
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        'Conf. Int. Low', 'Conf. Int. Upp.']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        'Conf. Int. Low', 'Conf. Int. Upp.']

    _, xname = _getnames(results, yname=yname, xname=xname)

    from pandas import DataFrame
    table = np.column_stack((params, std_err, tvalues, pvalues, conf_int))
    return DataFrame(table, columns=param_header, index=xname)


def summary_params_2d(result, extras=None, endog_names=None, exog_names=None,
                      title=None):
    """create summary table of regression parameters with several equations

    This allows interleaving of parameters with bse and/or tvalues

    Parameters
    ----------
    result : result instance
        the result instance with params and attributes in extras
    extras : list[str]
        additional attributes to add below a parameter row, e.g. bse or tvalues
    endog_names : {list[str], None}
        names for rows of the parameter array (multivariate endog)
    exog_names : {list[str], None}
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    title : None or string

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    """
    if endog_names is None:
        # TODO: note the [1:] is specific to current MNLogit
        endog_names = ['endog_%d' % i for i in
                       np.unique(result.model.endog)[1:]]
    if exog_names is None:
        exog_names = ['var%d' % i for i in range(len(result.params))]

    # TODO: check formatting options with different values
    res_params = [[forg(item, prec=4) for item in row] for row in result.params]
    if extras:
        extras_list = [[['%10s' % ('(' + forg(v, prec=3).strip() + ')')
                         for v in col]
                        for col in getattr(result, what)]
                       for what in extras
                       ]
        data = lzip(res_params, *extras_list)
        data = [i for j in data for i in j]  #flatten
        stubs = lzip(endog_names, *[['']*len(endog_names)]*len(extras))
        stubs = [i for j in stubs for i in j] #flatten
    else:
        data = res_params
        stubs = endog_names

    txt_fmt = copy.deepcopy(fmt_params)
    txt_fmt["data_fmts"] = ["%s"]*result.params.shape[1]

    return SimpleTable(data, headers=exog_names,
                             stubs=stubs,
                             title=title,
                             txt_fmt=txt_fmt)


def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.05,
                          use_t=True, keep_headers=True, endog_cols=False):
    """summary table for parameters that are 2d, e.g. multi-equation models

    Parameters
    ----------
    result : result instance
        the result instance with params, bse, tvalues and conf_int
    endog_names : {list[str], None}
        names for rows of the parameter array (multivariate endog)
    exog_names : {list[str], None}
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    keep_headers : bool
        If true (default), then sub-tables keep their headers. If false, then
        only the first headers are kept, the other headerse are blanked out
    endog_cols : bool
        If false (default) then params and other result statistics have
        equations by rows. If true, then equations are assumed to be in columns.
        Not implemented yet.

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    """

    res = result
    params = res.params
    if params.ndim == 2:  # we've got multiple equations
        n_equ = params.shape[1]
        if len(endog_names) != params.shape[1]:
            raise ValueError('endog_names has wrong length')
    else:
        if len(endog_names) != len(params):
            raise ValueError('endog_names has wrong length')
        n_equ = 1

    #VAR does not have conf_int
    #params = res.params.T # this is a convention for multi-eq models

    # check that we have the right length of names
    if not isinstance(endog_names, list):
        # TODO: this might be specific to multinomial logit type, move?
        if endog_names is None:
            endog_basename = 'endog'
        else:
            endog_basename = endog_names
        # TODO: note, the [1:] is specific to current MNLogit
        endog_names = res.model.endog_names[1:]

    tables = []
    for eq in range(n_equ):
        restup = (res, res.params[:,eq], res.bse[:,eq], res.tvalues[:,eq],
                  res.pvalues[:,eq], res.conf_int(alpha)[eq])

        skiph = False
        tble = summary_params(restup, yname=endog_names[eq],
                              xname=exog_names, alpha=alpha, use_t=use_t,
                              skip_header=skiph)

        tables.append(tble)

    # add titles, they will be moved to header lines in table_extend
    for i in range(len(endog_names)):
        tables[i].title = endog_names[i]

    table_all = table_extend(tables, keep_headers=keep_headers)

    return tables, table_all


def table_extend(tables, keep_headers=True):
    """extend a list of SimpleTables, adding titles to header of subtables

    This function returns the merged table as a deepcopy, in contrast to the
    SimpleTable extend method.

    Parameters
    ----------
    tables : list of SimpleTable instances
    keep_headers : bool
        If true, then all headers are kept. If falls, then the headers of
        subtables are blanked out.

    Returns
    -------
    table_all : SimpleTable
        merged tables as a single SimpleTable instance

    """
    from copy import deepcopy
    for ii, t in enumerate(tables[:]): #[1:]:
        t = deepcopy(t)

        #move title to first cell of header
        # TODO: check if we have multiline headers
        if t[0].datatype == 'header':
            t[0][0].data = t.title
            t[0][0]._datatype = None
            t[0][0].row = t[0][1].row
            if not keep_headers and (ii > 0):
                for c in t[0][1:]:
                    c.data = ''

        # add separating line and extend tables
        if ii == 0:
            table_all = t
        else:
            r1 = table_all[-1]
            r1.add_format('txt', row_dec_below='-')
            table_all.extend(t)

    table_all.title = None
    return table_all


def summary_return(tables, return_fmt='text'):
    # join table parts then print
    if return_fmt == 'text':
        strdrop = lambda x: str(x).rsplit('\n',1)[0]
        # convert to string drop last line
        return '\n'.join(lmap(strdrop, tables[:-1]) + [str(tables[-1])])
    elif return_fmt == 'tables':
        return tables
    elif return_fmt == 'csv':
        return '\n'.join(x.as_csv() for x in tables)
    elif return_fmt == 'latex':
        # TODO: insert \hline after updating SimpleTable
        table = copy.deepcopy(tables[0])
        for part in tables[1:]:
            table.extend(part)
        return table.as_latex_tabular()
    elif return_fmt == 'html':
        return "\n".join(table.as_html() for table in tables)
    else:
        raise ValueError('available output formats are text, csv, latex, html')


class Summary:
    """
    Result summary

    Construction does not take any parameters. Tables and text can be added
    with the `add_` methods.

    Attributes
    ----------
    tables : list of tables
        Contains the list of SimpleTable instances, horizontally concatenated
        tables are not saved separately.
    extra_txt : str
        extra lines that are added to the text output, used for warnings
        and explanations.
    """
    def __init__(self):
        self.tables = []
        self.extra_txt = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        """Display as HTML in IPython notebook."""
        return self.as_html()

    def _repr_latex_(self):
        """Display as LaTeX when converting IPython notebook to PDF."""
        return self.as_latex()

    def add_table_2cols(self, res,  title=None, gleft=None, gright=None,
                        yname=None, xname=None):
        """
        Add a double table, 2 tables with one column merged horizontally

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        title : str, optional
            if None, then a default title is used.
        gleft : list[tuple], optional
            elements for the left table, tuples are (name, value) pairs
            If gleft is None, then a default table is created
        gright : list[tuple], optional
            elements for the right table, tuples are (name, value) pairs
        yname : str, optional
            optional name for the endogenous variable, default is "y"
        xname : list[str], optional
            optional names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model.
        """

        table = summary_top(res, title=title, gleft=gleft, gright=gright,
                            yname=yname, xname=xname)
        self.tables.append(table)

    def add_table_params(self, res, yname=None, xname=None, alpha=.05,
                         use_t=True):
        """create and add a table for the parameter estimates

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        yname : {str, None}
            optional name for the endogenous variable, default is "y"
        xname : {list[str], None}
            optional names for the exogenous variables, default is "var_xx"
        alpha : float
            significance level for the confidence intervals
        use_t : bool
            indicator whether the p-values are based on the Student-t
            distribution (if True) or on the normal distribution (if False)

        Returns
        -------
        None : table is attached

        """
        if res.params.ndim == 1:
            table = summary_params(res, yname=yname, xname=xname, alpha=alpha,
                                   use_t=use_t)
        elif res.params.ndim == 2:
            _, table = summary_params_2dflat(res, endog_names=yname,
                                             exog_names=xname,
                                             alpha=alpha, use_t=use_t)
        else:
            raise ValueError('params has to be 1d or 2d')
        self.tables.append(table)

    def add_extra_txt(self, etext):
        """add additional text that will be added at the end in text format

        Parameters
        ----------
        etext : list[str]
            string with lines that are added to the text output.

        """
        self.extra_txt = '\n'.join(etext)

    def as_text(self):
        """return tables as string

        Returns
        -------
        txt : str
            summary tables and extra text as one string

        """
        txt = summary_return(self.tables, return_fmt='text')
        if self.extra_txt is not None:
            txt = txt + '\n\n' + self.extra_txt
        return txt

    def as_latex(self):
        """return tables as string

        Returns
        -------
        latex : str
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.

        """
        latex = summary_return(self.tables, return_fmt='latex')
        if self.extra_txt is not None:
            latex = latex + '\n\n' + self.extra_txt.replace('\n', ' \\newline\n ')
        return latex

    def as_csv(self):
        """return tables as string

        Returns
        -------
        csv : str
            concatenated summary tables in comma delimited format

        """
        csv = summary_return(self.tables, return_fmt='csv')
        if self.extra_txt is not None:
            csv = csv + '\n\n' + self.extra_txt
        return csv

    def as_html(self):
        """return tables as string

        Returns
        -------
        html : str
            concatenated summary tables in HTML format

        """
        html = summary_return(self.tables, return_fmt='html')
        if self.extra_txt is not None:
            html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
        return html
