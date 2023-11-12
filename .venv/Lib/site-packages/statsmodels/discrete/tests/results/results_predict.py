"""This file has been manually edited based on the generated results

edits
- rearrange Bunch
- use DataFrame

note seond `_cons` in params_table rownames is lnalpha

"""
# flake8: noqa

import numpy as np
import pandas as pd

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
           rank = 9,
           N = 3629,
           ic = 4,
           k = 9,
           k_eq = 2,
           k_dv = 1,
           converged = 1,
           rc = 0,
           k_autoCns = 0,
           ll = -10404.95565541838,
           k_eq_model = 1,
           ll_0 = -10786.68925314471,
           rank0 = 2,
           df_m = 7,
           chi2 = 763.467195452653,
           p = 1.4153888670e-160,
           ll_c = -14287.94887436967,
           chi2_c = 7765.986437902575,
           r2_p = .0353893200005773,
           k_aux = 1,
           alpha = .6166738507905131,
           cmdline = "nbreg docvis private medicaid aget aget2 educyr actlim totchr",
           cmd = "nbreg",
           predict = "nbreg_p",
           dispers = "mean",
           diparm_opt2 = "noprob",
           chi2_ct = "LR",
           chi2type = "LR",
           opt = "moptimize",
           vce = "oim",
           title = "Negative binomial regression",
           diparm1 = "lnalpha, exp label(",
           user = "nbreg_lf",
           crittype = "log likelihood",
           ml_method = "e2",
           singularHmethod = "m-marquardt",
           technique = "nr",
           which = "max",
           depvar = "docvis",
           properties = "b V",
          )

params_table = np.array([
     .18528179233626,  .03348067897193,  5.5339914848088,  3.130241768e-08,
     .11966086737334,  .25090271729919, np.nan,  1.9599639845401,
                   0,  .08475784499449,  .04718372808048,  1.7963363312438,
     .07244104305261, -.00772056269958,  .17723625268856, np.nan,
     1.9599639845401,                0,  .22409326577213,  .04170620298531,
     5.3731399583668,  7.737722210e-08,  .14235060998901,  .30583592155526,
    np.nan,  1.9599639845401,                0, -.04817183015548,
     .00992361535076, -4.8542621265318,  1.208358166e-06, -.06762175883941,
    -.02872190147156, np.nan,  1.9599639845401,                0,
     .02692548760568,  .00419162167105,  6.4236445267994,  1.330497007e-10,
     .01871006009359,  .03514091511776, np.nan,  1.9599639845401,
                   0,  .17048038202011,  .03448967943245,  4.9429390132204,
     7.695356233e-07,  .10288185249418,  .23807891154605, np.nan,
     1.9599639845401,                0,  .27516170294682,  .01205852749453,
     22.818847746673,  2.98049648e-115,  .25152742335095,  .29879598254269,
    np.nan,  1.9599639845401,                0,  .67840343342789,
      .0664120899438,   10.21505924602,  1.697754022e-24,  .54823812900001,
     .80856873785576, np.nan,  1.9599639845401,                0,
    -.48341499971517,  .03134835693943, -15.420744399751,  1.187278967e-53,
    -.54485665029097, -.42197334913938, np.nan,  1.9599639845401,
                   0]).reshape(9,9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()
params_table_rownames = 'private medicaid aget aget2 educyr actlim totchr _cons _cons'.split()

# results for
# margins , predict(n) predict(pr(0)) predict(pr(1)) predict(pr(0, 1)) predict(pr(2, .)) atmeans
table = np.array([
     6.1604164491362,  .09102737953925,  67.676521946673,                0,
     5.9820060636322,  6.3388268346402, np.nan,  1.9599639845401,
                   0,  .07860475517176,  .00344783069748,  22.798322211469,
     4.76427218e-115,  .07184713117991,  .08536237916362, np.nan,
     1.9599639845401,                0,  .10090462231979,  .00218578691875,
      46.16397941361,                0,  .09662055868115,  .10518868595842,
    np.nan,  1.9599639845401,                0,  .17950937749155,
     .00553924697666,  32.406819599838,  2.20005624e-230,  .16865265291582,
     .19036610206727, np.nan,  1.9599639845401,                0,
     .82049062250845,  .00553924699078,  148.12313368113,                0,
     .80963389790505,  .83134734711186, np.nan,  1.9599639845401,
                   0]).reshape(5,9)

table_colnames = 'b se z pvalue ll ul df crit eform'.split()
table_rownames = '1bn._predict 2._predict 3._predict 4._predict 5._predict'.split()
dframe_atmeans = pd.DataFrame(table, index=table_rownames, columns=table_colnames)

# result for
# margins, predict(n) predict(pr(0)) predict(pr(1)) predict(pr(0, 1)) predict(pr(2, .))
table = np.array([
     6.8071952338104,  .10838829819462,  62.803783685096,                0,
     6.5947580730033,  7.0196323946174, np.nan,  1.9599639845401,
                   0,  .08826646029161,  .00350687276409,  25.169564517851,
     8.63155623e-140,  .08139311597563,  .09513980460758, np.nan,
     1.9599639845401,                0,  .10719978561286,  .00205026104517,
     52.285920305334,                0,  .10318134780543,   .1112182234203,
    np.nan,  1.9599639845401,                0,  .19546624590447,
      .0054522133947,  35.850806223874,  1.78661674e-281,  .18478010401484,
      .2061523877941, np.nan,  1.9599639845401,                0,
     .80453375409553,  .00545221340471,  147.56094348787,                0,
     .79384761218628,  .81521989600478, np.nan,  1.9599639845401,
                   0]).reshape(5,9)

table_colnames = 'b se z pvalue ll ul df crit eform'.split()
table_rownames = '1bn._predict 2._predict 3._predict 4._predict 5._predict'.split()
dframe_mean = pd.DataFrame(table, index=table_rownames, columns=table_colnames)

results_nb_docvis = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    results_margins_atmeans=dframe_atmeans,
    results_margins_mean=dframe_mean,
    **est,
    )

# ############################# ZINBP


est = dict(
           rank = 11,
           N = 3629,
           ic = 8,
           k = 11,
           k_eq = 3,
           k_dv = 1,
           converged = 1,
           rc = 0,
           k_autoCns = 0,
           ll = -10404.95308201019,
           k_eq_model = 1,
           ll_0 = -10775.51516555833,
           chi2 = 741.1241670962918,
           p = 9.2654212845e-153,
           N_zero = 392,
           df_m = 7,
           df_c = 2,
           k_aux = 1,
           cmdline = "zinb docvis private medicaid aget aget2 educyr actlim totchr, inflate(aget)",
           cmd = "zinb",
           predict = "zip_p",
           inflate = "logit",
           chi2type = "LR",
           opt = "moptimize",
           vce = "oim",
           title = "Zero-inflated negative binomial regression",
           diparm1 = "lnalpha, exp label(",
           user = "zinb_llf",
           crittype = "log likelihood",
           ml_method = "e2",
           singularHmethod = "m-marquardt",
           technique = "nr",
           which = "max",
           depvar = "docvis",
           properties = "b V",
          )

params_table = np.array([
     .18517571292817,  .03350948180038,  5.5260691296648,  3.274851365e-08,
     .11949833545881,  .25085309039752, np.nan,  1.9599639845401,
                   0,  .08473133853831,  .04717665613525,  1.7960437529823,
     .07248755882811, -.00773320839781,  .17719588547443, np.nan,
     1.9599639845401,                0,  .22335574980273,  .04293022169984,
     5.2027625518539,  1.963476761e-07,  .13921406142272,  .30749743818274,
    np.nan,  1.9599639845401,                0, -.04804896097964,
     .01006690700638, -4.7729616404713,  1.815363776e-06, -.06777973614785,
    -.02831818581142, np.nan,  1.9599639845401,                0,
      .0269244937276,  .00419096037609,  6.4244209707123,  1.323724094e-10,
     .01871036232982,  .03513862512538, np.nan,  1.9599639845401,
                   0,  .17042579343453,  .03449014549225,  4.9412894901473,
     7.760757819e-07,  .10282635044816,  .23802523642089, np.nan,
     1.9599639845401,                0,  .27500074932161,  .01226558007071,
     22.420525383741,  2.48238139e-111,  .25096065413353,  .29904084450969,
    np.nan,  1.9599639845401,                0,  .67986743798706,
     .06944204778986,  9.7904289925953,  1.237696321e-22,  .54376352530622,
     .81597135066789, np.nan,  1.9599639845401,                0,
    -1.2833474076485,   3.692336844421, -.34757051204241,  .72816275506989,
     -8.520194641504,  5.9534998262071, np.nan,  1.9599639845401,
                   0, -6.5587800419911,  13.305282477745, -.49294556902205,
     .62205104781253, -32.636654502503,  19.519094418521, np.nan,
     1.9599639845401,                0,  -.4845756474516,  .03531398529193,
    -13.721919048382,  7.505227546e-43, -.55378978677435, -.41536150812884,
    np.nan,  1.9599639845401,                0]).reshape(11,9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()
params_table_rownames = 'private medicaid aget aget2 educyr actlim totchr _cons aget _cons _cons'.split()

# results for
# margins , predict(n) predict(pr(0)) predict(pr(1)) predict(pr(0, 1)) predict(pr(2, .)) atmeans
table = np.array([
     6.1616899436815,  .09285785618544,  66.356151184199,                0,
     5.9796918898764,  6.3436879974865, np.nan,  1.9599639845401,
                   0,  .07857785668717,  .00351221423708,  22.372740209725,
     7.25412664e-111,   .0716940432765,  .08546167009785, np.nan,
     1.9599639845401,                0,  .10079961393875,  .00263347068017,
     38.276338027191,                0,  .09563810625128,  .10596112162622,
    np.nan,  1.9599639845401,                0,  .17937747062593,
     .00586287199331,  30.595494977635,  1.40505722e-205,  .16788645267307,
     .19086848857879, np.nan,  1.9599639845401,                0,
     .82062252937407,  .00586287199668,  139.96937505016,                0,
     .80913151141461,  .83211354733353, np.nan,  1.9599639845401,
                   0]).reshape(5,9)

table_colnames = 'b se z pvalue ll ul df crit eform'.split()
table_rownames = '1bn._predict 2._predict 3._predict 4._predict 5._predict'.split()
dframe_atmeans = pd.DataFrame(table, index=table_rownames, columns=table_colnames)

# result for
# margins, predict(n) predict(pr(0)) predict(pr(1)) predict(pr(0, 1)) predict(pr(2, .))
table = np.array([
     6.8063733751586,  .10879833124057,  62.559538345387,                0,
      6.593132564349,  7.0196141859682, np.nan,  1.9599639845401,
                   0,  .08842743693234,  .00405939469823,    21.7834045482,
     3.33356305e-105,  .08047116952478,   .0963837043399, np.nan,
     1.9599639845401,                0,  .10706809868425,  .00273617889716,
     39.130518401155,                0,  .10170528659055,  .11243091077794,
    np.nan,  1.9599639845401,                0,  .19549553561658,
     .00545764150876,  35.820516115406,  5.29431574e-281,  .18479875481889,
     .20619231641428, np.nan,  1.9599639845401,                0,
     .80450446438342,   .0054576415014,  147.40881462742,                0,
     .79380768360013,   .8152012451667, np.nan,  1.9599639845401,
                   0]).reshape(5,9)

table_colnames = 'b se z pvalue ll ul df crit eform'.split()
table_rownames = '1bn._predict 2._predict 3._predict 4._predict 5._predict'.split()
dframe_mean = pd.DataFrame(table, index=table_rownames, columns=table_colnames)

results_zinb_docvis = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    results_margins_atmeans=dframe_atmeans,
    results_margins_mean=dframe_mean,
    **est,
    )
