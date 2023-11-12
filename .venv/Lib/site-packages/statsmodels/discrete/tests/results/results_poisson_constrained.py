import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
           rank=7,
           N=10,
           ic=3,
           k=8,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-33.45804471711131,
           k_eq_model=1,
           ll_0=-349.6684656479622,
           df_m=6,
           chi2=632.4208418617018,
           p=2.3617193197e-133,
           r2_p=.9043149497192691,
           cmdline="poisson deaths lnpyears smokes i.agecat",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           gof="poiss_g",
           chi2type="LR",
           opt="moptimize",
           vce="oim",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    .66308184237808,  .63593388706566,  1.0426899019923,  .29709193621918,
    -.58332567281917,  1.9094893575753, np.nan,  1.9599639845401,
    0,  .84966723812924,  .94279599903649,  .90122066597395,
    .36747100512904, -.99817896475073,  2.6975134410092, np.nan,
    1.9599639845401,                0,                0, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    np.nan,  1.9599639845401,                0,  1.3944392032504,
    .25613243411925,  5.4442117338454,  5.203529593e-08,   .8924288571041,
    1.8964495493967, np.nan,  1.9599639845401,                0,
    2.389284381366,  .48305517266329,  4.9461935542328,  7.567871319e-07,
    1.4425136404002,  3.3360551223318, np.nan,  1.9599639845401,
    0,  2.8385093615484,  .98099727008295,  2.8934936397003,
    .00380982006764,  .91579004325369,  4.7612286798431, np.nan,
    1.9599639845401,                0,  2.9103531988515,   1.500316321385,
    1.9398263935201,  .05240079188831, -.03021275648066,  5.8509191541838,
    np.nan,  1.9599639845401,                0,  -4.724924181641,
    6.0276019460727, -.78388125558284,  .43310978942119, -16.538806909087,
    7.088958545805, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                         '3.agecat', '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .40441190871844, -.59566294916097,                0,   .1055698685775,
    .28413388045122,  .61269322798077,  .94624135329227, -3.8311942353131,
    -.59566294916097,  .88886429579921,                0, -.15587944298625,
    -.4190789999425, -.90299843943229, -1.3940094688194,  5.6335527795822,
    0,                0,                0,                0,
    0,                0,                0,                0,
    .1055698685775, -.15587944298625,                0,  .06560382380785,
    .10360281461667,  .18937107288073,  .27643306166968,  -1.029211453947,
    .28413388045122,  -.4190789999425,                0,  .10360281461667,
    .23334229983676,  .45990880867889,  .69424104947043, -2.7206801001387,
    .61269322798077, -.90299843943229,                0,  .18937107288073,
    .45990880867889,  .96235564391021,  1.4630024143274, -5.8333014154113,
    .94624135329227, -1.3940094688194,                0,  .27643306166968,
    .69424104947043,  1.4630024143274,  2.2509490642142,  -8.993394678922,
    -3.8311942353131,  5.6335527795822,                0,  -1.029211453947,
    -2.7206801001387, -5.8333014154113,  -8.993394678922,  36.331985220299
    ]).reshape(8, 8)

cov_colnames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                '3.agecat', '4.agecat', '5.agecat', '_cons']

cov_rownames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                '3.agecat', '4.agecat', '5.agecat', '_cons']


results_noexposure_noconstraint = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=6,
           N=10,
           ic=3,
           k=7,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-33.6001534405213,
           k_eq_model=1,
           ll_0=-495.0676356770329,
           df_m=5,
           chi2=922.9349644730232,
           p=2.8920463572e-197,
           r2_p=.9321301757191799,
           cmdline="poisson deaths smokes i.agecat, exposure(pyears)",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(pyears)",
           gof="poiss_g",
           chi2type="LR",
           opt="moptimize",
           vce="oim",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    .35453563725291,  .10737411818853,  3.3018723993653,  .00096041750265,
    .14408623273163,  .56498504177418, np.nan,  1.9599639845401,
    0,                0, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    1.9599639845401,                0,  1.4840070063099,  .19510337263434,
    7.606260139291,  2.821411159e-14,  1.1016114226842,  1.8664025899355,
    np.nan,  1.9599639845401,                0,  2.6275051184579,
    .18372726944827,  14.301116684248,  2.153264398e-46,  2.2674062873614,
    2.9876039495544, np.nan,  1.9599639845401,                0,
    3.350492785161,  .18479918093323,  18.130452571495,  1.832448146e-73,
    2.9882930461593,  3.7126925241626, np.nan,  1.9599639845401,
    0,  3.7000964518246,  .19221951212105,   19.24932807807,
    1.430055953e-82,  3.3233531309415,  4.0768397727077, np.nan,
    1.9599639845401,                0,  -7.919325711822,  .19176181876223,
    -41.297719029467,                0, -8.2951719702059, -7.5434794534381,
    np.nan,  1.9599639845401,                0]).reshape(7, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                         '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .01152920125677,                0, -.00061561668833, -.00090117889461,
    -.00087280941113, -.00045274641397, -.00921219275997,                0,
    0,                0,                0,                0,
    0,                0, -.00061561668833,                0,
    .0380653260133,  .02945988432334,  .02945836949789,   .0294359396881,
    -.0289198676971, -.00090117889461,                0,  .02945988432334,
    .03375570953892,   .0294799877675,  .02944715358419, -.02869169455392,
    -.00087280941113,                0,  .02945836949789,   .0294799877675,
    .03415073727359,  .02944603952766, -.02871436265941, -.00045274641397,
    0,   .0294359396881,  .02944715358419,  .02944603952766,
    .03694834084006, -.02905000614546, -.00921219275997,                0,
    -.0289198676971, -.02869169455392, -.02871436265941, -.02905000614546,
    .036772595135]).reshape(7, 7)

cov_colnames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']

cov_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']


results_exposure_noconstraint = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=6,
           N=10,
           ic=4,
           k=8,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-33.46699798755848,
           k_eq_model=1,
           df_m=5,
           chi2=452.5895246742914,
           p=1.35732711092e-95,
           r2_p=np.nan,
           cmdline="poisson deaths lnpyears smokes i.agecat, constraints(1)",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vce="oim",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    .57966535352347,  .13107152221057,  4.4225117992619,  9.756001957e-06,
    .32276989059191,  .83656081645503, np.nan,  1.9599639845401,
    0,  .97254074124891,  .22289894431919,  4.3631464663029,
    .00001282050472,   .5356668381913,  1.4094146443065, np.nan,
    1.9599639845401,                0,                0, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    np.nan,  1.9599639845401,                0,  1.3727621378494,
    .19798042377276,  6.9338276567436,  4.096036246e-12,  .98472763761078,
    1.760796638088, np.nan,  1.9599639845401,                0,
    2.3307703209845,  .20530981936838,     11.352454199,  7.210981748e-30,
    1.92837046935,  2.7331701726189, np.nan,  1.9599639845401,
    0,    2.71338890728,  .29962471107816,  9.0559583604312,
    1.353737255e-19,  2.1261352646886,  3.3006425498714, np.nan,
    1.9599639845401,                0,    2.71338890728,  .29962471107816,
    9.0559583604312,  1.353737255e-19,  2.1261352646886,  3.3006425498714,
    np.nan,  1.9599639845401,                0, -3.9347864312059,
    1.2543868840549, -3.1368204508696,  .00170790683415, -6.3933395466329,
    -1.476233315779, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                         '3.agecat', '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .0171797439346, -.02561346650005,                0,  .00445310785396,
    .01204526460873,  .03142116278001,  .03142116278001, -.16245493266167,
    -.02561346650005,  .04968393937861,                0,  -.0069699735991,
    -.01845598801461, -.04723465558226, -.04723465558226,   .2326939064726,
    0,                0,                0,                0,
    0,                0,                0,                0,
    .00445310785396,  -.0069699735991,                0,  .03919624819724,
    .03254829669461,  .03756752462584,  .03756752462584, -.07124751761252,
    .01204526460873, -.01845598801461,                0,  .03254829669461,
    .04215212192908,  .05145895528528,  .05145895528528, -.14290240509701,
    .03142116278001, -.04723465558226,                0,  .03756752462584,
    .05145895528528,  .08977496748867,  .08977496748867, -.32621483141938,
    .03142116278001, -.04723465558226,                0,  .03756752462584,
    .05145895528528,  .08977496748867,  .08977496748867, -.32621483141938,
    -.16245493266167,   .2326939064726,                0, -.07124751761252,
    -.14290240509701, -.32621483141938, -.32621483141938,  1.5734864548889
    ]).reshape(8, 8)

cov_colnames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                '3.agecat', '4.agecat', '5.agecat', '_cons']

cov_rownames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                '3.agecat', '4.agecat', '5.agecat', '_cons']


results_noexposure_constraint = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=5,
           N=10,
           ic=3,
           k=7,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-38.45090497564205,
           k_eq_model=1,
           df_m=4,
           chi2=641.6446542589836,
           p=1.5005477751e-137,
           r2_p=np.nan,
           cmdline=("poisson deaths smokes i.agecat, "
                    "exposure(pyears) constraints(1)"),
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(pyears)",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vce="oim",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    .34304077058284,   .1073083520206,   3.196776058186,  .00138972774083,
    .13272026538212,  .55336127578356, np.nan,  1.9599639845401,
    0,                0, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    1.9599639845401,                0,  1.4846230896448,  .19510453584194,
    7.6093724999174,  2.754298692e-14,  1.1022252261742,  1.8670209531154,
    np.nan,  1.9599639845401,                0,  2.6284071093765,
    .18373002757074,  14.305811326156,  2.012766793e-46,  2.2683028724593,
    2.9885113462937, np.nan,  1.9599639845401,                0,
    3.4712405808805,  .17983994458502,  19.301833020969,  5.183735658e-83,
    3.1187607665121,  3.8237203952488, np.nan,  1.9599639845401,
    0,  3.4712405808805,  .17983994458502,  19.301833020969,
    5.183735658e-83,  3.1187607665121,  3.8237203952488, np.nan,
    1.9599639845401,                0, -7.9101515866812,  .19164951521841,
    -41.274049546467,                0, -8.2857777341639, -7.5345254391986,
    np.nan,  1.9599639845401,                0]).reshape(7, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                         '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .01151508241338,                0, -.00061947268694, -.00090708285562,
    -.00074959767622, -.00074959767622, -.00917958318314,                0,
    0,                0,                0,                0,
    0,                0, -.00061947268694,                0,
    .0380657799061,  .02946056271023,   .0294520905375,   .0294520905375,
    -.02891793401778, -.00090708285562,                0,  .02946056271023,
    .03375672303114,  .02947081310555,  .02947081310555, -.02868865719866,
    -.00074959767622,                0,   .0294520905375,  .02947081310555,
    .03234240566834,  .03234240566834, -.02881420109427, -.00074959767622,
    0,   .0294520905375,  .02947081310555,  .03234240566834,
    .03234240566834, -.02881420109427, -.00917958318314,                0,
    -.02891793401778, -.02868865719866, -.02881420109427, -.02881420109427,
    .03672953668345]).reshape(7, 7)

cov_colnames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']

cov_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']


results_exposure_constraint = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=6,
           N=10,
           ic=3,
           k=8,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-33.78306559091298,
           k_eq_model=1,
           df_m=5,
           chi2=526.719430888018,
           p=1.3614066522e-111,
           r2_p=np.nan,
           cmdline="poisson deaths lnpyears smokes i.agecat, constraints(2)",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vce="oim",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    1.1598786864273,  .13082965708054,  8.8655639119598,  7.611783820e-19,
    .90345727043975,  1.4163001024149, np.nan,  1.9599639845401,
    0,  .12111539473831,  .22317899375276,  .54268277090847,
    .58734823873758, -.31630739512299,  .55853818459962, np.nan,
    1.9599639845401,                0,                0, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    np.nan,  1.9599639845401,                0,  1.5276244194375,
    .19848759770871,  7.6963217705896,  1.400389019e-14,  1.1385958765506,
    1.9166529623245, np.nan,  1.9599639845401,                0,
    2.7415571106656,  .20647039325801,  13.278209371354,  3.097119459e-40,
    2.3368825760061,  3.1462316453252, np.nan,  1.9599639845401,
    0,   3.587300073596,  .30160673316211,  11.893965482753,
    1.272196529e-32,  2.9961617391034,  4.1784384080885, np.nan,
    1.9599639845401,                0,   4.087300073596,  .30160673316211,
    13.551753406643,  7.735990122e-42,  3.4961617391034,  4.6784384080885,
    np.nan,  1.9599639845401,                0, -9.4376201542802,
    1.2537557101599, -7.5274792990385,  5.172920628e-14, -11.894936191605,
    -6.9803041169553, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                         '3.agecat', '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .01711639917181, -.02559852137367,                0,  .00475026273828,
    .012305588195,  .03167368550108,  .03167368550108, -.16210959536359,
    -.02559852137367,   .0498088632525,                0, -.00783669874902,
    -.01946551099054,  -.0482099128044,  -.0482099128044,  .23336630265161,
    0,                0,                0,                0,
    0,                0,                0,                0,
    .00475026273828, -.00783669874902,                0,  .03939732644417,
    .0328943776068,   .0382554606876,   .0382554606876, -.07382466315002,
    .012305588195, -.01946551099054,                0,   .0328943776068,
    .04263002329212,  .05226051095238,  .05226051095238, -.14512177326509,
    .03167368550108,  -.0482099128044,                0,   .0382554606876,
    .05226051095238,  .09096662148872,  .09096662148872, -.32873181469848,
    .03167368550108,  -.0482099128044,                0,   .0382554606876,
    .05226051095238,  .09096662148872,  .09096662148872, -.32873181469848,
    -.16210959536359,  .23336630265161,                0, -.07382466315002,
    -.14512177326509, -.32873181469848, -.32873181469848,  1.5719033807586
    ]).reshape(8, 8)

cov_colnames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                '3.agecat', '4.agecat', '5.agecat', '_cons']

cov_rownames = ['lnpyears', 'smokes', '1b.agecat', '2.agecat',
                '3.agecat', '4.agecat', '5.agecat', '_cons']


results_noexposure_constraint2 = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=5,
           N=10,
           ic=3,
           k=7,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-34.5367006700131,
           k_eq_model=1,
           df_m=4,
           chi2=554.4168921897579,
           p=1.1331093797e-118,
           r2_p=np.nan,
           cmdline=("poisson deaths smokes i.agecat, "
                    "exposure(pyears) constraints(2)"),
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(pyears)",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vce="oim",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    .35978347114582,  .10730668667519,  3.3528522992687,  .00079983377167,
    .14946622996212,  .57010071232952, np.nan,  1.9599639845401,
    0,                0, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    1.9599639845401,                0,  1.4837272702102,  .19510269288329,
    7.6048528509946,  2.852282383e-14,  1.1013330188722,  1.8661215215483,
    np.nan,  1.9599639845401,                0,  2.6270956495127,
    .18372567328363,  14.299012231442,  2.219372691e-46,  2.2669999468414,
    2.987191352184, np.nan,  1.9599639845401,                0,
    3.2898291023835,  .17982035319735,  18.295087535352,  9.055555257e-75,
    2.9373876864294,  3.6422705183376, np.nan,  1.9599639845401,
    0,  3.7898291023835,  .17982035319735,  21.075640409983,
    1.330935038e-98,  3.4373876864294,  4.1422705183376, np.nan,
    1.9599639845401,                0, -7.9235211042587,  .19177810950798,
    -41.316087245761,                0, -8.2993992919175, -7.5476429165999,
    np.nan,  1.9599639845401,                0]).reshape(7, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                         '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .01151472500521,                0, -.00061274288972, -.00089685568608,
    -.00069335681347, -.00069335681347, -.00921031399899,                0,
    0,                0,                0,                0,
    0,                0, -.00061274288972,                0,
    .03806506077031,  .02945948985187,  .02944866089267,  .02944866089267,
    -.02892164840477, -.00089685568608,                0,  .02945948985187,
    .03375512302352,  .02946576868665,  .02946576868665,  -.0286943943397,
    -.00069335681347,                0,  .02944866089267,  .02946576868665,
    .03233535942402,  .03233535942402, -.02885716752919, -.00069335681347,
    0,  .02944866089267,  .02946576868665,  .03233535942402,
    .03233535942402, -.02885716752919, -.00921031399899,                0,
    -.02892164840477,  -.0286943943397, -.02885716752919, -.02885716752919,
    .03677884328645]).reshape(7, 7)

cov_colnames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']

cov_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']


results_exposure_constraint2 = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )


est = dict(
           rank=5,
           N=10,
           ic=3,
           k=7,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-34.5367006700131,
           k_eq_model=1,
           df_m=4,
           chi2=582.5215805315736,
           p=9.3932644024e-125,
           r2_p=np.nan,
           cmdline=("poisson deaths smokes i.agecat,"
                    "exposure(pyears) constraints(2) vce(robust)"),
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(pyears)",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vcetype="Robust",
           vce="robust",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log pseudolikelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="deaths",
           properties="b V",
          )

params_table = np.array([
    .35978347114582,   .1172393358046,  3.0687948603312,  .00214924117257,
    .1299985953974,  .58956834689424, np.nan,  1.9599639845401,
    0,                0, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan,
    1.9599639845401,                0,  1.4837272702102,  .21969092615175,
    6.7537030145039,  1.441186055e-11,  1.0531409672225,  1.9143135731979,
    np.nan,  1.9599639845401,                0,  2.6270956495127,
    .20894895542061,  12.572906355164,  2.975796525e-36,   2.217563222281,
    3.0366280767443, np.nan,  1.9599639845401,                0,
    3.2898291023835,   .2211846822073,  14.873675109654,  4.885611722e-50,
    2.8563150913252,  3.7233431134417, np.nan,  1.9599639845401,
    0,  3.7898291023835,   .2211846822073,  17.134229479922,
    8.243780087e-66,  3.3563150913252,  4.2233431134417, np.nan,
    1.9599639845401,                0, -7.9235211042587,   .2479876721169,
    -31.951270144281,  5.18748229e-224, -8.4095680102177, -7.4374741982996,
    np.nan,  1.9599639845401,                0]).reshape(7, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                         '4.agecat', '5.agecat', '_cons']

cov = np.array([
    .0137450618599,                0,  .00249770233028,  .00412347653263,
    .00486142402447,  .00486142402447, -.01620342093134,                0,
    0,                0,                0,                0,
    0,                0,  .00249770233028,                0,
    .04826410303341,  .04389964215014,  .04391744129373,  .04391744129373,
    -.04609122424924,  .00412347653263,                0,  .04389964215014,
    .04365966597136,  .04367917402468,  .04367917402468, -.04726310745444,
    .00486142402447,                0,  .04391744129373,  .04367917402468,
    .04892266364314,  .04892266364314, -.04794543190806,  .00486142402447,
    0,  .04391744129373,  .04367917402468,  .04892266364314,
    .04892266364314, -.04794543190806, -.01620342093134,                0,
    -.04609122424924, -.04726310745444, -.04794543190806, -.04794543190806,
    .06149788552196]).reshape(7, 7)

cov_colnames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']

cov_rownames = ['smokes', '1b.agecat', '2.agecat', '3.agecat',
                '4.agecat', '5.agecat', '_cons']


results_exposure_constraint2_robust = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )
