import numpy as np

from statsmodels.tools.testing import MarginTableTestBunch

est = dict(
           rank=7,
           N=17,
           ic=6,
           k=7,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-28.46285727296058,
           k_eq_model=1,
           ll_0=-101.6359341820935,
           df_m=6,
           chi2=146.3461538182658,
           p=4.58013206701e-29,
           r2_p=.719952814897477,
           properties="b V",
           depvar="sexecutions",
           which="max",
           technique="nr",
           singularHmethod="m-marquardt",
           ml_method="e2",
           crittype="log likelihood",
           user="poiss_lf",
           title="Poisson regression",
           vce="oim",
           opt="moptimize",
           chi2type="LR",
           gof="poiss_g",
           estat_cmd="poisson_estat",
           predict="poisso_p",
           cmd="poisson",
           cmdline="poisson sexecutions sincome sperpoverty sperblack LN_VC100k96 south sdegree",  # noqa:E501
          )

margins_table = np.array([
    47.514189267677,  12.722695157081,  3.7346009380122,    .000188013074,
    22.578164973516,  72.450213561838, np.nan,  1.9599639845401,
    0,  2.3754103372885,  7.6314378245266,  .31126642081184,
    .75559809249357,  -12.58193294904,  17.332753623617, np.nan,
    1.9599639845401,                0, -11.583732327397,  3.8511214886273,
    -3.007885459237,  .00263072269737, -19.131791745195, -4.0356729095995,
    np.nan,  1.9599639845401,                0,  -1.807106397978,
    14.19277372084, -.12732580914219,  .89868253380624, -29.624431731551,
    26.010218935595, np.nan,  1.9599639845401,                0,
    10.852916363139,  2.6197368291491,  4.1427506161617,  .00003431650408,
    5.7183265290336,  15.987506197244, np.nan,  1.9599639845401,
    0, -26.588397789444,  7.6315578612519, -3.4840065780596,
    .00049396734722, -41.545976343431, -11.630819235457, np.nan,
    1.9599639845401,                0]).reshape(6, 9)

margins_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

margins_table_rownames = ['sincome', 'sperpoverty', 'sperblack',
                          'LN_VC100k96', 'south', 'sdegree']

margins_cov = np.array([
    10.87507957467,  3.4816608831283,  .87483487811437,  3.1229403520191,
    -.87306122632875, -2.2870394487277, -12.321063650937,  3.4816608831283,
    5.1715652306254,  .27473956091394,  1.7908952063684, -.92880259796684,
    1.8964947971413, -9.0063087868006,  .87483487811437,  .27473956091394,
    1.1098392181639, -.99390727840297, -.34477731736542, -.98869834020742,
    .41772084541889,  3.1229403520191,  1.7908952063684, -.99390727840297,
    17.912620004361, -.30763138390107,  2.8490197200257, -21.269786576194,
    -.87306122632875, -.92880259796684, -.34477731736542, -.30763138390107,
    .42666000427673,  .05265352402592,   1.461997775289, -2.2870394487277,
    1.8964947971413, -.98869834020742,  2.8490197200257,  .05265352402592,
    4.0773252373088,   -4.46154120848, -12.321063650937, -9.0063087868006,
    .41772084541889, -21.269786576194,   1.461997775289,   -4.46154120848,
    37.559994394326]).reshape(7, 7)

margins_cov_colnames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        'south', 'sdegree', '_cons']

margins_cov_rownames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        'south', 'sdegree', '_cons']


results_poisson_margins_cont = MarginTableTestBunch(
                margins_table=margins_table,
                margins_table_colnames=margins_table_colnames,
                margins_table_rownames=margins_table_rownames,
                margins_cov=margins_cov,
                margins_cov_colnames=margins_cov_colnames,
                margins_cov_rownames=margins_cov_rownames,
                **est
                )

est = dict(
           alpha=1.1399915663048,
           rank=8,
           N=17,
           ic=6,
           k=8,
           k_eq=2,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-27.58269157281191,
           k_eq_model=1,
           ll_0=-32.87628220135203,
           rank0=2,
           df_m=6,
           chi2=10.58718125708024,
           p=.1020042170100994,
           ll_c=-28.46285727296058,
           chi2_c=1.760331400297339,
           r2_p=.1610154881905236,
           k_aux=1,
           properties="b V",
           depvar="sexecutions",
           which="max",
           technique="nr",
           singularHmethod="m-marquardt",
           ml_method="e2",
           crittype="log likelihood",
           user="nbreg_lf",
           diparm1="lnalpha, exp label(",
           title="Negative binomial regression",
           vce="oim",
           opt="moptimize",
           chi2type="LR",
           chi2_ct="LR",
           diparm_opt2="noprob",
           dispers="mean",
           predict="nbreg_p",
           cmd="nbreg",
           cmdline="nbreg sexecutions sincome sperpoverty sperblack LN_VC100k96 south sdegree",  # noqa:E501
          )

margins_table = np.array([
    38.76996449636,  35.863089953808,  1.0810547709719,  .27967275079666,
    -31.520400187424,  109.06032918014, np.nan,  1.9599639845401,
    0,  2.5208248279391,  11.710699937092,  .21525825454332,
    .82956597472339, -20.431725282518,  25.473374938396, np.nan,
    1.9599639845401,                0,  -8.225606184332,   9.557721280021,
    -.86062419517573,  .38944505570119, -26.958395667445,  10.507183298781,
    np.nan,  1.9599639845401,                0, -4.4150939806524,
    28.010544627225, -.15762256819387,  .87475421903252, -59.314752637366,
    50.484564676062, np.nan,  1.9599639845401,                0,
    7.0049476220304,  6.3399264323903,  1.1048941492826,  .26920545789466,
    -5.4210798500881,  19.430975094149, np.nan,  1.9599639845401,
    0, -25.128303596214,  23.247820190364, -1.0808885904335,
    .279746674501, -70.693193888391,  20.436586695964, np.nan,
    1.9599639845401,                0]).reshape(6, 9)

margins_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

margins_table_rownames = ['sincome', 'sperpoverty', 'sperblack',
                          'LN_VC100k96', 'south', 'sdegree']

margins_cov = np.array([
    44.468037032422,  13.291812805254,  .84306554343753, -.38095027773819,
    -2.1265212254924,  -18.06714825989, -30.427077474507,  .36347806905257,
    13.291812805254,  15.093124820143,  3.3717840254072, -7.6860995498613,
    -3.3867901970823, -1.4200645173727, -12.979849717094,  .51706617429388,
    .84306554343753,  3.3717840254072,  5.6928040093481, -12.140553562993,
    -2.5831646721297, -1.8071496111137,   7.961664784177,  .27439267406128,
    -.38095027773819, -7.6860995498613, -12.140553562993,  91.950706114029,
    6.6107070350689,  9.5470604840407, -82.665769963947, -1.1433180909155,
    -2.1265212254924, -3.3867901970823, -2.5831646721297,  6.6107070350689,
    2.0499053083335,  1.7094543055869,  -3.029543334606, -.34297224102579,
    -18.06714825989, -1.4200645173727, -1.8071496111137,  9.5470604840407,
    1.7094543055869,  18.442703265156, -6.5839965105886, -.61952491151176,
    -30.427077474507, -12.979849717094,   7.961664784177, -82.665769963947,
    -3.029543334606, -6.5839965105886,  111.12618806587,  .88600743091011,
    .36347806905257,  .51706617429388,  .27439267406128, -1.1433180909155,
    -.34297224102579, -.61952491151176,  .88600743091011,  .71851239110057
    ]).reshape(8, 8)

margins_cov_colnames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        'south', 'sdegree', '_cons', '_cons']

margins_cov_rownames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        'south', 'sdegree', '_cons', '_cons']


results_negbin_margins_cont = MarginTableTestBunch(
                margins_table=margins_table,
                margins_table_colnames=margins_table_colnames,
                margins_table_rownames=margins_table_rownames,
                margins_cov=margins_cov,
                margins_cov_colnames=margins_cov_colnames,
                margins_cov_rownames=margins_cov_rownames,
                **est
                )

est = dict(
           rank=7,
           N=17,
           ic=6,
           k=8,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-28.46285727296058,
           k_eq_model=1,
           ll_0=-101.6359341820935,
           df_m=6,
           chi2=146.3461538182658,
           p=4.58013206701e-29,
           r2_p=.719952814897477,
           properties="b V",
           depvar="sexecutions",
           which="max",
           technique="nr",
           singularHmethod="m-marquardt",
           ml_method="e2",
           crittype="log likelihood",
           user="poiss_lf",
           title="Poisson regression",
           vce="oim",
           opt="moptimize",
           chi2type="LR",
           gof="poiss_g",
           estat_cmd="poisson_estat",
           predict="poisso_p",
           cmd="poisson",
           cmdline="poisson sexecutions sincome sperpoverty sperblack LN_VC100k96 i.south sdegree",  # noqa:E501
          )

margins_table = np.array([
    47.514189267677,   12.72269515678,  3.7346009381004,  .00018801307393,
    22.578164974105,  72.450213561249, np.nan,  1.9599639845401,
    0,  2.3754103372885,  7.6314378245485,  .31126642081095,
    .75559809249425, -12.581932949083,   17.33275362366, np.nan,
    1.9599639845401,                0, -11.583732327397,  3.8511214887188,
    -3.0078854591656,  .00263072269799, -19.131791745374, -4.0356729094203,
    np.nan,  1.9599639845401,                0,  -1.807106397978,
    14.192773720841, -.12732580914219,  .89868253380624, -29.624431731552,
    26.010218935596, np.nan,  1.9599639845401,                0,
    0, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan,  1.9599639845401,
    0,  12.894515685772,  5.7673506886042,  2.2357779822979,
    .02536631788468,  1.5907160498956,  24.198315321648, np.nan,
    1.9599639845401,                0, -26.588397789444,  7.6315578608763,
    -3.4840065782311,  .00049396734691, -41.545976342695, -11.630819236193,
    np.nan,  1.9599639845401,                0]).reshape(7, 9)

margins_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

margins_table_rownames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                          '0b.south', '1.south', 'sdegree']

margins_cov = np.array([
    10.875079574674,  3.4816608831298,  .87483487811447,  3.1229403520208,
    0,   -.873061226329, -2.2870394487282, -12.321063650942,
    3.4816608831298,  5.1715652306252,  .27473956091396,  1.7908952063684,
    0, -.92880259796679,  1.8964947971405, -9.0063087868012,
    .87483487811447,  .27473956091396,   1.109839218164,  -.9939072784041,
    0, -.34477731736544, -.98869834020768,  .41772084541996,
    3.1229403520208,  1.7908952063684,  -.9939072784041,  17.912620004373,
    0, -.30763138390086,  2.8490197200274, -21.269786576207,
    0,                0,                0,                0,
    0,                0,                0,                0,
    -.873061226329, -.92880259796679, -.34477731736544, -.30763138390086,
    0,  .42666000427672,  .05265352402609,  1.4619977752889,
    -2.2870394487282,  1.8964947971405, -.98869834020768,  2.8490197200274,
    0,  .05265352402609,  4.0773252373089, -4.4615412084808,
    -12.321063650942, -9.0063087868012,  .41772084541996, -21.269786576207,
    0,  1.4619977752889, -4.4615412084808,  37.559994394343
    ]).reshape(8, 8)

margins_cov_colnames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        '0b.south', '1.south', 'sdegree', '_cons']

margins_cov_rownames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        '0b.south', '1.south', 'sdegree', '_cons']


results_poisson_margins_dummy = MarginTableTestBunch(
                margins_table=margins_table,
                margins_table_colnames=margins_table_colnames,
                margins_table_rownames=margins_table_rownames,
                margins_cov=margins_cov,
                margins_cov_colnames=margins_cov_colnames,
                margins_cov_rownames=margins_cov_rownames,
                **est
                )

est = dict(
           alpha=1.139991566304804,
           rank=8,
           N=17,
           ic=6,
           k=9,
           k_eq=2,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=1,
           ll=-27.58269157281191,
           k_eq_model=1,
           ll_0=-32.87628220135203,
           rank0=2,
           df_m=6,
           chi2=10.58718125708025,
           p=.1020042170100991,
           ll_c=-28.46285727296058,
           chi2_c=1.760331400297339,
           r2_p=.1610154881905237,
           k_aux=1,
           properties="b V",
           depvar="sexecutions",
           which="max",
           technique="nr",
           singularHmethod="m-marquardt",
           ml_method="e2",
           crittype="log likelihood",
           user="nbreg_lf",
           diparm1="lnalpha, exp label(",
           title="Negative binomial regression",
           vce="oim",
           opt="moptimize",
           chi2type="LR",
           chi2_ct="LR",
           diparm_opt2="noprob",
           dispers="mean",
           predict="nbreg_p",
           cmd="nbreg",
           cmdline="nbreg sexecutions sincome sperpoverty sperblack LN_VC100k96 i.south sdegree",  # noqa:E501
          )

margins_table = np.array([
    38.769964496355,  35.863089979665,  1.0810547701924,  .27967275114341,
    -31.520400238107,  109.06032923082, np.nan,  1.9599639845401,
    0,  2.5208248279388,  11.710699937639,  .21525825453324,
    .82956597473124,  -20.43172528359,  25.473374939467, np.nan,
    1.9599639845401,                0, -8.2256061843309,  9.5577212853699,
    -.86062419469397,  .38944505596662, -26.958395677928,  10.507183309266,
    np.nan,  1.9599639845401,                0, -4.4150939806521,
    28.010544626815, -.15762256819618,  .87475421903071, -59.314752636561,
    50.484564675257, np.nan,  1.9599639845401,                0,
    0, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan,  1.9599639845401,
    0,  8.0380552593041,  8.8634487485248,  .90687671214231,
    .36447199739385, -9.3339850666211,  25.410095585229, np.nan,
    1.9599639845401,                0,  -25.12830359621,  23.247820207656,
    -1.0808885896294,  .27974667485873, -70.693193922279,  20.436586729858,
    np.nan,  1.9599639845401,                0]).reshape(7, 9)

margins_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

margins_table_rownames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                          '0b.south', '1.south', 'sdegree']


margins_cov = np.array([
    44.468037032424,  13.291812805256,  .84306554343906, -.38095027774827,
    0, -2.1265212254934, -18.067148259892, -30.427077474499,
    .36347806905277,  13.291812805256,  15.093124820144,  3.3717840254072,
    -7.6860995498609,                0, -3.3867901970823, -1.4200645173736,
    -12.979849717095,  .51706617429393,  .84306554343906,  3.3717840254072,
    5.6928040093478,  -12.14055356299,                0, -2.5831646721296,
    -1.8071496111144,  7.9616647841741,  .27439267406129, -.38095027774827,
    -7.6860995498609,  -12.14055356299,  91.950706114005,                0,
    6.6107070350678,  9.5470604840447, -82.665769963921, -1.1433180909154,
    0,                0,                0,                0,
    0,                0,                0,                0,
    0, -2.1265212254934, -3.3867901970823, -2.5831646721296,
    6.6107070350678,                0,  2.0499053083335,  1.7094543055874,
    -3.0295433346046, -.34297224102581, -18.067148259892, -1.4200645173736,
    -1.8071496111144,  9.5470604840447,                0,  1.7094543055874,
    18.442703265157, -6.5839965105912, -.61952491151187, -30.427077474499,
    -12.979849717095,  7.9616647841741, -82.665769963921,                0,
    -3.0295433346046, -6.5839965105912,  111.12618806584,  .88600743090998,
    .36347806905277,  .51706617429393,  .27439267406129, -1.1433180909154,
    0, -.34297224102581, -.61952491151187,  .88600743090998,
    .71851239110059]).reshape(9, 9)

margins_cov_colnames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        '0b.south', '1.south', 'sdegree', '_cons', '_cons']

margins_cov_rownames = ['sincome', 'sperpoverty', 'sperblack', 'LN_VC100k96',
                        '0b.south', '1.south', 'sdegree', '_cons', '_cons']


results_negbin_margins_dummy = MarginTableTestBunch(
                margins_table=margins_table,
                margins_table_colnames=margins_table_colnames,
                margins_table_rownames=margins_table_rownames,
                margins_cov=margins_cov,
                margins_cov_colnames=margins_cov_colnames,
                margins_cov_rownames=margins_cov_rownames,
                **est
                )
