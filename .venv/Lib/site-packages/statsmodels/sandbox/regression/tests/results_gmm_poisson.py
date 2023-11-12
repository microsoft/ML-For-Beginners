import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
    rank=8,
    N=3629,
    Q=4.59536484786e-20,
    J=1.66765790329e-16,
    J_df=0,
    k_1=8,
    converged=1,
    has_xtinst=0,
    type=1,
    n_eq=1,
    k=8,
    n_moments=8,
    k_aux=8,
    k_eq_model=0,
    k_eq=8,
    cmdline="gmm ( docvis - exp({xb:private medicaid aget aget2 educyr actlim totchr}+{b0})), instruments(incomet ssiratio aget aget2 educyr actlim totchr) onestep vce(robust)",  # noqa:E501
    cmd="gmm",
    estat_cmd="gmm_estat",
    predict="gmm_p",
    marginsnotok="_ALL",
    eqnames="1",
    technique="gn",
    winit="Unadjusted",
    estimator="onestep",
    wmatrix="robust",
    vce="robust",
    vcetype="Robust",
    params="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    inst_1="incomet ssiratio aget aget2 educyr actlim totchr _cons",
    params_1="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    sexp_1="docvis - exp( ({xb_private} *private + {xb_medicaid} *medicaid + {xb_aget} *aget + {xb_aget2} *aget2 + {xb_educyr} *educyr + {xb_actlim} *actlim + {xb_totchr} *totchr) + {b0} )",  # noqa:E501
    properties="b V",
)

params_table = np.array([
    .62093805844748,  .35860052573857,   1.731559252928,  .08335206643438,
    -.08190605683724,  1.3237821737322, np.nan,  1.9599639845401,
    0,  .68895699568302,  .43817618784254,  1.5723286997298,
    .11587434043505,  -.1698525513714,  1.5477665427374, np.nan,
    1.9599639845401,                0,  .25750627258076,  .05009451793791,
    5.1404082358855,  2.741421857e-07,  .15932282159956,  .35568972356197,
    np.nan,  1.9599639845401,                0, -.05352997420414,
    .01103202674353, -4.8522339048464,  1.220785200e-06, -.07515234929795,
    -.03190759911034, np.nan,  1.9599639845401,                0,
    .03106248018916,  .01032090201131,  3.0096671933432,  .00261534090329,
    .01083388395902,  .05129107641931, np.nan,  1.9599639845401,
    0,  .14175365608301,   .0494498280382,  2.8666157539212,
    .00414886404159,  .04483377408643,  .23867353807958, np.nan,
    1.9599639845401,                0,  .23128095221422,  .01565221628818,
    14.776243054406,  2.084750820e-49,  .20060317201116,  .26195873241727,
    np.nan,  1.9599639845401,                0,  .34763567088735,
    .31615794015526,  1.0995633091379,  .27152243570261, -.27202250524333,
    .96729384701803, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['_cons'] * 8

cov = np.array([
    .12859433705998,  .13265896898444,  .00910916927048, -.00144786113189,
    -.00037337560793, -.00152379041042, -.00336772308907, -.09899309651531,
    .13265896898444,  .19199837159222,  .00979636564963, -.00135323134276,
    .00180599814286, -.00930935415071, -.00460031335865, -.13429156867927,
    .00910916927048,  .00979636564963,  .00250946072743, -.00052373946978,
    5.155389870e-07, -.00016461502154, -.00025816911604, -.00869892550441,
    -.00144786113189, -.00135323134276, -.00052373946978,  .00012170561407,
    8.334416260e-06, -.00002526568199,  .00003797456789,  .00131001446811,
    -.00037337560793,  .00180599814286,  5.155389870e-07,  8.334416260e-06,
    .00010652101833, -.00026856403693, -.00003344387872, -.00122933496346,
    -.00152379041042, -.00930935415071, -.00016461502154, -.00002526568199,
    -.00026856403693,  .00244528549301,  .00003610001892,  .00527355381855,
    -.00336772308907, -.00460031335865, -.00025816911604,  .00003797456789,
    -.00003344387872,  .00003610001892,  .00024499187473,  .00300075896709,
    -.09899309651531, -.13429156867927, -.00869892550441,  .00131001446811,
    -.00122933496346,  .00527355381855,  .00300075896709,  .09995584312322
    ]).reshape(8, 8)

cov_colnames = ['_cons'] * 8

cov_rownames = ['_cons'] * 8


results_addonestep = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)


est = dict(
    rank=8,
    N=3629,
    Q=6.09567389485e-33,
    J=2.21212005644e-29,
    J_df=0,
    k_1=8,
    converged=1,
    has_xtinst=0,
    type=1,
    n_eq=1,
    k=8,
    n_moments=8,
    k_aux=8,
    k_eq_model=0,
    k_eq=8,
    cmdline="gmm ( docvis - exp({xb:private medicaid aget aget2 educyr actlim totchr}+{b0})), instruments(incomet ssiratio aget aget2 educyr actlim totchr) twostep vce(robust)",  # noqa:E501
    cmd="gmm",
    estat_cmd="gmm_estat",
    predict="gmm_p",
    marginsnotok="_ALL",
    eqnames="1",
    technique="gn",
    winit="Unadjusted",
    estimator="twostep",
    wmatrix="robust",
    vce="robust",
    vcetype="Robust",
    params="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    inst_1="incomet ssiratio aget aget2 educyr actlim totchr _cons",
    params_1="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    sexp_1="docvis - exp( ({xb_private} *private + {xb_medicaid} *medicaid + {xb_aget} *aget + {xb_aget2} *aget2 + {xb_educyr} *educyr + {xb_actlim} *actlim + {xb_totchr} *totchr) + {b0} )",  # noqa:E501
    properties="b V",
    )

params_table = np.array([
    .6209380584426,  .35860052570457,  1.7315592530786,  .08335206640755,
    -.08190605677548,  1.3237821736607, np.nan,  1.9599639845401,
    0,  .68895699501744,  .43817618789764,  1.5723286980131,
    .11587434083298, -.16985255214498,  1.5477665421799, np.nan,
    1.9599639845401,                0,  .25750627271754,  .05009451794125,
    5.1404082382732,  2.741421823e-07,  .15932282172979,  .35568972370529,
    np.nan,  1.9599639845401,                0, -.05352997423123,
    .01103202674378, -4.8522339071944,  1.220785186e-06, -.07515234932551,
    -.03190759913694, np.nan,  1.9599639845401,                0,
    .03106248018903,  .01032090201422,  3.0096671924822,   .0026153409107,
    .01083388395319,  .05129107642488, np.nan,  1.9599639845401,
    0,  .14175365616691,  .04944982804302,  2.8666157553386,
    .00414886402301,  .04483377416089,  .23867353817294, np.nan,
    1.9599639845401,                0,  .23128095224221,  .01565221628892,
    14.776243055497,  2.084750786e-49,  .20060317203771,  .26195873244672,
    np.nan,  1.9599639845401,                0,  .34763567064032,
    .31615794015859,   1.099563308345,  .27152243604826, -.27202250549689,
    .96729384677754, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['_cons'] * 8

cov = np.array([
    .12859433703559,   .1326589689683,  .00910916927021, -.00144786113188,
    -.00037337560766, -.00152379040753, -.00336772308885, -.09899309649807,
    .1326589689683,   .1919983716405,  .00979636565235, -.00135323134324,
    .00180599814488, -.00930935415256, -.00460031335946, -.13429156869395,
    .00910916927021,  .00979636565235,  .00250946072777, -.00052373946983,
    5.155391569e-07, -.00016461502162, -.00025816911611, -.00869892550672,
    -.00144786113188, -.00135323134324, -.00052373946983,  .00012170561408,
    8.334416227e-06, -.00002526568198,   .0000379745679,  .00131001446858,
    -.00037337560766,  .00180599814488,  5.155391569e-07,  8.334416227e-06,
    .00010652101839, -.00026856403706, -.00003344387875, -.00122933496459,
    -.00152379040753, -.00930935415256, -.00016461502162, -.00002526568198,
    -.00026856403706,  .00244528549348,  .00003610001887,  .00527355381795,
    -.00336772308885, -.00460031335946, -.00025816911611,   .0000379745679,
    -.00003344387875,  .00003610001887,  .00024499187475,  .00300075896724,
    -.09899309649807, -.13429156869395, -.00869892550672,  .00131001446858,
    -.00122933496459,  .00527355381795,  .00300075896724,  .09995584312533
    ]).reshape(8, 8)

cov_colnames = ['_cons'] * 8

cov_rownames = ['_cons'] * 8

results_addtwostep = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)


est = dict(
    rank=8,
    N=3629,
    Q=.0002538911897719,
    J=.9213711276820714,
    J_df=1,
    k_1=8,
    converged=1,
    has_xtinst=0,
    type=1,
    n_eq=1,
    k=8,
    n_moments=9,
    k_aux=8,
    k_eq_model=0,
    k_eq=8,
    cmdline="gmm ( (docvis / exp({xb:private medicaid aget aget2 educyr actlim totchr}+{b0})) - 1), instruments(income ssiratio medicaid aget aget2 educyr actlim totchr) onestep vce(robust)",  # noqa:E501
    cmd="gmm",
    estat_cmd="gmm_estat",
    predict="gmm_p",
    marginsnotok="_ALL",
    eqnames="1",
    technique="gn",
    winit="Unadjusted",
    estimator="onestep",
    wmatrix="robust",
    vce="robust",
    vcetype="Robust",
    params="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    inst_1="income ssiratio medicaid aget aget2 educyr actlim totchr _cons",
    params_1="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    sexp_1="(docvis / exp( ({xb_private} *private + {xb_medicaid} *medicaid + {xb_aget} *aget + {xb_aget2} *aget2 + {xb_educyr} *educyr + {xb_actlim} *actlim + {xb_totchr} *totchr) + {b0} )) - 1",  # noqa:E501
    properties="b V",
)

params_table = np.array([
    .67045580921478,  .25039046077656,  2.6776411814389,  .00741425985435,
    .17969952402034,  1.1612120944092, np.nan,  1.9599639845401,
    0,  .28551241628798,  .10358919281318,  2.7561988710819,
    .00584774303307,  .08248132918657,   .4885435033894, np.nan,
    1.9599639845401,                0,   .2672004738793,  .05203985579809,
    5.1345352476769,  2.828420839e-07,  .16520423075439,  .36919671700421,
    np.nan,  1.9599639845401,                0,  -.0560702624564,
    .01191485946838, -4.7059105149509,  2.527353692e-06, -.07942295789528,
    -.03271756701753, np.nan,  1.9599639845401,                0,
    .01448379701656,  .00782559934942,  1.8508227127214,  .06419506241955,
    -.00085409586574,  .02982168989887, np.nan,  1.9599639845401,
    0,  .18130374188096,   .0382173439987,  4.7440173206998,
    2.095209222e-06,  .10639912405874,  .25620835970318, np.nan,
    1.9599639845401,                0,  .28146161235562,  .01380395117777,
    20.389931022715,  2.054354003e-92,  .25440636520284,  .30851685950839,
    np.nan,  1.9599639845401,                0,  .51399857133918,
    .10262653035745,  5.0084375799215,  5.487366567e-07,  .31285426798028,
    .71514287469808, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['_cons'] * 8

cov = np.array([
    .0626953828479,  .02323594786658,  .00535172023578, -.00103050587759,
    -.00154311442856,  .00154515839603, -.00043159973572, -.01570852578318,
    .02323594786658,  .01073072086769,  .00207768328305, -.00039713375955,
    -.00049396171685,  .00027652302157, -.00020408147523, -.00701276303887,
    .00535172023578,  .00207768328305,  .00270814659149, -.00059652725999,
    -.00012298559534,  .00021079055266, -.00004341699196,  -.0031278522429,
    -.00103050587759, -.00039713375955, -.00059652725999,  .00014196387615,
    .00002481291175, -.00006035908648,  .00001093157006,  .00059187926133,
    -.00154311442856, -.00049396171685, -.00012298559534,  .00002481291175,
    .00006124000518, -.00001857594061,  .00001436652009,  .00008106194688,
    .00154515839603,  .00027652302157,  .00021079055266, -.00006035908648,
    -.00001857594061,  .00146056538231, -.00016708887634, -.00074321753343,
    -.00043159973572, -.00020408147523, -.00004341699196,  .00001093157006,
    .00001436652009, -.00016708887634,  .00019054906812, -.00028024031412,
    -.01570852578318, -.00701276303887,  -.0031278522429,  .00059187926133,
    .00008106194688, -.00074321753343, -.00028024031412,  .01053220473321
    ]).reshape(8, 8)

cov_colnames = ['_cons'] * 8

cov_rownames = ['_cons'] * 8


results_multonestep = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)


est = dict(
    rank=8,
    N=3629,
    Q=.0002589826272982,
    J=.9398479544653281,
    J_df=1,
    k_1=8,
    converged=1,
    has_xtinst=0,
    type=1,
    n_eq=1,
    k=8,
    n_moments=9,
    k_aux=8,
    k_eq_model=0,
    k_eq=8,
    cmdline="gmm ( (docvis / exp({xb:private medicaid aget aget2 educyr actlim totchr}+{b0})) - 1), instruments(income ssiratio medicaid aget aget2 educyr actlim totchr) twostep vce(robust)",  # noqa:E501
    cmd="gmm",
    estat_cmd="gmm_estat",
    predict="gmm_p",
    marginsnotok="_ALL",
    eqnames="1",
    technique="gn",
    winit="Unadjusted",
    estimator="twostep",
    wmatrix="robust",
    vce="robust",
    vcetype="Robust",
    params="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    inst_1="income ssiratio medicaid aget aget2 educyr actlim totchr _cons",
    params_1="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    sexp_1="(docvis / exp( ({xb_private} *private + {xb_medicaid} *medicaid + {xb_aget} *aget + {xb_aget2} *aget2 + {xb_educyr} *educyr + {xb_actlim} *actlim + {xb_totchr} *totchr) + {b0} )) - 1",  # noqa:E501
    properties="b V",
    )

params_table = np.array([
    .67815288158883,  .25053953449054,  2.7067699433856,  .00679413212727,
    .18710441728393,  1.1692013458937, np.nan,  1.9599639845401,
    0,  .28872837589732,   .1032733938985,  2.7957672833051,
    .00517766683505,  .08631624329503,  .49114050849961, np.nan,
    1.9599639845401,                0,  .27067071818542,  .05199695467114,
    5.2055109745809,  1.934635127e-07,  .16875855972422,  .37258287664662,
    np.nan,  1.9599639845401,                0, -.05690856524563,
    .01189861686254, -4.7827882772482,  1.728801925e-06, -.08022942576205,
    -.03358770472921, np.nan,  1.9599639845401,                0,
    .01438118999252,  .00783219080428,  1.8361644081315,  .06633334485657,
    -.00096962190392,  .02973200188896, np.nan,  1.9599639845401,
    0,  .18038262255626,  .03826653224544,  4.7138481584715,
    2.430818311e-06,  .10538159754195,  .25538364757056, np.nan,
    1.9599639845401,                0,  .28251027986119,  .01378475918788,
    20.494393555287,  2.415775858e-93,  .25549264831739,  .30952791140498,
    np.nan,  1.9599639845401,                0,   .5077134442587,
    .10235830367214,  4.9601588346456,  7.043556343e-07,  .30709485554269,
    .7083320329747, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['_cons'] * 8

cov = np.array([
    .06277005834274,  .02315710174743,  .00533574120292, -.00102544979294,
    -.00154463417995,   .0015508406274, -.00043796451278, -.01559999387335,
    .02315710174743,  .01066539388732,  .00206217803508, -.00039331197813,
    -.00049172930967,  .00027603135609, -.00020644763374, -.00694810289238,
    .00533574120292,  .00206217803508,  .00270368329507,  -.0005950942106,
    -.00012276584915,  .00021462173623, -.00004681980342, -.00310767551047,
    -.00102544979294, -.00039331197813,  -.0005950942106,  .00014157708324,
    .00002474211336, -.00006134660609,  .00001178280314,  .00058658157366,
    -.00154463417995, -.00049172930967, -.00012276584915,  .00002474211336,
    .00006134321279, -.00001855941375,  .00001443470174,   .0000776612477,
    .0015508406274,  .00027603135609,  .00021462173623, -.00006134660609,
    -.00001855941375,  .00146432749009, -.00016643326394, -.00074847803836,
    -.00043796451278, -.00020644763374, -.00004681980342,  .00001178280314,
    .00001443470174, -.00016643326394,  .00019001958587, -.00027573517109,
    -.01559999387335, -.00694810289238, -.00310767551047,  .00058658157366,
    .0000776612477, -.00074847803836, -.00027573517109,  .01047722233064
    ]).reshape(8, 8)

cov_colnames = ['_cons'] * 8

cov_rownames = ['_cons'] * 8

results_multtwostep = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)


est = dict(
    rank=8,
    N=3629,
    Q=.0002590497181628,
    J=.940091427212973,
    J_df=1,
    k_1=8,
    converged=1,
    has_xtinst=0,
    type=1,
    n_eq=1,
    k=8,
    n_moments=9,
    k_aux=8,
    k_eq_model=0,
    k_eq=8,
    cmdline="gmm ( (docvis / exp({xb:private medicaid aget aget2 educyr actlim totchr}+{b0})) - 1), instruments(income ssiratio medicaid aget aget2 educyr actlim totchr) twostep wmatrix(robust) vce(unadjusted) center",  # noqa:E501
    cmd="gmm",
    estat_cmd="gmm_estat",
    predict="gmm_p",
    marginsnotok="_ALL",
    eqnames="1",
    technique="gn",
    winit="Unadjusted",
    estimator="twostep",
    wmatrix="robust",
    vce="unadjusted",
    params="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    inst_1="income ssiratio medicaid aget aget2 educyr actlim totchr _cons",
    params_1="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    sexp_1="(docvis / exp( ({xb_private} *private + {xb_medicaid} *medicaid + {xb_aget} *aget + {xb_aget2} *aget2 + {xb_educyr} *educyr + {xb_actlim} *actlim + {xb_totchr} *totchr) + {b0} )) - 1",  # noqa:E501
    properties="b V",
)

params_table = np.array([
    .67815486150911,  .25018082946574,  2.7106587781218,  .00671496899138,
    .1878094461339,  1.1685002768843, np.nan,  1.9599639845401,
    0,  .28872920226215,  .10311429027815,  2.8000891193967,
    .00510884999633,  .08662890702558,  .49082949749873, np.nan,
    1.9599639845401,                0,  .27067161407481,   .0518802415232,
    5.2172388972735,  1.816099638e-07,  .16898820918009,  .37235501896953,
    np.nan,  1.9599639845401,                0, -.05690878166227,
    .0118728670827, -4.7931793783164,  1.641587211e-06, -.08017917353758,
    -.03363838978695, np.nan,  1.9599639845401,                0,
    .01438116368432,  .00781887593806,  1.8392878718448,   .0658728559523,
    -.00094355155385,   .0297058789225, np.nan,  1.9599639845401,
    0,  .18038238197017,  .03819661477822,  4.7224703816696,
    2.329970297e-06,  .10551839267351,  .25524637126682, np.nan,
    1.9599639845401,                0,  .28251055147828,  .01376659609161,
    20.521452768591,  1.385109204e-93,  .25552851894901,  .30949258400755,
    np.nan,  1.9599639845401,                0,  .50771182444237,
    .10208891085993,  4.9732318639284,  6.584582712e-07,  .30762123593598,
    .70780241294876, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['xb_private', 'xb_medicaid', 'xb_aget', 'xb_aget2',
                         'xb_educyr', 'xb_actlim', 'xb_totchr', 'b0']

cov = np.array([
    .06259044743217,  .02308524749042,  .00531802921719,  -.0010223122446,
    -.00154027662468,  .00154945994717, -.00043816683551, -.01554486097815,
    .02308524749042,  .01063255685957,  .00205438168765, -.00039193802388,
    -.00049039628782,   .0002760841411,  -.0002064504141, -.00691934867666,
    .00531802921719,  .00205438168765,  .00269155946051, -.00059250696972,
    -.00012247118567,  .00021403084056, -.00004749600121, -.00308951213731,
    -.0010223122446, -.00039193802388, -.00059250696972,  .00014096497276,
    .00002468288871, -.00006115240604,  .00001190303672,  .00058327928125,
    -.00154027662468, -.00049039628782, -.00012247118567,  .00002468288871,
    .00006113482093, -.00001854325518,  .00001439868646,  .00007784185009,
    .00154945994717,   .0002760841411,  .00021403084056, -.00006115240604,
    -.00001854325518,  .00145898138052, -.00016596475072, -.00074697007542,
    -.00043816683551,  -.0002064504141, -.00004749600121,  .00001190303672,
    .00001439868646, -.00016596475072,  .00018951916795, -.00027350320218,
    -.01554486097815, -.00691934867666, -.00308951213731,  .00058327928125,
    .00007784185009, -.00074697007542, -.00027350320218,  .01042214572057
    ]).reshape(8, 8)

cov_colnames = ['xb_private', 'xb_medicaid', 'xb_aget', 'xb_aget2',
                'xb_educyr', 'xb_actlim', 'xb_totchr', 'b0']

cov_rownames = ['xb_private', 'xb_medicaid', 'xb_aget', 'xb_aget2',
                'xb_educyr', 'xb_actlim', 'xb_totchr', 'b0']


results_multtwostepdefault = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)


est = dict(
    rank=8,
    N=3629,
    Q=.0002590497181628,
    J=.940091427212973,
    J_df=1,
    k_1=8,
    converged=1,
    has_xtinst=0,
    type=1,
    n_eq=1,
    k=8,
    n_moments=9,
    k_aux=8,
    k_eq_model=0,
    k_eq=8,
    cmdline="gmm ( (docvis / exp({xb:private medicaid aget aget2 educyr actlim totchr}+{b0})) - 1), instruments(income ssiratio medicaid aget aget2 educyr actlim totchr) twostep wmatrix(robust) center",  # noqa:E501
    cmd="gmm",
    estat_cmd="gmm_estat",
    predict="gmm_p",
    marginsnotok="_ALL",
    eqnames="1",
    technique="gn",
    winit="Unadjusted",
    estimator="twostep",
    wmatrix="robust",
    vce="robust",
    vcetype="Robust",
    params="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    inst_1="income ssiratio medicaid aget aget2 educyr actlim totchr _cons",
    params_1="xb_private xb_medicaid xb_aget xb_aget2 xb_educyr xb_actlim xb_totchr b0",  # noqa:E501
    sexp_1="(docvis / exp( ({xb_private} *private + {xb_medicaid} *medicaid + {xb_aget} *aget + {xb_aget2} *aget2 + {xb_educyr} *educyr + {xb_actlim} *actlim + {xb_totchr} *totchr) + {b0} )) - 1",  # noqa:E501
    properties="b V",
)

params_table = np.array([
    .67815486150911,  .25053960844836,  2.7067770469869,  .00679398676131,
    .18710625224955,  1.1692034707687, np.nan,  1.9599639845401,
    0,  .28872920226215,  .10327332768441,  2.7957770775479,
    .00517750993835,  .08631719943712,  .49114120508719, np.nan,
    1.9599639845401,                0,  .27067161407481,  .05199697557915,
    5.2055261110869,  1.934477426e-07,  .16875941463467,  .37258381351495,
    np.nan,  1.9599639845401,                0, -.05690878166227,
    .01189862079945, -4.7828048831437,  1.728659059e-06, -.08022964989488,
    -.03358791342965, np.nan,  1.9599639845401,                0,
    .01438116368432,  .00783219272776,  1.8361605982125,  .06633390816397,
    -.00096965198207,  .02973197935072, np.nan,  1.9599639845401,
    0,  .18038238197017,  .03826654814775,    4.71383991244,
    2.430916736e-06,  .10538132578791,  .25538343815243, np.nan,
    1.9599639845401,                0,  .28251055147828,  .01378476509846,
    20.494404471929,  2.415234157e-93,  .25549290834996,   .3095281946066,
    np.nan,  1.9599639845401,                0,  .50771182444237,
    .10235828870929,   4.960143734762,  7.044103886e-07,    .307093265053,
    .70833038383174, np.nan,  1.9599639845401,                0
    ]).reshape(8, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = ['_cons'] * 8

cov = np.array([
    .06277009540146,  .02315708886727,  .00533574465012,  -.0010254503134,
    -.00154463481696,  .00155084007911, -.00043796389511, -.01559997980204,
    .02315708886727,  .01066538021101,  .00206217721135, -.00039331175814,
    -.00049172883672,  .00027603038575, -.00020644729789, -.00694809209467,
    .00533574465012,  .00206217721135,  .00270368546938, -.00059509464294,
    -.000122765895,  .00021462183651, -.00004681968717,   -.003107676362,
    -.0010254503134, -.00039331175814, -.00059509464294,  .00014157717693,
    .00002474211983, -.00006134664668,  .00001178278294,  .00058658166731,
    -.00154463481696, -.00049172883672,   -.000122765895,  .00002474211983,
    .00006134324292, -.00001855938213,  .00001443468876,  .00007766055925,
    .00155084007911,  .00027603038575,  .00021462183651, -.00006134664668,
    -.00001855938213,  .00146432870714, -.00016643336248, -.00074847778305,
    -.00043796389511, -.00020644729789, -.00004681968717,  .00001178278294,
    .00001443468876, -.00016643336248,  .00019001974882, -.00027573582025,
    -.01559997980204, -.00694809209467,   -.003107676362,  .00058658166731,
    .00007766055925, -.00074847778305, -.00027573582025,   .0104772192675
    ]).reshape(8, 8)

cov_colnames = ['_cons'] * 8

cov_rownames = ['_cons'] * 8


results_multtwostepcenter = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)
