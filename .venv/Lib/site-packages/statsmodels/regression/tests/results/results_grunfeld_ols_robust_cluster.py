import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch


est = dict(
    N_clust=10,
    N=200,
    df_m=2,
    df_r=9,
    F=51.59060716590177,
    r2=.8124080178314147,
    rmse=94.40840193979599,
    mss=7604093.484267689,
    rss=1755850.432294737,
    r2_a=.8105035307027997,
    ll=-1191.80235741801,
    ll_0=-1359.150955647688,
    rank=3,
    cmdline="regress invest mvalue kstock, vce(cluster company)",
    title="Linear regression",
    marginsok="XB default",
    vce="cluster",
    depvar="invest",
    cmd="regress",
    properties="b V",
    predict="regres_p",
    model="ols",
    estat_cmd="regress_estat",
    vcetype="Robust",
    clustvar="company",
    )

params_table = np.array([
    .11556215606596,  .01589433647768,  7.2706499090564,  .00004710548549,
    .07960666895505,  .15151764317688,                9,  2.2621571627982,
    0,  .23067848754982,  .08496711097464,  2.7149150406994,
    .02380515903536,  .03846952885627,  .42288744624337,                9,
    2.2621571627982,                0, -42.714369016733,  20.425202580078,
    -2.0912580352272,  .06604843284516, -88.919387334862,  3.4906493013959,
    9,  2.2621571627982,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00025262993207, -.00065043385106,  .20961897960949, -.00065043385106,
    .00721940994738, -1.2171040967615,  .20961897960949, -1.2171040967615,
    417.18890043724]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_cluster = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
    )

est = dict(
    N=200,
    inexog_ct=2,
    exexog_ct=0,
    endog_ct=0,
    partial_ct=0,
    N_clust=10,
    df_m=2,
    sdofminus=0,
    dofminus=0,
    r2=.8124080178314146,
    rmse=93.69766358599176,
    rss=1755850.432294737,
    mss=7604093.484267682,
    r2_a=.8105035307027995,
    F=51.59060716590192,
    Fp=.0000117341240941,
    Fdf1=2,
    Fdf2=9,
    yy=13620706.07273678,
    yyc=9359943.916562419,
    partialcons=0,
    cons=1,
    jdf=0,
    j=0,
    ll=-1191.802357418011,
    rankV=3,
    rankS=3,
    rankxx=3,
    rankzz=3,
    r2c=.8124080178314146,
    r2u=.8710896173136538,
    clustvar="company",
    hacsubtitleV="Statistics robust to heteroskedasticity and clustering on company",  # noqa:E501
    hacsubtitleB="Estimates efficient for homoskedasticity only",
    title="OLS estimation",
    predict="ivreg2_p",
    version="03.1.07",
    cmdline="ivreg2 invest mvalue kstock, cluster(company)",
    cmd="ivreg2",
    model="ols",
    depvar="invest",
    vcetype="Robust",
    vce="robust cluster",
    partialsmall="small",
    inexog="mvalue kstock",
    insts="mvalue kstock",
    properties="b V",
)

params_table = np.array([
    .11556215606596,  .01500272788516,  7.7027429245215,  1.331761148e-14,
    .08615734974119,  .14496696239074, np.nan,  1.9599639845401,
    0,  .23067848754982,  .08020079648691,  2.8762618035529,
    .00402415789383,  .07348781490405,  .38786916019559, np.nan,
    1.9599639845401,                0, -42.714369016733,   19.27943055305,
    -2.2155410088072,  .02672295281194, -80.501358543152, -4.9273794903145,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .000225081844, -.00057950714469,   .1867610305767, -.00057950714469,
    .00643216775713, -1.0843847053056,   .1867610305767, -1.0843847053056,
    371.69644244987]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_cluster_large = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N=200,
    N_g=10,
    df_m=2,
    df_r=9,
    F=97.97910905239282,
    r2=.8124080178314147,
    rmse=94.40840193979599,
    lag=4,
    cmd="xtscc",
    predict="xtscc_p",
    method="Pooled OLS",
    depvar="invest",
    vcetype="Drisc/Kraay",
    title="Regression with Driscoll-Kraay standard errors",
    groupvar="company",
    properties="b V",
    )

params_table = np.array([
    .11556215606596,   .0134360177573,  8.6009231420662,  .00001235433261,
    .08516777225681,  .14595653987512,                9,  2.2621571627982,
    0,  .23067848754982,  .04930800664089,   4.678317037431,
    .00115494570515,  .11913602714384,   .3422209479558,                9,
    2.2621571627982,                0, -42.714369016733,  12.190347184209,
    -3.5039501641153,   .0066818746948, -70.290850216489, -15.137887816977,
    9,  2.2621571627982,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00018052657317, -.00035661054613, -.06728261073866, -.00035661054613,
    .0024312795189, -.32394785247278, -.06728261073866, -.32394785247278,
    148.60456447156]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_nw_groupsum4 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    df_m=2,
    df_r=197,
    F=73.07593045506036,
    N=200,
    lag=4,
    rank=3,
    title="Regression with Newey-West standard errors",
    cmd="newey",
    cmdline="newey invest mvalue kstock, lag(4) force",
    estat_cmd="newey_estat",
    predict="newey_p",
    vcetype="Newey-West",
    depvar="invest",
    properties="b V",
)

params_table = np.array([
    .11556215606596,  .01142785251475,  10.112324771147,  1.251631065e-19,
    .0930255277205,  .13809878441142,              197,  1.9720790337785,
    0,  .23067848754982,  .06842168281423,  3.3714237660029,
    .00089998163666,  .09574552141602,  .36561145368361,              197,
    1.9720790337785,                0, -42.714369016733,  16.179042041128,
    -2.6401049523298,  .00895205094219, -74.620718612662, -10.808019420804,
    197,  1.9720790337785,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .0001305958131, -.00022910455176,  .00889686530849, -.00022910455176,
    .00468152667913, -.88403667445531,  .00889686530849, -.88403667445531,
    261.76140136858]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_nw_panel4 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N=200,
    inexog_ct=2,
    exexog_ct=0,
    endog_ct=0,
    partial_ct=0,
    df_r=9,
    N_clust=10,
    N_clust1=10,
    N_clust2=20,
    df_m=2,
    sdofminus=0,
    dofminus=0,
    r2=.8124080178314146,
    rmse=94.40840193979601,
    rss=1755850.432294737,
    mss=7604093.484267682,
    r2_a=.8105035307027995,
    F=57.99124535923564,
    Fp=7.21555935862e-06,
    Fdf1=2,
    partialcons=0,
    cons=1,
    jdf=0,
    j=0,
    ll=-1191.802357418011,
    rankV=3,
    rankS=3,
    rankxx=3,
    rankzz=3,
    r2c=.8124080178314146,
    r2u=.8710896173136538,
    yyc=9359943.916562419,
    yy=13620706.07273678,
    Fdf2=9,
    clustvar="company time",
    hacsubtitleV="Statistics robust to heteroskedasticity and clustering on company and time",  # noqa:E501
    hacsubtitleB="Estimates efficient for homoskedasticity only",
    title="OLS estimation",
    predict="ivreg2_p",
    version="03.1.07",
    cmdline="ivreg2 invest mvalue kstock, cluster(company time) small",
    cmd="ivreg2",
    model="ols",
    depvar="invest",
    vcetype="Robust",
    clustvar2="time",
    clustvar1="company",
    vce="robust two-way cluster",
    partialsmall="small",
    small="small",
    inexog="mvalue kstock",
    insts="mvalue kstock",
    properties="b V",
)

params_table = np.array([
    .11556215606596,  .01635175387097,  7.0672636695645,  .00005873628221,
    .07857191892244,  .15255239320949,                9,  2.2621571627982,
    0,  .23067848754982,  .07847391274682,  2.9395563375824,
    .01649863150032,  .05315816373679,  .40819881136285,                9,
    2.2621571627982,                0, -42.714369016733,  19.505607409785,
    -2.189850750062,  .05626393734425, -86.839118533508,  1.4103805000422,
    9,  2.2621571627982,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00026737985466, -.00070163493529,  .19641438763743, -.00070163493529,
    .0061581549818, -.99627581152391,  .19641438763743, -.99627581152391,
    380.46872042467]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_cluster_2groups_small = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N=200,
    inexog_ct=2,
    exexog_ct=0,
    endog_ct=0,
    partial_ct=0,
    N_clust=10,
    N_clust1=10,
    N_clust2=20,
    df_m=2,
    sdofminus=0,
    dofminus=0,
    r2=.8124080178314146,
    rmse=93.69766358599176,
    rss=1755850.432294737,
    mss=7604093.484267682,
    r2_a=.8105035307027995,
    F=57.99124535923565,
    Fp=7.21555935862e-06,
    Fdf1=2,
    Fdf2=9,
    partialcons=0,
    cons=1,
    jdf=0,
    j=0,
    ll=-1191.802357418011,
    rankV=3,
    rankS=3,
    rankxx=3,
    rankzz=3,
    r2c=.8124080178314146,
    r2u=.8710896173136538,
    yyc=9359943.916562419,
    yy=13620706.07273678,
    clustvar="company time",
    hacsubtitleV="Statistics robust to heteroskedasticity and clustering on company and time",  # noqa:E501
    hacsubtitleB="Estimates efficient for homoskedasticity only",
    title="OLS estimation",
    predict="ivreg2_p",
    version="03.1.07",
    cmdline="ivreg2 invest mvalue kstock, cluster(company time)",
    cmd="ivreg2",
    model="ols",
    depvar="invest",
    vcetype="Robust",
    clustvar2="time",
    clustvar1="company",
    vce="robust two-way cluster",
    partialsmall="small",
    inexog="mvalue kstock",
    insts="mvalue kstock",
    properties="b V",
)

params_table = np.array([
    .11556215606596,  .01543448599542,   7.487269488613,  7.032121917e-14,
    .08531111939505,  .14581319273688, np.nan,  1.9599639845401,
    0,  .23067848754982,  .07407184066336,  3.1142534799181,
    .00184410987255,  .08550034758104,   .3758566275186, np.nan,
    1.9599639845401,                0, -42.714369016733,  18.411420987265,
    -2.319993065515,  .02034125246974, -78.800091055978, -6.6286469774879,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00023822335794, -.00062512499511,  .17499633632219, -.00062512499511,
    .00548663757926, -.88763669036779,  .17499633632219, -.88763669036779,
    338.98042277032]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_cluster_2groups_large = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N=200,
    inexog_ct=2,
    exexog_ct=0,
    endog_ct=0,
    partial_ct=0,
    bw=5,
    N_clust=20,
    df_m=2,
    sdofminus=0,
    dofminus=0,
    r2=.8124080178314146,
    rmse=93.69766358599176,
    rss=1755850.432294737,
    mss=7604093.484267682,
    r2_a=.8105035307027995,
    F=92.14467466912147,
    Fp=1.66368179227e-10,
    Fdf1=2,
    Fdf2=19,
    yy=13620706.07273678,
    partialcons=0,
    cons=1,
    jdf=0,
    j=0,
    ll=-1191.802357418011,
    rankV=3,
    rankS=3,
    rankxx=3,
    rankzz=3,
    r2c=.8124080178314146,
    r2u=.8710896173136538,
    yyc=9359943.916562419,
    clustvar="year",
    hacsubtitleV2="and kernel-robust to common correlated disturbances (Driscoll-Kraay)",  # noqa:E501
    hacsubtitleV="Statistics robust to heteroskedasticity and clustering on year",  # noqa:E501
    hacsubtitleB="Estimates efficient for homoskedasticity only",
    title="OLS estimation",
    predict="ivreg2_p",
    version="03.1.07",
    cmdline="ivreg2 invest mvalue kstock, dkraay(5)",
    cmd="ivreg2",
    model="ols",
    depvar="invest",
    vcetype="Robust",
    vce="cluster ac bartlett bw=5",
    partialsmall="small",
    ivar="company",
    tvar="year",
    kernel="Bartlett",
    inexog="mvalue kstock",
    insts="mvalue kstock",
    properties="b V",
)

params_table = np.array([
    .11556215606596,   .0134360177573,  8.6009231420662,  7.907743030e-18,
    .08922804516602,  .14189626696591, np.nan,  1.9599639845401,
    0,  .23067848754982,  .04930800664089,   4.678317037431,
    2.892390940e-06,  .13403657038422,  .32732040471542, np.nan,
    1.9599639845401,                0, -42.714369016733,  12.190347184209,
    -3.5039501641153,  .00045841113727, -66.607010456823, -18.821727576643,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00018052657317, -.00035661054613, -.06728261073866, -.00035661054613,
    .0024312795189, -.32394785247278, -.06728261073866, -.32394785247278,
    148.60456447156]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_nw_groupsum4_ivreg_large = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N=200,
    inexog_ct=2,
    exexog_ct=0,
    endog_ct=0,
    partial_ct=0,
    bw=5,
    df_r=19,
    N_clust=20,
    df_m=2,
    sdofminus=0,
    dofminus=0,
    r2=.8124080178314146,
    rmse=94.40840193979601,
    rss=1755850.432294737,
    mss=7604093.484267682,
    r2_a=.8105035307027995,
    F=92.14467466912149,
    Fp=1.66368179227e-10,
    Fdf1=2,
    Fdf2=19,
    partialcons=0,
    cons=1,
    jdf=0,
    j=0,
    ll=-1191.802357418011,
    rankV=3,
    rankS=3,
    rankxx=3,
    rankzz=3,
    r2c=.8124080178314146,
    r2u=.8710896173136538,
    yyc=9359943.916562419,
    yy=13620706.07273678,
    clustvar="year",
    hacsubtitleV2="and kernel-robust to common correlated disturbances (Driscoll-Kraay)",  # noqa:E501
    hacsubtitleV="Statistics robust to heteroskedasticity and clustering on year",  # noqa:E501
    hacsubtitleB="Estimates efficient for homoskedasticity only",
    title="OLS estimation",
    predict="ivreg2_p",
    version="03.1.07",
    cmdline="ivreg2 invest mvalue kstock, dkraay(5) small",
    cmd="ivreg2",
    model="ols",
    depvar="invest",
    vcetype="Robust",
    vce="cluster ac bartlett bw=5",
    partialsmall="small",
    small="small",
    ivar="company",
    tvar="year",
    kernel="Bartlett",
    inexog="mvalue kstock",
    insts="mvalue kstock",
    properties="b V",
)

params_table = np.array([
    .11556215606596,   .0138548615926,  8.3409101775303,  8.967911239e-08,
    .08656359748216,  .14456071464977,               19,  2.0930240544083,
    0,  .23067848754982,   .0508450956047,  4.5368876743442,
    .00022550505646,  .12425847940049,  .33709849569915,               19,
    2.0930240544083,                0, -42.714369016733,  12.570359466158,
    -3.3980228752988,  .00301793225123,  -69.02443375196, -16.404304281506,
    19,  2.0930240544083,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00019195718975, -.00037919048186, -.07154282413568, -.00037919048186,
    .00258522374705, -.34445964542925, -.07154282413568, -.34445964542925,
    158.01393710842]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_nw_groupsum4_ivreg_small = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)


# ----------------------------------------------------------------
# WLS

est = dict(
    N=200,
    df_m=2,
    df_r=197,
    F=158.2726503915062,
    r2=.7728224625923459,
    rmse=35.1783035325949,
    mss=829335.6968772264,
    rss=243790.0687679817,
    r2_a=.7705160916541971,
    ll=-994.3622459900876,
    ll_0=-1142.564592396746,
    rank=3,
    cmdline="regress invest mvalue kstock [aw=1/mvalue], robust",
    title="Linear regression",
    marginsok="XB default",
    vce="robust",
    depvar="invest",
    cmd="regress",
    properties="b V",
    predict="regres_p",
    model="ols",
    estat_cmd="regress_estat",
    wexp="= 1/mvalue",
    wtype="aweight",
    vcetype="Robust",
)

params_table = np.array([
    .11694307068216,  .00768545583365,    15.2161528494,  4.371656843e-35,
    .10178674436759,  .13209939699674,              197,  1.9720790337785,
    0,  .10410756769914,  .00986959606725,  10.548310892334,
    6.565731752e-21,  .08464394422305,  .12357119117523,              197,
    1.9720790337785,                0, -9.2723336171089,  2.3458404391932,
    -3.9526702081656,  .00010767530575, -13.898516363832, -4.6461508703863,
    197,  1.9720790337785,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00005906623137,  6.805470065e-06, -.01210153268743,  6.805470065e-06,
    .00009740892653, -.01511046663892, -.01210153268743, -.01511046663892,
    5.502967366154]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_hc1_wls_small = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N_clust=10,
    N=200,
    df_m=2,
    df_r=9,
    F=22.90591346432732,
    r2=.7728224625923459,
    rmse=35.1783035325949,
    mss=829335.6968772264,
    rss=243790.0687679817,
    r2_a=.7705160916541971,
    ll=-994.3622459900876,
    ll_0=-1142.564592396746,
    rank=3,
    cmdline="regress invest mvalue kstock[aw=1/mvalue], vce(cluster company)",
    title="Linear regression",
    marginsok="XB default",
    vce="cluster",
    depvar="invest",
    cmd="regress",
    properties="b V",
    predict="regres_p",
    model="ols",
    estat_cmd="regress_estat",
    wexp="= 1/mvalue",
    wtype="aweight",
    vcetype="Robust",
    clustvar="company",
)

params_table = np.array([
    .11694307068216,  .02609630113434,  4.4812124936848,  .00152974827456,
    .05790913614858,  .17597700521575,                9,  2.2621571627982,
    0,  .10410756769914,  .02285882773869,  4.5543703679489,
    .00137730504553,  .05239730679689,  .15581782860139,                9,
    2.2621571627982,                0, -9.2723336171089,  5.7204731422962,
    -1.6209032690934,  .13948922172294, -22.212942910549,  3.6682756763312,
    9,  2.2621571627982,                0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00068101693289, -.00006496077364, -.08926939086077, -.00006496077364,
    .00052252600559,  -.0697116307149, -.08926939086077,  -.0697116307149,
    32.723812971732]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_cluster_wls_small = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)

est = dict(
    N=200,
    inexog_ct=2,
    exexog_ct=0,
    endog_ct=0,
    partial_ct=0,
    N_clust=10,
    df_m=2,
    sdofminus=0,
    dofminus=0,
    r2=.772822462592346,
    rmse=34.91346937558495,
    rss=243790.0687679817,
    mss=829335.6968772268,
    r2_a=.7705160916541972,
    F=22.9059134643273,
    Fp=.000294548654088,
    Fdf1=2,
    Fdf2=9,
    yy=1401938.856802022,
    yyc=1073125.765645209,
    partialcons=0,
    cons=1,
    jdf=0,
    j=0,
    ll=-994.3622459900874,
    rankV=3,
    rankS=3,
    rankxx=3,
    rankzz=3,
    r2c=.772822462592346,
    r2u=.8261050632949187,
    clustvar="company",
    hacsubtitleV="Statistics robust to heteroskedasticity and clustering on company",  # noqa:E501
    hacsubtitleB="Estimates efficient for homoskedasticity only",
    title="OLS estimation",
    predict="ivreg2_p",
    version="03.1.07",
    cmdline="ivreg2 invest mvalue kstock [aw=1/mvalue], cluster(company)",
    cmd="ivreg2",
    wtype="aweight",
    wexp="=1/mvalue",
    model="ols",
    depvar="invest",
    vcetype="Robust",
    vce="robust cluster",
    partialsmall="small",
    inexog="mvalue kstock",
    insts="mvalue kstock",
    properties="b V",
    )

params_table = np.array([
    .11694307068216,  .02463240320082,  4.7475298990826,  2.059159576e-06,
    .06866444755588,  .16522169380844, np.nan,  1.9599639845401,
    0,  .10410756769914,  .02157653909108,  4.8250355286218,
    1.399783125e-06,  .06181832816961,  .14639680722867, np.nan,
    1.9599639845401,                0, -9.2723336171089,  5.3995775192484,
    -1.7172331694572,  .08593657730569, -19.855311086568,    1.31064385235,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'mvalue kstock _cons'.split()

cov = np.array([
    .00060675528745, -.00005787711139, -.07953498994782, -.00005787711139,
    .00046554703915, -.06210991017966, -.07953498994782, -.06210991017966,
    29.155437386372]).reshape(3, 3)

cov_colnames = 'mvalue kstock _cons'.split()

cov_rownames = 'mvalue kstock _cons'.split()


results_cluster_wls_large = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est
)
