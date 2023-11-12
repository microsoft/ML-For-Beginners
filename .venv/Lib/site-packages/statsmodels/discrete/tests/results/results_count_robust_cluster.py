import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
           rank=3,
           N=34,
           ic=1,
           k=3,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           N_clust=5,
           ll=-354.2436413025559,
           k_eq_model=1,
           ll_0=-356.2029100704882,
           df_m=2,
           chi2=5.204189583786304,
           p=.0741181533729996,
           r2_p=.0055004288638308,
           cmdline="poisson accident yr_con op_75_79, vce(cluster ship)",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vcetype="Robust",
           clustvar="ship",
           vce="cluster",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log pseudolikelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    -.02172061893549,  .19933709357097, -.10896426022065,  .91323083771076,
    -.41241414311748,  .36897290524649, np.nan,  1.9599639845401,
    0,  .22148585072024,  .11093628220713,  1.9965140918162,
    .04587799343723,  .00405473301549,  .43891696842499, np.nan,
    1.9599639845401,                0,  2.2697077143215,  1.1048569901548,
    2.054299999499,  .03994666479943,  .10422780555076,  4.4351876230922,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons'.split()

cov = np.array([
    .03973527687332,  .00976206273414, -.21171095768584,  .00976206273414,
    .01230685870994, -.06297293767114, -.21171095768584, -.06297293767114,
    1.2207089686939]).reshape(3, 3)

cov_colnames = 'yr_con op_75_79 _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons'.split()


results_poisson_clu = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=3,
           N=34,
           ic=1,
           k=3,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-354.2436413025559,
           k_eq_model=1,
           ll_0=-356.2029100704882,
           df_m=2,
           chi2=.1635672212515404,
           p=.9214713337295277,
           r2_p=.0055004288638308,
           cmdline="poisson accident yr_con op_75_79, vce(robust)",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
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
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    -.02172061893549,  .19233713248134, -.11292993014545,  .91008610728406,
    -.39869447148862,  .35525323361764, np.nan,  1.9599639845401,
    0,  .22148585072024,  .55301404772037,    .400506735106,
    .68878332380143,  -.8624017657564,  1.3053734671969, np.nan,
    1.9599639845401,                0,  2.2697077143215,  .66532523368388,
    3.4114258702533,  .00064624070669,  .96569421829539,  3.5737212103476,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons'.split()

cov = np.array([
    .03699357253114, -.01521223175214, -.09585501859714, -.01521223175214,
    .30582453697607,  -.1649339692102, -.09585501859714,  -.1649339692102,
    .44265766657651]).reshape(3, 3)

cov_colnames = 'yr_con op_75_79 _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons'.split()


results_poisson_hc1 = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=3,
           N=34,
           ic=4,
           k=3,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-91.28727940081573,
           k_eq_model=1,
           ll_0=-122.0974139280415,
           df_m=2,
           chi2=61.62026905445154,
           p=4.16225408420e-14,
           r2_p=.2523405986746273,
           cmdline="poisson accident yr_con op_75_79, exposure(service)",
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(service)",
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
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    .30633819450439,  .05790831365493,   5.290055523458,  1.222792336e-07,
    .19283998533528,   .4198364036735, np.nan,  1.9599639845401,
    0,  .35592229608495,  .12151759298719,  2.9289775030556,
    .00340079035234,  .11775219034206,  .59409240182785, np.nan,
    1.9599639845401,                0,  -6.974712802772,  .13252425018256,
    -52.629709605328,                0,  -7.234455560208,  -6.714970045336,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons'.split()

cov = np.array([
    .00335337279036, -.00315267340017, -.00589654294427, -.00315267340017,
    .0147665254054, -.00165060980569, -.00589654294427, -.00165060980569,
    .01756267688645]).reshape(3, 3)

cov_colnames = 'yr_con op_75_79 _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons'.split()


results_poisson_exposure_nonrobust = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=3,
           N=34,
           ic=4,
           k=3,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-91.28727940081573,
           k_eq_model=1,
           ll_0=-122.0974139280415,
           df_m=2,
           chi2=15.1822804640621,
           p=.0005049050167458,
           r2_p=.2523405986746273,
           cmdline="poisson accident yr_con op_75_79, exposure(service) vce(robust)",  # noqa:E501
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(service)",
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
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    .30633819450439,  .09144457613957,  3.3499875819514,  .00080815183366,
    .12711011868929,  .48556627031949, np.nan,  1.9599639845401,
    0,  .35592229608495,  .16103531267836,  2.2102127177276,
    .02709040275274,  .04029888299621,  .67154570917369, np.nan,
    1.9599639845401,                0,  -6.974712802772,   .2558675415017,
    -27.259076168227,  1.29723387e-163, -7.4762039689282, -6.4732216366159,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons'.split()

cov = np.array([
    .00836211050535,  .00098797681063, -.01860743122756,  .00098797681063,
    .02593237192942, -.02395236210603, -.01860743122756, -.02395236210603,
    .06546819879413]).reshape(3, 3)

cov_colnames = 'yr_con op_75_79 _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons'.split()


results_poisson_exposure_hc1 = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=3,
           N=34,
           ic=4,
           k=3,
           k_eq=1,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           N_clust=5,
           ll=-91.28727940081573,
           k_eq_model=1,
           ll_0=-122.0974139280415,
           df_m=2,
           chi2=340.7343047354823,
           p=1.02443835269e-74,
           r2_p=.2523405986746273,
           cmdline="poisson accident yr_con op_75_79, exposure(service) vce(cluster ship)",  # noqa:E501
           cmd="poisson",
           predict="poisso_p",
           estat_cmd="poisson_estat",
           offset="ln(service)",
           gof="poiss_g",
           chi2type="Wald",
           opt="moptimize",
           vcetype="Robust",
           clustvar="ship",
           vce="cluster",
           title="Poisson regression",
           user="poiss_lf",
           crittype="log pseudolikelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    .30633819450439,  .03817694295902,  8.0241677504982,  1.022165435e-15,
    .23151276126487,  .38116362774391, np.nan,  1.9599639845401,
    0,  .35592229608495,  .09213163536669,  3.8631930787765,
    .00011191448109,  .17534760892947,  .53649698324044, np.nan,
    1.9599639845401,                0,  -6.974712802772,   .0968656626603,
    -72.003975518463,                0, -7.1645660129248, -6.7848595926192,
    np.nan,  1.9599639845401,                0]).reshape(3, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons'.split()

cov = np.array([
    .0014574789737, -.00277745275086,  .00108765624666, -.00277745275086,
    .00848823823534, -.00469929607507,  .00108765624666, -.00469929607507,
    .00938295660262]).reshape(3, 3)

cov_colnames = 'yr_con op_75_79 _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons'.split()


results_poisson_exposure_clu = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=4,
           N=34,
           ic=2,
           k=4,
           k_eq=2,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           N_clust=5,
           ll=-109.0877965183258,
           k_eq_model=1,
           ll_0=-109.1684720604314,
           rank0=2,
           df_m=2,
           chi2=5.472439553195301,
           p=.0648148991694882,
           k_aux=1,
           alpha=2.330298308905143,
           cmdline="nbreg accident yr_con op_75_79, vce(cluster ship)",
           cmd="nbreg",
           predict="nbreg_p",
           dispers="mean",
           diparm_opt2="noprob",
           chi2type="Wald",
           opt="moptimize",
           vcetype="Robust",
           clustvar="ship",
           vce="cluster",
           title="Negative binomial regression",
           diparm1="lnalpha, exp label(",
           user="nbreg_lf",
           crittype="log pseudolikelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    -.03536709401845,  .27216090050938, -.12994921001605,  .89660661037787,
    -.56879265701682,  .49805846897992, np.nan,  1.9599639845401,
    0,  .23211570238882,  .09972456245386,  2.3275680201277,
    .01993505322091,  .03665915160525,  .42757225317239, np.nan,
    1.9599639845401,                0,  2.2952623989519,  1.2335785495143,
    1.8606536242509,  .06279310688494, -.12250713019722,  4.7130319281011,
    np.nan,  1.9599639845401,                0,  .84599628895555,
    .22483100011931, np.nan, np.nan,  .40533562611357,
    1.2866569517975, np.nan,  1.9599639845401,                0,
    2.3302983089051,  .52392329936749, np.nan, np.nan,
    1.4998057895818,  3.6206622525444, np.nan,  1.9599639845401,
    0]).reshape(5, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons _cons alpha'.split()

cov = np.array([
    .07407155576607, -.00421355148283, -.32663130963457,  .02015715724983,
    -.00421355148283,  .00994498835661,  .00992613461881, -.00714955450361,
    -.32663130963457,  .00992613461881,  1.5217160378218, -.09288283512096,
    .02015715724983, -.00714955450361, -.09288283512096,  .05054897861465
    ]).reshape(4, 4)

cov_colnames = 'yr_con op_75_79 _cons _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons _cons'.split()


results_negbin_clu = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=4,
           N=34,
           ic=2,
           k=4,
           k_eq=2,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-109.0877965183258,
           k_eq_model=1,
           ll_0=-109.1684720604314,
           rank0=2,
           df_m=2,
           chi2=.1711221347493475,
           p=.9179970816706797,
           r2_p=.0007390003778831,
           k_aux=1,
           alpha=2.330298308905143,
           cmdline="nbreg accident yr_con op_75_79, vce(robust)",
           cmd="nbreg",
           predict="nbreg_p",
           dispers="mean",
           diparm_opt2="noprob",
           chi2type="Wald",
           opt="moptimize",
           vcetype="Robust",
           vce="robust",
           title="Negative binomial regression",
           diparm1="lnalpha, exp label(",
           user="nbreg_lf",
           crittype="log pseudolikelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    -.03536709401845,  .26106873337039, -.13547043172065,  .89223994079058,
    -.5470524089139,    .476318220877, np.nan,  1.9599639845401,
    0,  .23211570238882,  .56245325203342,  .41268443475019,
    .67983783029986, -.87027241458412,  1.3345038193618, np.nan,
    1.9599639845401,                0,  2.2952623989519,  .76040210713867,
    3.0184850586341,  .00254041928465,  .80490165519179,  3.7856231427121,
    np.nan,  1.9599639845401,                0,  .84599628895555,
    .24005700345444, np.nan, np.nan,  .37549320794823,
    1.3164993699629, np.nan,  1.9599639845401,                0,
    2.3302983089051,  .55940442919073, np.nan, np.nan,
    1.4557092049439,  3.7303399539165, np.nan,  1.9599639845401,
    0]).reshape(5, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons _cons alpha'.split()

cov = np.array([
    .06815688354362, -.03840590969835, -.16217402790798,  .02098165591138,
    -.03840590969835,  .31635366072297, -.11049674936104, -.02643483668568,
    -.16217402790798, -.11049674936104,  .57821136454093, -.03915049342584,
    .02098165591138, -.02643483668568, -.03915049342584,  .05762736490753
    ]).reshape(4, 4)

cov_colnames = 'yr_con op_75_79 _cons _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons _cons'.split()


results_negbin_hc1 = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=4,
           N=34,
           ic=4,
           k=4,
           k_eq=2,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           ll=-82.49115612464289,
           k_eq_model=1,
           ll_0=-84.68893065247886,
           rank0=2,
           df_m=2,
           chi2=4.39554905567195,
           p=.1110500222994781,
           ll_c=-91.28727940081573,
           chi2_c=17.5922465523457,
           r2_p=.0259511427397111,
           k_aux=1,
           alpha=.2457422083490335,
           cmdline="nbreg accident yr_con op_75_79, exposure(service)",
           cmd="nbreg",
           predict="nbreg_p",
           offset="ln(service)",
           dispers="mean",
           diparm_opt2="noprob",
           chi2_ct="LR",
           chi2type="LR",
           opt="moptimize",
           vce="oim",
           title="Negative binomial regression",
           diparm1="lnalpha, exp label(",
           user="nbreg_lf",
           crittype="log likelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    .28503762550355,  .14983643534827,  1.9023251910727,  .05712865433138,
    -.00863639135093,  .57871164235802, np.nan,  1.9599639845401,
    0,  .17127003537767,  .27580549562862,  .62098122804736,
    .53461197443513, -.36929880279264,  .71183887354798, np.nan,
    1.9599639845401,                0, -6.5908639033905,  .40391814231008,
    -16.31732574748,  7.432080344e-60, -7.3825289150206, -5.7991988917604,
    np.nan,  1.9599639845401,                0, -1.4034722260565,
    .51305874839271, np.nan, np.nan, -2.4090488948595,
    -.39789555725363, np.nan,  1.9599639845401,                0,
    .24574220834903,  .12608018984282, np.nan, np.nan,
    .089900758997,  .67173218155228, np.nan,  1.9599639845401,
    0]).reshape(5, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons _cons alpha'.split()

cov = np.array([
    .02245095735788, -.01097939549632, -.05127649084781,  .00045725833006,
    -.01097939549632,  .07606867141895,  -.0197375670989, -.00926008351523,
    -.05127649084781,  -.0197375670989,  .16314986568722,  .02198323898312,
    .00045725833006, -.00926008351523,  .02198323898312,  .26322927930229
    ]).reshape(4, 4)

cov_colnames = 'yr_con op_75_79 _cons _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons _cons'.split()


results_negbin_exposure_nonrobust = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )

est = dict(
           rank=4,
           N=34,
           ic=4,
           k=4,
           k_eq=2,
           k_dv=1,
           converged=1,
           rc=0,
           k_autoCns=0,
           N_clust=5,
           ll=-82.49115612464289,
           k_eq_model=1,
           ll_0=-84.68893065247886,
           rank0=2,
           df_m=2,
           chi2=5.473741859983782,
           p=.0647727084656973,
           k_aux=1,
           alpha=.2457422083490335,
           cmdline="nbreg accident yr_con op_75_79, exposure(service) vce(cluster ship)",  # noqa:E501
           cmd="nbreg",
           predict="nbreg_p",
           offset="ln(service)",
           dispers="mean",
           diparm_opt2="noprob",
           chi2type="Wald",
           opt="moptimize",
           vcetype="Robust",
           clustvar="ship",
           vce="cluster",
           title="Negative binomial regression",
           diparm1="lnalpha, exp label(",
           user="nbreg_lf",
           crittype="log pseudolikelihood",
           ml_method="e2",
           singularHmethod="m-marquardt",
           technique="nr",
           which="max",
           depvar="accident",
           properties="b V",
          )

params_table = np.array([
    .28503762550355,  .14270989695062,  1.9973220610073,  .04579020833966,
    .00533136724292,  .56474388376418, np.nan,  1.9599639845401,
    0,  .17127003537767,  .17997186802799,  .95164892854829,
    .34127505843023, -.18146834418759,  .52400841494293, np.nan,
    1.9599639845401,                0, -6.5908639033905,  .62542746996715,
    -10.538174640357,  5.760612980e-26, -7.8166792194681, -5.3650485873129,
    np.nan,  1.9599639845401,                0, -1.4034722260565,
    .86579403765571, np.nan, np.nan, -3.1003973578913,
    .29345290577817, np.nan,  1.9599639845401,                0,
    .24574220834903,  .21276213878894, np.nan, np.nan,
    .0450313052935,  1.3410500222158, np.nan,  1.9599639845401,
    0]).reshape(5, 9)

params_table_colnames = 'b se z pvalue ll ul df crit eform'.split()

params_table_rownames = 'yr_con op_75_79 _cons _cons alpha'.split()

cov = np.array([
    .02036611468766, -.00330004038514, -.08114367170947, -.07133030733881,
    -.00330004038514,  .03238987328148, -.03020509748676, -.09492663454187,
    -.08114367170947, -.03020509748676,  .39115952018952,  .43276143586693,
    -.07133030733881, -.09492663454187,  .43276143586693,  .74959931564018
    ]).reshape(4, 4)

cov_colnames = 'yr_con op_75_79 _cons _cons'.split()

cov_rownames = 'yr_con op_75_79 _cons _cons'.split()


results_negbin_exposure_clu = ParamsTableTestBunch(
                params_table=params_table,
                params_table_colnames=params_table_colnames,
                params_table_rownames=params_table_rownames,
                cov=cov,
                cov_colnames=cov_colnames,
                cov_rownames=cov_rownames,
                **est
                )
