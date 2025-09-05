<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T11:53:25+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "sl"
}
-->
# Napovedovanje časovnih vrst z ARIMA

V prejšnji lekciji ste se naučili nekaj o napovedovanju časovnih vrst in naložili podatkovni niz, ki prikazuje nihanja električne obremenitve skozi časovno obdobje.

🎥 Kliknite zgornjo sliko za video: Kratek uvod v modele ARIMA. Primer je narejen v R, vendar so koncepti univerzalni.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

V tej lekciji boste spoznali specifičen način za gradnjo modelov z [ARIMA: *A*uto*R*egresivno *I*ntegrirano *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA modeli so še posebej primerni za podatke, ki kažejo [nestacionarnost](https://wikipedia.org/wiki/Stationary_process).

## Splošni koncepti

Da bi lahko delali z ARIMA, morate poznati nekaj osnovnih konceptov:

- 🎓 **Stacionarnost**. V statističnem kontekstu stacionarnost pomeni podatke, katerih porazdelitev se ne spreminja, ko jih premaknemo v času. Nestacionarni podatki kažejo nihanja zaradi trendov, ki jih je treba transformirati za analizo. Sezonskost, na primer, lahko povzroči nihanja v podatkih, ki jih je mogoče odpraviti s procesom 'sezonskega razlikovanja'.

- 🎓 **[Razlikovanje](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Razlikovanje podatkov, ponovno v statističnem kontekstu, se nanaša na proces transformacije nestacionarnih podatkov v stacionarne z odstranitvijo njihovega nespremenljivega trenda. "Razlikovanje odstrani spremembe v ravni časovne vrste, odpravi trend in sezonskost ter posledično stabilizira povprečje časovne vrste." [Študija Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA v kontekstu časovnih vrst

Razčlenimo dele ARIMA, da bolje razumemo, kako nam pomaga modelirati časovne vrste in napovedovati podatke.

- **AR - za AutoRegresivno**. Avtoregresivni modeli, kot že ime pove, gledajo 'nazaj' v času, da analizirajo prejšnje vrednosti v vaših podatkih in naredijo predpostavke o njih. Te prejšnje vrednosti se imenujejo 'zaostanki'. Primer bi bili podatki, ki prikazujejo mesečno prodajo svinčnikov. Skupna prodaja vsakega meseca bi se štela kot 'spremenljivka v razvoju' v podatkovnem nizu. Ta model je zgrajen tako, da se "spremenljivka v razvoju regresira na svoje zaostale (tj. prejšnje) vrednosti." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - za Integrirano**. V nasprotju s podobnimi modeli 'ARMA' se 'I' v ARIMA nanaša na njegovo *[integrirano](https://wikipedia.org/wiki/Order_of_integration)* lastnost. Podatki so 'integrirani', ko se uporabijo koraki razlikovanja za odpravo nestacionarnosti.

- **MA - za Premično Povprečje**. Vidik [premičnega povprečja](https://wikipedia.org/wiki/Moving-average_model) v tem modelu se nanaša na izhodno spremenljivko, ki je določena z opazovanjem trenutnih in preteklih vrednosti zaostankov.

Zaključek: ARIMA se uporablja za prilagoditev modela posebni obliki podatkov časovnih vrst čim bolj natančno.

## Naloga - zgradite ARIMA model

Odprite mapo [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) v tej lekciji in poiščite datoteko [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Zaženite beležko, da naložite knjižnico Python `statsmodels`; to boste potrebovali za modele ARIMA.

1. Naložite potrebne knjižnice.

1. Nato naložite še nekaj knjižnic, ki so uporabne za vizualizacijo podatkov:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. Naložite podatke iz datoteke `/data/energy.csv` v Pandas dataframe in si jih oglejte:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Prikažite vse razpoložljive podatke o energiji od januarja 2012 do decembra 2014. Presenečenj ne bi smelo biti, saj smo te podatke videli v prejšnji lekciji:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Zdaj pa zgradimo model!

### Ustvarite učne in testne podatkovne nize

Zdaj so vaši podatki naloženi, zato jih lahko ločite na učni in testni niz. Model boste trenirali na učnem nizu. Kot običajno, ko bo model končal učenje, boste ocenili njegovo natančnost z uporabo testnega niza. Poskrbeti morate, da testni niz zajema kasnejše časovno obdobje od učnega niza, da zagotovite, da model ne pridobi informacij iz prihodnjih časovnih obdobij.

1. Dodelite dvomesečno obdobje od 1. septembra do 31. oktobra 2014 učnemu nizu. Testni niz bo vključeval dvomesečno obdobje od 1. novembra do 31. decembra 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Ker ti podatki odražajo dnevno porabo energije, obstaja močan sezonski vzorec, vendar je poraba najbolj podobna porabi v bolj nedavnih dneh.

1. Vizualizirajte razlike:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![učni in testni podatki](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Zato bi moralo biti dovolj, da za učenje podatkov uporabimo relativno majhno časovno okno.

    > Opomba: Ker funkcija, ki jo uporabljamo za prilagoditev modela ARIMA, uporablja validacijo znotraj vzorca med prilagajanjem, bomo izpustili validacijske podatke.

### Pripravite podatke za učenje

Zdaj morate pripraviti podatke za učenje z izvajanjem filtriranja in skaliranja podatkov. Filtrirajte svoj podatkovni niz, da vključite le potrebna časovna obdobja in stolpce, ter skalirajte podatke, da zagotovite, da so prikazani v intervalu 0,1.

1. Filtrirajte izvirni podatkovni niz, da vključite le prej omenjena časovna obdobja za vsak niz in le potrebni stolpec 'load' ter datum:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Lahko vidite obliko podatkov:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skalirajte podatke, da bodo v razponu (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Vizualizirajte izvirne in skalirane podatke:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![izvirni](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Izvirni podatki

    ![skalirani](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Skalirani podatki

1. Zdaj, ko ste kalibrirali skalirane podatke, lahko skalirate testne podatke:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementirajte ARIMA

Čas je, da implementirate ARIMA! Zdaj boste uporabili knjižnico `statsmodels`, ki ste jo prej namestili.

Zdaj morate slediti več korakom:

   1. Določite model z uporabo `SARIMAX()` in podajte parametre modela: parametre p, d in q ter parametre P, D in Q.
   2. Pripravite model za učne podatke z uporabo funkcije fit().
   3. Naredite napovedi z uporabo funkcije `forecast()` in določite število korakov (obzorje), ki jih želite napovedati.

> 🎓 Kaj pomenijo vsi ti parametri? V modelu ARIMA obstajajo 3 parametri, ki pomagajo modelirati glavne vidike časovne vrste: sezonskost, trend in šum. Ti parametri so:

`p`: parameter, povezan z avtoregresivnim vidikom modela, ki vključuje *pretekle* vrednosti.
`d`: parameter, povezan z integriranim delom modela, ki vpliva na količino *razlikovanja* (🎓 spomnite se razlikovanja 👆?), ki se uporabi na časovni vrsti.
`q`: parameter, povezan z delom modela, ki se nanaša na premično povprečje.

> Opomba: Če vaši podatki vsebujejo sezonski vidik - kar ti podatki vsebujejo -, uporabimo sezonski ARIMA model (SARIMA). V tem primeru morate uporabiti še en niz parametrov: `P`, `D` in `Q`, ki opisujejo enake povezave kot `p`, `d` in `q`, vendar ustrezajo sezonskim komponentam modela.

1. Začnite z nastavitvijo želene vrednosti obzorja. Poskusimo s 3 urami:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Izbira najboljših vrednosti za parametre modela ARIMA je lahko zahtevna, saj je nekoliko subjektivna in časovno intenzivna. Morda boste želeli uporabiti funkcijo `auto_arima()` iz knjižnice [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Za zdaj poskusite nekaj ročnih izbir, da najdete dober model.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Tabela rezultatov je natisnjena.

Zgradili ste svoj prvi model! Zdaj moramo najti način za njegovo oceno.

### Ocenite svoj model

Za oceno svojega modela lahko izvedete tako imenovano validacijo `walk forward`. V praksi se modeli časovnih vrst ponovno trenirajo vsakič, ko so na voljo novi podatki. To omogoča modelu, da naredi najboljšo napoved na vsakem časovnem koraku.

Začnite na začetku časovne vrste z uporabo te tehnike, trenirajte model na učnem podatkovnem nizu. Nato naredite napoved za naslednji časovni korak. Napoved se oceni glede na znano vrednost. Učni niz se nato razširi, da vključuje znano vrednost, in proces se ponovi.

> Opomba: Učno okno podatkovnega niza bi morali ohraniti fiksno za bolj učinkovito učenje, tako da vsakič, ko dodate novo opazovanje v učni niz, odstranite opazovanje z začetka niza.

Ta proces zagotavlja bolj robustno oceno, kako bo model deloval v praksi. Vendar pa to prinaša računske stroške ustvarjanja toliko modelov. To je sprejemljivo, če so podatki majhni ali če je model preprost, vendar bi lahko bil problem pri večjih podatkih.

Validacija z metodo walk-forward je zlati standard za ocenjevanje modelov časovnih vrst in jo priporočamo za vaše projekte.

1. Najprej ustvarite testno podatkovno točko za vsak korak obzorja.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    Podatki se horizontalno premaknejo glede na točko obzorja.

1. Naredite napovedi za svoje testne podatke z uporabo tega drsnega okna v zanki dolžine testnih podatkov:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    Lahko opazujete potek učenja:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Primerjajte napovedi z dejansko obremenitvijo:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Izhod
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Opazujte napovedi urnih podatkov v primerjavi z dejansko obremenitvijo. Kako natančen je model?

### Preverite natančnost modela

Preverite natančnost svojega modela z oceno njegove povprečne absolutne odstotne napake (MAPE) za vse napovedi.
> **🧮 Poglejmo matematiko**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) se uporablja za prikaz natančnosti napovedi kot razmerje, opredeljeno z zgornjo formulo. Razlika med dejansko vrednostjo in napovedano vrednostjo je deljena z dejansko vrednostjo. "Absolutna vrednost v tem izračunu se sešteje za vsako napovedano točko v času in deli s številom prilagojenih točk n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Izrazite enačbo v kodi:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Izračunajte MAPE za en korak:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE napovedi za en korak:  0.5570581332313952 %

1. Izpišite MAPE za več korakov:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Nizka vrednost je najboljša: upoštevajte, da je napoved z MAPE 10 odstopanje za 10 %.

1. Kot vedno je tovrstno merjenje natančnosti lažje razumeti vizualno, zato ga prikažimo na grafu:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![model časovne vrste](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Zelo lep graf, ki prikazuje model z dobro natančnostjo. Odlično opravljeno!

---

## 🚀Izziv

Raziskujte načine za testiranje natančnosti modela časovnih vrst. V tej lekciji smo se dotaknili MAPE, vendar obstajajo tudi druge metode, ki jih lahko uporabite. Raziskujte jih in jih označite. Koristen dokument najdete [tukaj](https://otexts.com/fpp2/accuracy.html).

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Ta lekcija se dotika le osnov napovedovanja časovnih vrst z ARIMA. Vzemite si čas za poglobitev znanja z raziskovanjem [tega repozitorija](https://microsoft.github.io/forecasting/) in njegovih različnih tipov modelov, da se naučite drugih načinov za gradnjo modelov časovnih vrst.

## Naloga

[Nov ARIMA model](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da se zavedate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.