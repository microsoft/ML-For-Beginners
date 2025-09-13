<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T11:51:52+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "hr"
}
-->
# Prognoza vremenskih serija s ARIMA

U prethodnoj lekciji naučili ste nešto o prognozi vremenskih serija i učitali skup podataka koji prikazuje fluktuacije električnog opterećenja tijekom određenog vremenskog razdoblja.

[![Uvod u ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Uvod u ARIMA")

> 🎥 Kliknite na sliku iznad za video: Kratak uvod u ARIMA modele. Primjer je napravljen u R-u, ali koncepti su univerzalni.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

U ovoj lekciji otkrit ćete specifičan način izrade modela pomoću [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA modeli posebno su prikladni za podatke koji pokazuju [nestacionarnost](https://wikipedia.org/wiki/Stationary_process).

## Opći koncepti

Da biste mogli raditi s ARIMA modelima, potrebno je razumjeti nekoliko ključnih pojmova:

- 🎓 **Stacionarnost**. U statističkom kontekstu, stacionarnost se odnosi na podatke čija distribucija ne mijenja kada se pomakne u vremenu. Nestacionarni podaci pokazuju fluktuacije zbog trendova koje je potrebno transformirati kako bi se analizirali. Sezonalnost, na primjer, može uzrokovati fluktuacije u podacima i može se eliminirati procesom 'sezonskog diferenciranja'.

- 🎓 **[Diferenciranje](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Diferenciranje podataka, opet u statističkom kontekstu, odnosi se na proces transformacije nestacionarnih podataka kako bi postali stacionarni uklanjanjem njihovog nekonstantnog trenda. "Diferenciranje uklanja promjene u razini vremenske serije, eliminirajući trend i sezonalnost te stabilizirajući srednju vrijednost vremenske serije." [Rad Shixionga i sur.](https://arxiv.org/abs/1904.07632)

## ARIMA u kontekstu vremenskih serija

Razložimo dijelove ARIMA modela kako bismo bolje razumjeli kako pomaže u modeliranju vremenskih serija i omogućuje prognoze.

- **AR - AutoRegresivno**. Autoregresivni modeli, kako ime sugerira, gledaju 'unatrag' u vremenu kako bi analizirali prethodne vrijednosti u vašim podacima i napravili pretpostavke o njima. Te prethodne vrijednosti nazivaju se 'zaostaci'. Primjer bi bili podaci koji prikazuju mjesečnu prodaju olovaka. Ukupna prodaja svakog mjeseca smatra se 'promjenjivom varijablom' u skupu podataka. Ovaj model se gradi tako da se "promjenjiva varijabla od interesa regresira na svoje vlastite zaostale (tj. prethodne) vrijednosti." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrirano**. Za razliku od sličnih 'ARMA' modela, 'I' u ARIMA odnosi se na njegov *[integrirani](https://wikipedia.org/wiki/Order_of_integration)* aspekt. Podaci se 'integriraju' kada se primjenjuju koraci diferenciranja kako bi se eliminirala nestacionarnost.

- **MA - Pomični prosjek**. [Pomični prosjek](https://wikipedia.org/wiki/Moving-average_model) u ovom modelu odnosi se na izlaznu varijablu koja se određuje promatranjem trenutnih i prošlih vrijednosti zaostataka.

Zaključak: ARIMA se koristi za izradu modela koji što bolje odgovara specifičnom obliku podataka vremenskih serija.

## Vježba - izrada ARIMA modela

Otvorite [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) mapu u ovoj lekciji i pronađite datoteku [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Pokrenite notebook kako biste učitali Python biblioteku `statsmodels`; trebat će vam za ARIMA modele.

1. Učitajte potrebne biblioteke.

1. Sada učitajte još nekoliko biblioteka korisnih za vizualizaciju podataka:

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

1. Učitajte podatke iz datoteke `/data/energy.csv` u Pandas dataframe i pogledajte ih:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Prikaz svih dostupnih podataka o energiji od siječnja 2012. do prosinca 2014. Ne bi trebalo biti iznenađenja jer smo te podatke vidjeli u prethodnoj lekciji:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Sada, izradimo model!

### Izrada skupa za treniranje i testiranje

Sada kada su podaci učitani, možete ih podijeliti na skup za treniranje i skup za testiranje. Model ćete trenirati na skupu za treniranje. Kao i obično, nakon što model završi treniranje, procijenit ćete njegovu točnost pomoću skupa za testiranje. Morate osigurati da skup za testiranje pokriva kasnije vremensko razdoblje od skupa za treniranje kako biste osigurali da model ne dobije informacije iz budućih vremenskih razdoblja.

1. Dodijelite dvomjesečno razdoblje od 1. rujna do 31. listopada 2014. skupu za treniranje. Skup za testiranje uključivat će dvomjesečno razdoblje od 1. studenog do 31. prosinca 2014.:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Budući da ovi podaci odražavaju dnevnu potrošnju energije, postoji snažan sezonski obrazac, ali potrošnja je najsličnija potrošnji u nedavnim danima.

1. Vizualizirajte razlike:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![podaci za treniranje i testiranje](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Stoga bi korištenje relativno malog vremenskog okvira za treniranje podataka trebalo biti dovoljno.

    > Napomena: Budući da funkcija koju koristimo za prilagodbu ARIMA modela koristi validaciju unutar skupa tijekom prilagodbe, izostavit ćemo podatke za validaciju.

### Priprema podataka za treniranje

Sada trebate pripremiti podatke za treniranje filtriranjem i skaliranjem podataka. Filtrirajte svoj skup podataka tako da uključuje samo potrebna vremenska razdoblja i stupce, te skalirajte podatke kako bi bili u intervalu 0,1.

1. Filtrirajte originalni skup podataka tako da uključuje samo prethodno navedena vremenska razdoblja po skupu i samo potrebni stupac 'load' plus datum:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Možete vidjeti oblik podataka:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skalirajte podatke tako da budu u rasponu (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Vizualizirajte originalne i skalirane podatke:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![originalni](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Originalni podaci

    ![skalirani](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Skalirani podaci

1. Sada kada ste kalibrirali skalirane podatke, možete skalirati podatke za testiranje:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementacija ARIMA

Vrijeme je za implementaciju ARIMA modela! Sada ćete koristiti biblioteku `statsmodels` koju ste ranije instalirali.

Sada trebate slijediti nekoliko koraka:

   1. Definirajte model pozivanjem `SARIMAX()` i prosljeđivanjem parametara modela: p, d i q parametara, te P, D i Q parametara.
   2. Pripremite model za podatke za treniranje pozivanjem funkcije `fit()`.
   3. Napravite prognoze pozivanjem funkcije `forecast()` i određivanjem broja koraka (horizonta) za prognozu.

> 🎓 Što znače svi ti parametri? U ARIMA modelu postoje 3 parametra koji se koriste za modeliranje glavnih aspekata vremenske serije: sezonalnost, trend i šum. Ti parametri su:

`p`: parametar povezan s autoregresivnim aspektom modela, koji uključuje *prošle* vrijednosti.
`d`: parametar povezan s integriranim dijelom modela, koji utječe na količinu *diferenciranja* (🎓 sjetite se diferenciranja 👆?) primijenjenog na vremensku seriju.
`q`: parametar povezan s dijelom modela koji se odnosi na pomični prosjek.

> Napomena: Ako vaši podaci imaju sezonski aspekt - što ovi podaci imaju - koristimo sezonski ARIMA model (SARIMA). U tom slučaju trebate koristiti drugi skup parametara: `P`, `D` i `Q` koji opisuju iste asocijacije kao `p`, `d` i `q`, ali odgovaraju sezonskim komponentama modela.

1. Započnite postavljanjem željene vrijednosti horizonta. Pokušajmo s 3 sata:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Odabir najboljih vrijednosti za parametre ARIMA modela može biti izazovan jer je donekle subjektivan i vremenski zahtjevan. Možete razmotriti korištenje funkcije `auto_arima()` iz biblioteke [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Za sada pokušajte s ručnim odabirom kako biste pronašli dobar model.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Ispisuje se tablica rezultata.

Napravili ste svoj prvi model! Sada trebamo pronaći način za njegovu evaluaciju.

### Evaluacija modela

Za evaluaciju modela možete koristiti tehniku nazvanu `walk forward` validacija. U praksi se modeli vremenskih serija ponovno treniraju svaki put kada postanu dostupni novi podaci. To omogućuje modelu da napravi najbolju prognozu u svakom vremenskom koraku.

Počevši od početka vremenske serije, koristeći ovu tehniku, trenirajte model na skupu podataka za treniranje. Zatim napravite prognozu za sljedeći vremenski korak. Prognoza se procjenjuje u odnosu na poznatu vrijednost. Skup za treniranje se zatim proširuje kako bi uključio poznatu vrijednost i proces se ponavlja.

> Napomena: Trebali biste zadržati fiksni prozor skupa za treniranje radi učinkovitijeg treniranja, tako da svaki put kada dodate novu promatranje u skup za treniranje, uklonite promatranje s početka skupa.

Ovaj proces pruža robusniju procjenu kako će model funkcionirati u praksi. Međutim, dolazi s računalnim troškom stvaranja toliko modela. To je prihvatljivo ako su podaci mali ili ako je model jednostavan, ali može biti problem na većim razmjerima.

Walk-forward validacija je zlatni standard za evaluaciju modela vremenskih serija i preporučuje se za vaše projekte.

1. Prvo, stvorite testnu točku podataka za svaki korak HORIZON-a.

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

    Podaci se horizontalno pomiču prema točki horizonta.

1. Napravite prognoze na testnim podacima koristeći ovaj pristup kliznog prozora u petlji veličine duljine testnih podataka:

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

    Možete pratiti proces treniranja:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Usporedite prognoze s stvarnim opterećenjem:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Izlaz
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Promatrajte prognozu podataka po satu u usporedbi sa stvarnim opterećenjem. Koliko je točno?

### Provjera točnosti modela

Provjerite točnost svog modela testiranjem njegove srednje apsolutne postotne pogreške (MAPE) na svim prognozama.
> **🧮 Pokaži mi matematiku**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) koristi se za prikaz točnosti predviđanja kao omjera definiranog gornjom formulom. Razlika između stvarne i predviđene vrijednosti dijeli se sa stvarnom vrijednošću. "Apsolutna vrijednost u ovom izračunu zbraja se za svaku predviđenu točku u vremenu i dijeli s brojem prilagođenih točaka n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Izrazite jednadžbu u kodu:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Izračunajte MAPE za jedan korak:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE prognoze za jedan korak:  0.5570581332313952 %

1. Ispišite MAPE za višekoraksku prognozu:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Niska vrijednost je najbolja: uzmite u obzir da prognoza s MAPE od 10 odstupa za 10%.

1. No, kao i uvijek, lakše je vizualno vidjeti ovakvo mjerenje točnosti, pa hajdemo to prikazati:

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

    ![model vremenskih serija](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Vrlo lijep graf, koji prikazuje model s dobrom točnošću. Bravo!

---

## 🚀Izazov

Istražite načine testiranja točnosti modela vremenskih serija. U ovoj lekciji spominjemo MAPE, ali postoje li drugi načini koje biste mogli koristiti? Istražite ih i zabilježite. Koristan dokument možete pronaći [ovdje](https://otexts.com/fpp2/accuracy.html)

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Ova lekcija pokriva samo osnove prognoziranja vremenskih serija pomoću ARIMA. Odvojite vrijeme za produbljivanje svog znanja istražujući [ovaj repozitorij](https://microsoft.github.io/forecasting/) i njegove različite vrste modela kako biste naučili druge načine izrade modela vremenskih serija.

## Zadatak

[Novi ARIMA model](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane ljudskog prevoditelja. Ne preuzimamo odgovornost za nesporazume ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.