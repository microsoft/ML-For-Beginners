<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T23:47:46+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "fi"
}
-->
# Aikasarjojen ennustaminen ARIMA-mallilla

Edellisessä oppitunnissa opit hieman aikasarjojen ennustamisesta ja latasit tietoaineiston, joka näyttää sähkönkulutuksen vaihtelut tietyn ajanjakson aikana.

[![Johdatus ARIMA-malliin](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Johdatus ARIMA-malliin")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi videon: Lyhyt johdatus ARIMA-malleihin. Esimerkki on tehty R-kielellä, mutta käsitteet ovat yleispäteviä.

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

## Johdanto

Tässä oppitunnissa tutustut erityiseen tapaan rakentaa malleja [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) avulla. ARIMA-mallit soveltuvat erityisesti datan analysointiin, joka osoittaa [ei-stationaarisuutta](https://wikipedia.org/wiki/Stationary_process).

## Yleiset käsitteet

Jotta voit työskennellä ARIMA-mallien kanssa, sinun täytyy tuntea muutamia käsitteitä:

- 🎓 **Stationaarisuus**. Tilastollisessa kontekstissa stationaarisuus viittaa dataan, jonka jakauma ei muutu ajan siirtyessä. Ei-stationaarinen data puolestaan osoittaa vaihteluita trendien vuoksi, jotka täytyy muuntaa analysoitavaksi. Esimerkiksi kausiluonteisuus voi aiheuttaa vaihteluita datassa, ja se voidaan poistaa 'kausittaisen differoinnin' avulla.

- 🎓 **[Differointi](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Differointi tarkoittaa ei-stationaarisen datan muuntamista stationaariseksi poistamalla sen ei-vakioinen trendi. "Differointi poistaa aikasarjan tason muutokset, eliminoi trendin ja kausiluonteisuuden ja vakauttaa siten aikasarjan keskiarvon." [Shixiong et al -tutkimus](https://arxiv.org/abs/1904.07632)

## ARIMA aikasarjojen kontekstissa

Puretaan ARIMA:n osat, jotta ymmärrämme paremmin, miten se auttaa mallintamaan aikasarjoja ja tekemään ennusteita niiden perusteella.

- **AR - AutoRegressiivinen**. Autoregressiiviset mallit, kuten nimi viittaa, katsovat 'taaksepäin' ajassa analysoidakseen datan aiempia arvoja ja tehdäkseen oletuksia niiden perusteella. Näitä aiempia arvoja kutsutaan 'viiveiksi'. Esimerkkinä voisi olla data, joka näyttää kuukausittaiset lyijykynien myyntiluvut. Jokaisen kuukauden myyntilukuja pidettäisiin datasetin 'muuttuvana muuttujana'. Tämä malli rakennetaan siten, että "kiinnostava muuttuva muuttuja regressoidaan omiin viivästettyihin (eli aiempiin) arvoihinsa." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integroitu**. Toisin kuin samankaltaiset 'ARMA'-mallit, ARIMA:n 'I' viittaa sen *[integroituun](https://wikipedia.org/wiki/Order_of_integration)* osaan. Data integroidaan, kun differointivaiheita sovelletaan ei-stationaarisuuden poistamiseksi.

- **MA - Liukuva keskiarvo**. Mallin [liukuva keskiarvo](https://wikipedia.org/wiki/Moving-average_model) viittaa ulostulomuuttujaan, joka määritetään tarkkailemalla nykyisiä ja aiempia viiveiden arvoja.

Yhteenveto: ARIMA:a käytetään mallin sovittamiseen erityiseen aikasarjadatan muotoon mahdollisimman tarkasti.

## Harjoitus - rakenna ARIMA-malli

Avaa tämän oppitunnin [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) -kansio ja etsi [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) -tiedosto.

1. Suorita notebook ladataksesi `statsmodels`-kirjaston Pythonissa; tarvitset tätä ARIMA-malleja varten.

1. Lataa tarvittavat kirjastot.

1. Lataa nyt lisää kirjastoja, jotka ovat hyödyllisiä datan visualisointiin:

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

1. Lataa data `/data/energy.csv` -tiedostosta Pandas-dataframeen ja tarkastele sitä:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Piirrä kaikki saatavilla oleva energiadata tammikuusta 2012 joulukuuhun 2014. Ei pitäisi olla yllätyksiä, sillä näimme tämän datan viime oppitunnissa:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Nyt rakennetaan malli!

### Luo harjoitus- ja testidatasetit

Kun datasi on ladattu, voit jakaa sen harjoitus- ja testidatasetiksi. Harjoitat mallisi harjoitusdatalla. Kuten tavallista, kun malli on valmis, arvioit sen tarkkuutta testidatan avulla. Sinun täytyy varmistaa, että testidata kattaa myöhemmän ajanjakson kuin harjoitusdata, jotta malli ei saa tietoa tulevista ajanjaksoista.

1. Määritä kahden kuukauden ajanjakso syyskuun 1. päivästä lokakuun 31. päivään 2014 harjoitusdataksi. Testidata sisältää kahden kuukauden ajanjakson marraskuun 1. päivästä joulukuun 31. päivään 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Koska tämä data heijastaa päivittäistä energiankulutusta, siinä on vahva kausiluonteinen kuvio, mutta kulutus on eniten samankaltaista lähimpien päivien kulutuksen kanssa.

1. Visualisoi erot:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![harjoitus- ja testidata](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Siksi suhteellisen pienen ajanjakson käyttäminen datan harjoittamiseen pitäisi olla riittävä.

    > Huom: Koska käyttämämme funktio ARIMA-mallin sovittamiseen käyttää sisäistä validointia sovituksen aikana, jätämme validointidatan pois.

### Valmistele data harjoitusta varten

Nyt sinun täytyy valmistella data harjoitusta varten suodattamalla ja skaalaamalla dataasi. Suodata datasetti sisältämään vain tarvittavat ajanjaksot ja sarakkeet, ja skaalaa data varmistaaksesi, että se on välillä 0,1.

1. Suodata alkuperäinen datasetti sisältämään vain edellä mainitut ajanjaksot per setti ja vain tarvittava sarake 'load' sekä päivämäärä:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Voit tarkastella datan muotoa:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skaalaa data välille (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualisoi alkuperäinen vs. skaalattu data:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![alkuperäinen](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Alkuperäinen data

    ![skaalattu](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Skaalattu data

1. Nyt kun olet kalibroinut skaalatun datan, voit skaalata testidatan:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Toteuta ARIMA

On aika toteuttaa ARIMA! Käytät nyt aiemmin asennettua `statsmodels`-kirjastoa.

Seuraa nyt useita vaiheita:

   1. Määritä malli kutsumalla `SARIMAX()` ja välittämällä mallin parametrit: p-, d- ja q-parametrit sekä P-, D- ja Q-parametrit.
   2. Valmistele malli harjoitusdataa varten kutsumalla fit()-funktio.
   3. Tee ennusteita kutsumalla `forecast()`-funktio ja määrittämällä ennustettavien askelten määrä (ennustehorisontti).

> 🎓 Mitä nämä parametrit tarkoittavat? ARIMA-mallissa on kolme parametria, joita käytetään mallintamaan aikasarjan keskeisiä piirteitä: kausiluonteisuus, trendi ja kohina. Nämä parametrit ovat:

`p`: parametri, joka liittyy mallin autoregressiiviseen osaan ja sisältää *menneet* arvot.
`d`: parametri, joka liittyy mallin integroituun osaan ja vaikuttaa siihen, kuinka paljon *differointia* (🎓 muista differointi 👆?) sovelletaan aikasarjaan.
`q`: parametri, joka liittyy mallin liukuvaan keskiarvoon.

> Huom: Jos datassasi on kausiluonteinen piirre - kuten tässä datassa on -, käytämme kausiluonteista ARIMA-mallia (SARIMA). Tässä tapauksessa sinun täytyy käyttää toista joukkoa parametreja: `P`, `D` ja `Q`, jotka kuvaavat samoja yhteyksiä kuin `p`, `d` ja `q`, mutta vastaavat mallin kausiluonteisia komponentteja.

1. Aloita asettamalla haluamasi horisonttiarvo. Kokeillaan 3 tuntia:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMA-mallin parametrien parhaiden arvojen valitseminen voi olla haastavaa, sillä se on osittain subjektiivista ja aikaa vievää. Voit harkita `auto_arima()`-funktion käyttöä [`pyramid`-kirjastosta](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Kokeile nyt joitakin manuaalisia valintoja löytääksesi hyvän mallin.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Tulostetaan tulostaulukko.

Olet rakentanut ensimmäisen mallisi! Nyt meidän täytyy löytää tapa arvioida sitä.

### Arvioi mallisi

Mallin arvioimiseksi voit käyttää niin sanottua `walk forward` -validointia. Käytännössä aikasarjamallit koulutetaan uudelleen aina, kun uutta dataa tulee saataville. Tämä mahdollistaa parhaan ennusteen tekemisen jokaisessa ajankohdassa.

Aloittaen aikasarjan alusta tällä tekniikalla, kouluta malli harjoitusdatalla. Tee sitten ennuste seuraavasta ajankohdasta. Ennustetta verrataan tunnettuun arvoon. Harjoitusdataa laajennetaan sisältämään tunnettu arvo, ja prosessi toistetaan.

> Huom: Pidä harjoitusdatan ikkuna kiinteänä tehokkaamman koulutuksen vuoksi, jotta joka kerta kun lisäät uuden havainnon harjoitusdataan, poistat havainnon datan alusta.

Tämä prosessi tarjoaa vankemman arvion siitä, miten malli toimii käytännössä. Se kuitenkin lisää laskentakustannuksia, koska niin monta mallia täytyy luoda. Tämä on hyväksyttävää, jos data on pieni tai malli on yksinkertainen, mutta voi olla ongelmallista suuremmassa mittakaavassa.

Walk-forward-validointi on aikasarjamallien arvioinnin kultainen standardi ja suositeltavaa omissa projekteissasi.

1. Luo ensin testidatapiste jokaiselle HORIZON-askeleelle.

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

    Data siirtyy horisontaalisesti horisonttipisteensä mukaan.

1. Tee ennusteita testidatallasi käyttämällä tätä liukuvaa ikkunamenetelmää silmukassa, joka on testidatan pituuden kokoinen:

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

    Voit seurata koulutuksen etenemistä:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Vertaa ennusteita todelliseen kuormitukseen:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Tuloste
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Tarkastele tuntikohtaisen datan ennustetta verrattuna todelliseen kuormitukseen. Kuinka tarkka tämä on?

### Tarkista mallin tarkkuus

Tarkista mallisi tarkkuus testaamalla sen keskimääräinen absoluuttinen prosenttivirhe (MAPE) kaikissa ennusteissa.
> **🧮 Näytä matematiikka**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) käytetään ennustetarkkuuden osoittamiseen suhteena, joka määritellään yllä olevan kaavan mukaan. Todellisen ja ennustetun ero jaetaan todellisella.
>
> "Tämän laskelman absoluuttinen arvo summataan jokaiselle ennustetulle ajankohdalle ja jaetaan sovitettujen pisteiden lukumäärällä n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Ilmaise yhtälö koodissa:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Laske yhden askeleen MAPE:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    Yhden askeleen ennusteen MAPE:  0.5570581332313952 %

1. Tulosta moniaskeleen ennusteen MAPE:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Pieni luku on paras: huomaa, että ennuste, jonka MAPE on 10, poikkeaa 10 %.

1. Mutta kuten aina, tämänkaltaisen tarkkuuden mittaaminen on helpompaa visuaalisesti, joten piirretään se:

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

    ![aikajaksomalli](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Erittäin hieno kuvaaja, joka näyttää mallin hyvällä tarkkuudella. Hyvää työtä!

---

## 🚀Haaste

Tutki tapoja testata aikajaksomallin tarkkuutta. Tässä oppitunnissa käsitellään MAPE:a, mutta onko olemassa muita menetelmiä, joita voisit käyttää? Tutki niitä ja tee muistiinpanoja. Hyödyllinen dokumentti löytyy [täältä](https://otexts.com/fpp2/accuracy.html).

## [Oppitunnin jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itseopiskelu

Tämä oppitunti käsittelee vain aikajaksomallinnuksen perusteita ARIMA:lla. Käytä aikaa syventääksesi tietämystäsi tutkimalla [tätä arkistoa](https://microsoft.github.io/forecasting/) ja sen erilaisia mallityyppejä oppiaksesi muita tapoja rakentaa aikajaksomalleja.

## Tehtävä

[Uusi ARIMA-malli](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.