<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-04T23:33:10+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "fi"
}
-->
# Logistinen regressio kategorioiden ennustamiseen

![Logistinen vs. lineaarinen regressio -infografiikka](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Esiluentakysely](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla myös R-kielellä!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Johdanto

Tässä viimeisessä oppitunnissa regressiosta, joka on yksi perinteisistä _klassisen_ koneoppimisen tekniikoista, tarkastelemme logistista regressiota. Tätä tekniikkaa käytetään mallintamaan ja ennustamaan binäärisiä kategorioita. Onko tämä karkki suklaata vai ei? Onko tämä tauti tarttuva vai ei? Valitseeko asiakas tämän tuotteen vai ei?

Tässä oppitunnissa opit:

- Uuden kirjaston datan visualisointiin
- Logistisen regression tekniikoita

✅ Syvennä ymmärrystäsi tämän tyyppisestä regressiosta tässä [Learn-moduulissa](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Esitiedot

Työskenneltyämme kurpitsadataa käyttäen olemme nyt riittävän tuttuja sen kanssa huomataksemme, että siinä on yksi binäärinen kategoria, jonka kanssa voimme työskennellä: `Color`.

Rakennetaan logistinen regressiomalli ennustamaan, _minkä värinen tietty kurpitsa todennäköisesti on_ (oranssi 🎃 vai valkoinen 👻).

> Miksi puhumme binäärisestä luokittelusta regressiota käsittelevässä oppitunnissa? Ainoastaan kielellisen mukavuuden vuoksi, sillä logistinen regressio on [todellisuudessa luokittelumenetelmä](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), vaikkakin lineaarinen sellainen. Opit muista tavoista luokitella dataa seuraavassa oppituntiryhmässä.

## Määrittele kysymys

Tässä yhteydessä ilmaisemme tämän binäärisesti: 'Valkoinen' tai 'Ei valkoinen'. Datasetissämme on myös 'raidallinen' kategoria, mutta sen esiintymiä on vähän, joten emme käytä sitä. Se katoaa joka tapauksessa, kun poistamme datasetistä puuttuvat arvot.

> 🎃 Hauska fakta: kutsumme joskus valkoisia kurpitsoja 'aavekurpitsoiksi'. Niitä ei ole kovin helppo kaivertaa, joten ne eivät ole yhtä suosittuja kuin oranssit, mutta ne näyttävät siisteiltä! Voisimme siis myös muotoilla kysymyksemme näin: 'Aave' vai 'Ei aave'. 👻

## Tietoa logistisesta regressiosta

Logistinen regressio eroaa aiemmin oppimastasi lineaarisesta regressiosta muutamalla tärkeällä tavalla.

[![ML aloittelijoille - Logistisen regression ymmärtäminen koneoppimisen luokittelussa](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML aloittelijoille - Logistisen regression ymmärtäminen koneoppimisen luokittelussa")

> 🎥 Klikkaa yllä olevaa kuvaa saadaksesi lyhyen videoesittelyn logistisesta regressiosta.

### Binäärinen luokittelu

Logistinen regressio ei tarjoa samoja ominaisuuksia kuin lineaarinen regressio. Edellinen tarjoaa ennusteen binäärisestä kategoriasta ("valkoinen tai ei valkoinen"), kun taas jälkimmäinen pystyy ennustamaan jatkuvia arvoja, esimerkiksi kurpitsan alkuperän ja sadonkorjuuajan perusteella, _kuinka paljon sen hinta nousee_.

![Kurpitsan luokittelumalli](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Muut luokittelut

Logistisessa regressiossa on myös muita tyyppejä, kuten multinomiaalinen ja ordinaalinen:

- **Multinomiaalinen**, jossa on useampi kuin yksi kategoria - "Oranssi, Valkoinen ja Raidallinen".
- **Ordinaalinen**, jossa on järjestettyjä kategorioita, hyödyllinen, jos haluamme järjestää tulokset loogisesti, kuten kurpitsat, jotka on järjestetty kokojen mukaan (mini, sm, med, lg, xl, xxl).

![Multinomiaalinen vs. ordinaalinen regressio](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Muuttujien EI tarvitse korreloida

Muistatko, kuinka lineaarinen regressio toimi paremmin korreloivien muuttujien kanssa? Logistinen regressio on päinvastainen - muuttujien ei tarvitse olla linjassa. Tämä sopii tälle datalle, jossa korrelaatiot ovat melko heikkoja.

### Tarvitset paljon puhdasta dataa

Logistinen regressio antaa tarkempia tuloksia, jos käytät enemmän dataa; pieni datasetimme ei ole optimaalinen tähän tehtävään, joten pidä tämä mielessä.

[![ML aloittelijoille - Datan analysointi ja valmistelu logistista regressiota varten](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML aloittelijoille - Datan analysointi ja valmistelu logistista regressiota varten")

> 🎥 Klikkaa yllä olevaa kuvaa saadaksesi lyhyen videoesittelyn datan valmistelusta lineaarista regressiota varten.

✅ Mieti, millaiset datatyypit sopisivat hyvin logistiseen regressioon.

## Harjoitus - siivoa data

Ensiksi siivoa dataa hieman, poista puuttuvat arvot ja valitse vain tietyt sarakkeet:

1. Lisää seuraava koodi:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Voit aina vilkaista uutta dataframeasi:

    ```python
    pumpkins.info
    ```

### Visualisointi - kategorinen kaavio

Olet nyt ladannut [aloitusmuistion](../../../../2-Regression/4-Logistic/notebook.ipynb) kurpitsadatalla ja siivonnut sen niin, että datasetissä on muutama muuttuja, mukaan lukien `Color`. Visualisoidaan dataframe muistiossa käyttäen eri kirjastoa: [Seaborn](https://seaborn.pydata.org/index.html), joka on rakennettu aiemmin käyttämämme Matplotlibin päälle.

Seaborn tarjoaa käteviä tapoja visualisoida dataa. Esimerkiksi voit verrata datan jakaumia jokaiselle `Variety`- ja `Color`-arvolle kategorisessa kaaviossa.

1. Luo tällainen kaavio käyttämällä `catplot`-funktiota, käyttäen kurpitsadataamme `pumpkins` ja määrittämällä värikartta jokaiselle kurpitsakategorialle (oranssi tai valkoinen):

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![Visualisoidun datan ruudukko](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Tarkastelemalla dataa voit nähdä, miten `Color`-data liittyy `Variety`-arvoon.

    ✅ Mitä mielenkiintoisia tutkimusaiheita voit keksiä tämän kategorisen kaavion perusteella?

### Datan esikäsittely: ominaisuuksien ja luokkien koodaus

Datasetissämme kaikki sarakkeet sisältävät merkkijonoarvoja. Kategorisen datan käsittely on ihmisille intuitiivista, mutta ei koneille. Koneoppimisalgoritmit toimivat hyvin numeroiden kanssa. Siksi koodaus on erittäin tärkeä vaihe datan esikäsittelyssä, koska se mahdollistaa kategorisen datan muuttamisen numeeriseksi dataksi ilman tietojen menetystä. Hyvä koodaus johtaa hyvän mallin rakentamiseen.

Ominaisuuksien koodaukseen on kaksi päätyyppiä:

1. Ordinaalikoodaus: sopii hyvin ordinaalisille muuttujille, jotka ovat kategorisia muuttujia, joiden data noudattaa loogista järjestystä, kuten datasetimme `Item Size` -sarake. Tämä luo kartoituksen, jossa jokainen kategoria esitetään numerolla, joka on sarakkeen kategorian järjestysnumero.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Kategorinen koodaus: sopii hyvin nominaalisille muuttujille, jotka ovat kategorisia muuttujia, joiden data ei noudata loogista järjestystä, kuten kaikki muut datasetin ominaisuudet paitsi `Item Size`. Tämä on yksi-hot-koodaus, mikä tarkoittaa, että jokainen kategoria esitetään binäärisellä sarakkeella: koodattu muuttuja on yhtä kuin 1, jos kurpitsa kuuluu kyseiseen `Variety`-arvoon, ja 0 muuten.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

Sitten `ColumnTransformer`-luokkaa käytetään yhdistämään useita koodauksia yhdeksi vaiheeksi ja soveltamaan niitä sopiviin sarakkeisiin.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

Toisaalta luokan koodaamiseen käytämme scikit-learnin `LabelEncoder`-luokkaa, joka on apuluokka normalisoimaan luokat siten, että ne sisältävät vain arvoja välillä 0 ja n_classes-1 (tässä 0 ja 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

Kun olemme koodanneet ominaisuudet ja luokan, voimme yhdistää ne uuteen dataframeen `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ Mitkä ovat ordinaalikoodauksen edut `Item Size` -sarakkeelle?

### Analysoi muuttujien välisiä suhteita

Nyt kun olemme esikäsitelleet datamme, voimme analysoida ominaisuuksien ja luokan välisiä suhteita saadaksemme käsityksen siitä, kuinka hyvin malli pystyy ennustamaan luokan annettujen ominaisuuksien perusteella. Paras tapa suorittaa tämäntyyppinen analyysi on datan visualisointi. Käytämme jälleen Seabornin `catplot`-funktiota visualisoidaksemme suhteet `Item Size`-, `Variety`- ja `Color`-arvojen välillä kategorisessa kaaviossa. Visualisointia varten käytämme koodattua `Item Size` -saraketta ja koodaamatonta `Variety`-saraketta.

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```

![Visualisoidun datan kaavio](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Käytä swarm-kaaviota

Koska `Color` on binäärinen kategoria (valkoinen tai ei), se tarvitsee '[erikoistuneen lähestymistavan](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar)' visualisointiin. On olemassa muita tapoja visualisoida tämän kategorian suhde muihin muuttujiin.

Voit visualisoida muuttujia rinnakkain Seabornin kaavioilla.

1. Kokeile 'swarm'-kaaviota arvojen jakauman näyttämiseksi:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Visualisoidun datan swarm-kaavio](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Huomio**: Yllä oleva koodi saattaa tuottaa varoituksen, koska Seaborn ei pysty esittämään näin suurta määrää datapisteitä swarm-kaaviossa. Mahdollinen ratkaisu on pienentää merkkien kokoa käyttämällä 'size'-parametria. Huomaa kuitenkin, että tämä voi vaikuttaa kaavion luettavuuteen.

> **🧮 Näytä matematiikka**
>
> Logistinen regressio perustuu 'maksimimäärän todennäköisyyden' käsitteeseen käyttäen [sigmoidifunktioita](https://wikipedia.org/wiki/Sigmoid_function). 'Sigmoidifunktio' näyttää graafilla 'S'-muotoiselta. Se ottaa arvon ja muuntaa sen välillä 0 ja 1. Sen käyrää kutsutaan myös 'logistiseksi käyräksi'. Sen kaava näyttää tältä:
>
> ![logistinen funktio](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> missä sigmoidin keskipiste on x:n 0-pisteessä, L on käyrän maksimiarvo ja k on käyrän jyrkkyys. Jos funktion tulos on yli 0,5, kyseinen luokka saa binäärisen valinnan arvon '1'. Muussa tapauksessa se luokitellaan arvoksi '0'.

## Rakenna mallisi

Binäärisen luokittelun mallin rakentaminen on yllättävän suoraviivaista Scikit-learnilla.

[![ML aloittelijoille - Logistinen regressio datan luokitteluun](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML aloittelijoille - Logistinen regressio datan luokitteluun")

> 🎥 Klikkaa yllä olevaa kuvaa saadaksesi lyhyen videoesittelyn logistisen regressiomallin rakentamisesta.

1. Valitse muuttujat, joita haluat käyttää luokittelumallissasi, ja jaa data koulutus- ja testijoukkoihin kutsumalla `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Nyt voit kouluttaa mallisi kutsumalla `fit()` koulutusdatalla ja tulostaa sen tuloksen:

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    Tarkastele mallisi tulostaulukkoa. Se ei ole huono, kun otetaan huomioon, että sinulla on vain noin 1000 riviä dataa:

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## Parempi ymmärrys virhemaatriksin avulla

Vaikka voit saada tulostaulukon raportin [termeistä](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) tulostamalla yllä olevat kohteet, saatat ymmärtää malliasi helpommin käyttämällä [virhemaatriksia](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) mallin suorituskyvyn arvioimiseen.

> 🎓 '[Virhemaatriksi](https://wikipedia.org/wiki/Confusion_matrix)' (tai 'virhemaatriksi') on taulukko, joka ilmaisee mallisi todelliset vs. väärät positiiviset ja negatiiviset arvot, ja siten arvioi ennusteiden tarkkuutta.

1. Käytä virhemaatriksia kutsumalla `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Tarkastele mallisi virhemaatriksia:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Scikit-learnissa virhemaatriksin rivit (akseli 0) ovat todellisia luokkia ja sarakkeet (akseli 1) ovat ennustettuja luokkia.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Mitä tässä tapahtuu? Oletetaan, että malliamme pyydetään luokittelemaan kurpitsat kahteen binääriseen kategoriaan, kategoriaan 'valkoinen' ja kategoriaan 'ei-valkoinen'.

- Jos mallisi ennustaa kurpitsan ei-valkoiseksi ja se kuuluu todellisuudessa kategoriaan 'ei-valkoinen', kutsumme sitä oikeaksi negatiiviseksi (TN), joka näkyy vasemmassa yläkulmassa.
- Jos mallisi ennustaa kurpitsan valkoiseksi ja se kuuluu todellisuudessa kategoriaan 'ei-valkoinen', kutsumme sitä vääräksi positiiviseksi (FP), joka näkyy oikeassa yläkulmassa.
- Jos mallisi ennustaa kurpitsan ei-valkoiseksi ja se kuuluu todellisuudessa kategoriaan 'valkoinen', kutsumme sitä vääräksi negatiiviseksi (FN), joka näkyy vasemmassa alakulmassa.
- Jos mallisi ennustaa kurpitsan valkoiseksi ja se kuuluu todellisuudessa kategoriaan 'valkoinen', kutsumme sitä oikeaksi positiiviseksi (TP), joka näkyy oikeassa alakulmassa.

Kuten arvata saattaa, on toivottavaa, että oikeiden positiivisten ja oikeiden negatiivisten määrä on suuri ja väärien positiivisten ja väärien negatiivisten määrä pieni, mikä tarkoittaa, että malli toimii paremmin.
Miten sekaannusmatriisi liittyy tarkkuuteen ja kattavuuteen? Muista, että yllä tulostettu luokitteluraportti näytti tarkkuuden (0.85) ja kattavuuden (0.67).

Tarkkuus = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Kattavuus = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ K: Miten malli suoriutui sekaannusmatriisin perusteella? V: Ei huono; on hyvä määrä oikeita negatiivisia, mutta myös joitakin vääriä negatiivisia.

Palataan aiemmin nähtyihin termeihin sekaannusmatriisin TP/TN ja FP/FN -kartoituksen avulla:

🎓 Tarkkuus: TP/(TP + FP) Relevanttien tapausten osuus haetuista tapauksista (esim. mitkä tunnisteet merkittiin oikein)

🎓 Kattavuus: TP/(TP + FN) Relevanttien tapausten osuus, jotka haettiin, riippumatta siitä, merkittiinkö ne oikein vai ei

🎓 f1-pisteet: (2 * tarkkuus * kattavuus)/(tarkkuus + kattavuus) Painotettu keskiarvo tarkkuudesta ja kattavuudesta, paras arvo on 1 ja huonoin 0

🎓 Tuki: Haettujen tunnisteiden esiintymien lukumäärä

🎓 Tarkkuus: (TP + TN)/(TP + TN + FP + FN) Oikein ennustettujen tunnisteiden prosenttiosuus näytteestä.

🎓 Makrokeskiarvo: Painottamattomien keskiarvojen laskeminen kullekin tunnisteelle, ottamatta huomioon tunnisteiden epätasapainoa.

🎓 Painotettu keskiarvo: Keskiarvojen laskeminen kullekin tunnisteelle, ottaen huomioon tunnisteiden epätasapaino painottamalla niitä niiden tuen (kunkin tunnisteen oikeiden tapausten lukumäärän) mukaan.

✅ Voitko miettiä, mitä metriikkaa sinun tulisi seurata, jos haluat vähentää väärien negatiivisten määrää?

## Visualisoi tämän mallin ROC-käyrä

[![ML aloittelijoille - Logistisen regressiomallin suorituskyvyn analysointi ROC-käyrillä](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML aloittelijoille - Logistisen regressiomallin suorituskyvyn analysointi ROC-käyrillä")

> 🎥 Klikkaa yllä olevaa kuvaa saadaksesi lyhyen videokatsauksen ROC-käyristä

Tehdään vielä yksi visualisointi, jotta näemme niin sanotun 'ROC'-käyrän:

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Käyttämällä Matplotlibia, piirrä mallin [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) eli ROC. ROC-käyriä käytetään usein luokittelijan tulosten tarkasteluun oikeiden ja väärien positiivisten osalta. "ROC-käyrillä Y-akselilla on yleensä oikeiden positiivisten osuus ja X-akselilla väärien positiivisten osuus." Käyrän jyrkkyys ja väli keskilinjan ja käyrän välillä ovat tärkeitä: haluat käyrän, joka nopeasti nousee ja menee linjan yli. Meidän tapauksessamme on aluksi vääriä positiivisia, ja sitten linja nousee ja menee kunnolla yli:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Lopuksi, käytä Scikit-learnin [`roc_auc_score` APIa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) laskeaksesi todellinen 'Area Under the Curve' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Tulos on `0.9749908725812341`. Koska AUC vaihtelee välillä 0–1, haluat korkean pistemäärän, sillä malli, joka on 100 % oikeassa ennusteissaan, saa AUC-arvon 1; tässä tapauksessa malli on _melko hyvä_.

Tulevissa luokitteluun liittyvissä oppitunneissa opit, kuinka voit parantaa mallisi pisteitä iteratiivisesti. Mutta nyt, onneksi olkoon! Olet suorittanut nämä regressio-oppitunnit!

---
## 🚀Haaste

Logistiseen regressioon liittyy paljon enemmän! Mutta paras tapa oppia on kokeilla. Etsi datasetti, joka sopii tämän tyyppiseen analyysiin, ja rakenna malli sen avulla. Mitä opit? vinkki: kokeile [Kagglea](https://www.kaggle.com/search?q=logistic+regression+datasets) löytääksesi mielenkiintoisia datasettejä.

## [Luennon jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itseopiskelu

Lue [tämän Stanfordin artikkelin](https://web.stanford.edu/~jurafsky/slp3/5.pdf) ensimmäiset sivut logistisen regression käytännön sovelluksista. Mieti tehtäviä, jotka sopivat paremmin jommallekummalle regressiotyypille, joita olemme tähän mennessä opiskelleet. Mikä toimisi parhaiten?

## Tehtävä

[Uudelleen yrittäminen tässä regressiossa](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.