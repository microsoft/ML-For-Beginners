<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T07:45:13+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "lt"
}
-->
# Logistinė regresija kategorijoms prognozuoti

![Logistinės ir linijinės regresijos infografika](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Prieš paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ši pamoka pasiekiama ir R kalba!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Įvadas

Šioje paskutinėje pamokoje apie regresiją, vieną iš pagrindinių _klasikinių_ ML technikų, apžvelgsime logistinės regresijos metodą. Šią techniką galite naudoti norėdami atrasti dėsningumus ir prognozuoti dvejetaines kategorijas. Ar šis saldainis yra šokoladas, ar ne? Ar ši liga yra užkrečiama, ar ne? Ar šis klientas pasirinks šį produktą, ar ne?

Šioje pamokoje sužinosite:

- Naują biblioteką duomenų vizualizacijai
- Logistinės regresijos technikas

✅ Gilinkite savo supratimą apie darbą su šio tipo regresija šiame [mokymosi modulyje](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Būtinos žinios

Dirbdami su moliūgų duomenimis jau pakankamai susipažinome su jais, kad suprastume, jog yra viena dvejetainė kategorija, su kuria galime dirbti: `Spalva`.

Sukurkime logistinės regresijos modelį, kuris prognozuotų, _kokia spalva greičiausiai bus tam tikras moliūgas_ (oranžinė 🎃 ar balta 👻).

> Kodėl kalbame apie dvejetainę klasifikaciją pamokoje apie regresiją? Tik dėl lingvistinio patogumo, nes logistinė regresija iš tiesų yra [klasifikacijos metodas](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), nors ir pagrįstas linijine regresija. Sužinokite apie kitus duomenų klasifikavimo būdus kitame pamokų cikle.

## Apibrėžkite klausimą

Mūsų tikslams išreikšime tai kaip dvejetainę kategoriją: „Balta“ arba „Ne balta“. Mūsų duomenų rinkinyje taip pat yra „dryžuota“ kategorija, tačiau jos pavyzdžių yra nedaug, todėl jos nenaudosime. Ji vis tiek išnyksta, kai pašaliname tuščias reikšmes iš duomenų rinkinio.

> 🎃 Smagus faktas: baltus moliūgus kartais vadiname „vaiduokliais“. Juos nėra lengva išskaptuoti, todėl jie nėra tokie populiarūs kaip oranžiniai, bet atrodo įspūdingai! Taigi galėtume reformuluoti savo klausimą kaip: „Vaiduoklis“ arba „Ne vaiduoklis“. 👻

## Apie logistinę regresiją

Logistinė regresija skiriasi nuo linijinės regresijos, kurią jau išmokote, keliais svarbiais aspektais.

[![ML pradedantiesiems - Logistinės regresijos supratimas mašininio mokymosi klasifikacijai](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML pradedantiesiems - Logistinės regresijos supratimas mašininio mokymosi klasifikacijai")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie logistinės regresijos apžvalgą.

### Dvejetainė klasifikacija

Logistinė regresija nepasiūlo tų pačių funkcijų kaip linijinė regresija. Pirmoji pateikia prognozę apie dvejetainę kategoriją („balta arba ne balta“), o antroji gali prognozuoti tęstines reikšmes, pavyzdžiui, atsižvelgiant į moliūgo kilmę ir derliaus nuėmimo laiką, _kaip padidės jo kaina_.

![Moliūgų klasifikavimo modelis](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografiką sukūrė [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Kitos klasifikacijos

Yra ir kitų logistinės regresijos tipų, įskaitant daugianarę ir tvarką turinčią:

- **Daugianarė**, kai yra daugiau nei viena kategorija - „Oranžinė, Balta ir Dryžuota“.
- **Tvarką turinti**, kai kategorijos yra išdėstytos logiška tvarka, naudinga, jei norėtume logiškai išdėstyti rezultatus, pavyzdžiui, moliūgus, kurie yra išdėstyti pagal ribotą dydžių skaičių (mini, mažas, vidutinis, didelis, labai didelis, milžiniškas).

![Daugianarė vs tvarką turinti regresija](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Kintamieji NETURI būti koreliuoti

Prisiminkite, kaip linijinė regresija geriau veikė su labiau koreliuotais kintamaisiais? Logistinė regresija yra priešinga – kintamieji neturi būti susiję. Tai tinka šiems duomenims, kurių koreliacijos yra gana silpnos.

### Reikia daug švarių duomenų

Logistinė regresija pateiks tikslesnius rezultatus, jei naudosite daugiau duomenų; mūsų mažas duomenų rinkinys nėra optimalus šiai užduočiai, todėl turėkite tai omenyje.

[![ML pradedantiesiems - Duomenų analizė ir paruošimas logistinei regresijai](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML pradedantiesiems - Duomenų analizė ir paruošimas logistinei regresijai")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie duomenų paruošimą linijinei regresijai

✅ Pagalvokite apie duomenų tipus, kurie geriausiai tiktų logistinei regresijai

## Užduotis - sutvarkykite duomenis

Pirmiausia šiek tiek išvalykite duomenis, pašalindami tuščias reikšmes ir pasirinkdami tik kelis stulpelius:

1. Pridėkite šį kodą:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Visada galite pažvelgti į savo naują duomenų rėmelį:

    ```python
    pumpkins.info
    ```

### Vizualizacija - kategorinis grafikas

Iki šiol įkėlėte [pradinį užrašų knygelę](../../../../2-Regression/4-Logistic/notebook.ipynb) su moliūgų duomenimis ir išvalėte ją, kad išsaugotumėte duomenų rinkinį, kuriame yra keli kintamieji, įskaitant `Spalvą`. Vizualizuokime duomenų rėmelį užrašų knygelėje naudodami kitą biblioteką: [Seaborn](https://seaborn.pydata.org/index.html), kuri yra sukurta ant Matplotlib, kurį naudojome anksčiau.

Seaborn siūlo keletą įdomių būdų vizualizuoti duomenis. Pavyzdžiui, galite palyginti duomenų pasiskirstymą pagal `Variety` ir `Color` kategoriniame grafike.

1. Sukurkite tokį grafiką naudodami `catplot` funkciją, naudodami mūsų moliūgų duomenis `pumpkins` ir nurodydami spalvų žemėlapį kiekvienai moliūgų kategorijai (oranžinė arba balta):

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

    ![Duomenų vizualizacijos tinklelis](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Stebėdami duomenis galite pamatyti, kaip `Spalva` duomenys susiję su `Variety`.

    ✅ Atsižvelgdami į šį kategorinį grafiką, kokius įdomius tyrimus galite įsivaizduoti?

### Duomenų paruošimas: požymių ir etikečių kodavimas

Mūsų moliūgų duomenų rinkinyje visos stulpelių reikšmės yra tekstinės. Dirbti su kategoriniais duomenimis žmonėms yra intuityvu, tačiau mašinoms – ne. Mašininio mokymosi algoritmai geriau veikia su skaitiniais duomenimis. Todėl kodavimas yra labai svarbus duomenų paruošimo etapas, nes jis leidžia paversti kategorinius duomenis skaitiniais, neprarandant informacijos. Geras kodavimas padeda sukurti gerą modelį.

Požymių kodavimui yra du pagrindiniai kodavimo tipai:

1. Ordinalinis kodavimas: jis gerai tinka tvarką turintiems kintamiesiems, kurie yra kategoriniai kintamieji, kurių duomenys turi logišką tvarką, kaip `Item Size` stulpelis mūsų duomenų rinkinyje. Jis sukuria žemėlapį, kuriame kiekviena kategorija yra atvaizduojama skaičiumi, kuris atitinka kategorijos tvarką stulpelyje.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Kategorinis kodavimas: jis gerai tinka nominaliems kintamiesiems, kurie yra kategoriniai kintamieji, kurių duomenys neturi logiškos tvarkos, kaip visi požymiai, išskyrus `Item Size` mūsų duomenų rinkinyje. Tai yra vieno karšto kodavimo metodas, kuris reiškia, kad kiekviena kategorija yra atvaizduojama dvejetainiu stulpeliu: užkoduota reikšmė yra lygi 1, jei moliūgas priklauso tai `Variety`, ir 0, jei ne.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

Tada `ColumnTransformer` naudojamas keliems kodavimo metodams sujungti į vieną žingsnį ir pritaikyti juos tinkamiems stulpeliams.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

Kita vertus, etiketės kodavimui naudojame scikit-learn `LabelEncoder` klasę, kuri yra pagalbinė klasė, padedanti normalizuoti etiketes, kad jos turėtų tik reikšmes tarp 0 ir n_classes-1 (čia, 0 ir 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

Kai užkoduojame požymius ir etiketes, galime juos sujungti į naują duomenų rėmelį `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ Kokie yra ordinalinio kodavimo privalumai `Item Size` stulpeliui?

### Analizuokite kintamųjų tarpusavio ryšius

Dabar, kai paruošėme duomenis, galime analizuoti požymių ir etikečių tarpusavio ryšius, kad suprastume, kaip gerai modelis galės prognozuoti etiketę pagal požymius. Geriausias būdas atlikti tokio tipo analizę yra duomenų vizualizavimas. Vėl naudosime Seaborn `catplot` funkciją, kad vizualizuotume `Item Size`, `Variety` ir `Color` tarpusavio ryšius kategoriniame grafike. Norėdami geriau vizualizuoti duomenis, naudosime užkoduotą `Item Size` stulpelį ir neužkoduotą `Variety` stulpelį.

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

![Duomenų vizualizacijos grafikas](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Naudokite „swarm“ grafiką

Kadangi `Color` yra dvejetainė kategorija (Balta arba Ne), jai reikia „[specialaus požiūrio](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) vizualizacijai“. Yra ir kitų būdų vizualizuoti šios kategorijos ryšį su kitais kintamaisiais.

Galite vizualizuoti kintamuosius šalia vienas kito naudodami Seaborn grafikus.

1. Išbandykite „swarm“ grafiką, kad parodytumėte reikšmių pasiskirstymą:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Duomenų vizualizacijos „swarm“ grafikas](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Atkreipkite dėmesį**: aukščiau pateiktas kodas gali generuoti įspėjimą, nes Seaborn nepavyksta atvaizduoti tokio kiekio duomenų taškų „swarm“ grafike. Galimas sprendimas yra sumažinti žymeklio dydį, naudojant „size“ parametrą. Tačiau atminkite, kad tai gali paveikti grafiko skaitomumą.

> **🧮 Parodykite matematiką**
>
> Logistinė regresija remiasi „maksimalaus tikėtinumo“ koncepcija, naudojant [sigmoidines funkcijas](https://wikipedia.org/wiki/Sigmoid_function). Sigmoidinė funkcija grafike atrodo kaip „S“ formos kreivė. Ji paima reikšmę ir priskiria ją intervalui tarp 0 ir 1. Jos kreivė taip pat vadinama „logistine kreive“. Jos formulė atrodo taip:
>
> ![logistinė funkcija](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> kur sigmoidės vidurys yra x ašies 0 taške, L yra kreivės maksimali reikšmė, o k yra kreivės statumas. Jei funkcijos rezultatas yra didesnis nei 0.5, atitinkama etiketė bus priskirta „1“ klasei iš dvejetainio pasirinkimo. Jei ne, ji bus priskirta „0“ klasei.

## Sukurkite savo modelį

Sukurti modelį, kuris rastų šias dvejetaines klasifikacijas, yra stebėtinai paprasta naudojant Scikit-learn.

[![ML pradedantiesiems - Logistinė regresija duomenų klasifikacijai](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML pradedantiesiems - Logistinė regresija duomenų klasifikacijai")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie linijinės regresijos modelio kūrimą

1. Pasirinkite kintamuosius, kuriuos norite naudoti savo klasifikavimo modelyje, ir padalykite mokymo bei testavimo rinkinius, iškviesdami `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Dabar galite apmokyti savo modelį, iškviesdami `fit()` su mokymo duomenimis, ir atspausdinti jo rezultatą:

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

    Pažvelkite į savo modelio rezultatų lentelę. Ji nėra bloga, atsižvelgiant į tai, kad turite tik apie 1000 duomenų eilučių:

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

## Geresnis supratimas per klaidų matricą

Nors galite gauti rezultatų lentelę [terminais](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report), atspausdindami aukščiau pateiktus elementus, galbūt galėsite geriau suprasti savo modelį naudodami [klaidų matricą](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix), kuri padeda suprasti, kaip modelis veikia.

> 🎓 „[Klaidų matrica](https://wikipedia.org/wiki/Confusion_matrix)“ (arba „klaidų matrica“) yra lentelė, kuri išreiškia jūsų modelio tikrus ir netikrus teigiamus bei neigiamus rezultatus, taip įvertinant prognozių tikslumą.

1. Norėdami naudoti klaidų matricą, iškvieskite `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Pažvelkite į savo modelio klaidų matricą:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Scikit-learn klaidų matricose eilutės (0 ašis) yra tikros etiketės, o stulpeliai (1 ašis) yra prognozuotos etiketės.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Kas čia vyksta? Tarkime, mūsų modelis yra paprašytas klasifikuoti moliūgus tarp dviejų dvejetainių kategorijų, kategorijos „balta“ ir kategorijos „ne balta“.

- Jei jūsų modelis prognozuoja moliūgą kaip ne baltą, o jis iš tikrųjų priklauso kategorijai „ne balta“, tai vadiname tikru neigiamu rezultatu, kurį rodo viršutinis
Kaip painiavos matrica susijusi su tikslumu ir atšaukimu? Atminkite, kad aukščiau pateiktoje klasifikacijos ataskaitoje buvo nurodytas tikslumas (0.85) ir atšaukimas (0.67).

Tikslumas = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Atšaukimas = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Klausimas: Remiantis painiavos matrica, kaip sekėsi modeliui? Atsakymas: Ne blogai; yra nemažai teigiamų neigiamų atvejų, tačiau taip pat keletas klaidingų neigiamų atvejų.

Grįžkime prie terminų, kuriuos matėme anksčiau, naudodamiesi painiavos matricos TP/TN ir FP/FN žemėlapiu:

🎓 Tikslumas: TP/(TP + FP) Reikšmingų atvejų dalis tarp gautų atvejų (pvz., kurie žymėjimai buvo gerai pažymėti)

🎓 Atšaukimas: TP/(TP + FN) Reikšmingų atvejų dalis, kurie buvo gauti, nesvarbu, ar jie buvo gerai pažymėti, ar ne

🎓 f1-rezultatas: (2 * tikslumas * atšaukimas)/(tikslumas + atšaukimas) Tikslumo ir atšaukimo svertinis vidurkis, geriausias yra 1, blogiausias - 0

🎓 Palaikymas: Kiekvieno gauto žymėjimo pasikartojimų skaičius

🎓 Tikslumas: (TP + TN)/(TP + TN + FP + FN) Procentas žymėjimų, kurie buvo tiksliai nuspėti mėginyje.

🎓 Makro vidurkis: Neįvertintų vidutinių metrikų skaičiavimas kiekvienam žymėjimui, neatsižvelgiant į žymėjimų disbalansą.

🎓 Svertinis vidurkis: Vidutinių metrikų skaičiavimas kiekvienam žymėjimui, atsižvelgiant į žymėjimų disbalansą, sveriant juos pagal jų palaikymą (tikrų atvejų skaičių kiekvienam žymėjimui).

✅ Ar galite pagalvoti, kurią metriką reikėtų stebėti, jei norite, kad jūsų modelis sumažintų klaidingų neigiamų atvejų skaičių?

## Vizualizuokite šio modelio ROC kreivę

[![ML pradedantiesiems - Logistinės regresijos našumo analizė su ROC kreivėmis](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML pradedantiesiems - Logistinės regresijos našumo analizė su ROC kreivėmis")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie ROC kreives

Atlikime dar vieną vizualizaciją, kad pamatytume vadinamąją „ROC“ kreivę:

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

Naudodami Matplotlib, nubrėžkite modelio [Gavimo veikimo charakteristikos](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) arba ROC kreivę. ROC kreivės dažnai naudojamos norint pamatyti klasifikatoriaus išvestį pagal tikrus ir klaidingus teigiamus atvejus. „ROC kreivėse paprastai Y ašyje pateikiamas tikrų teigiamų atvejų rodiklis, o X ašyje - klaidingų teigiamų atvejų rodiklis.“ Taigi kreivės statumas ir erdvė tarp vidurio linijos ir kreivės yra svarbūs: norite kreivės, kuri greitai kyla aukštyn ir virš linijos. Mūsų atveju pradžioje yra klaidingų teigiamų atvejų, o tada linija tinkamai kyla aukštyn ir virš.

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Galiausiai naudokite Scikit-learn [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score), kad apskaičiuotumėte faktinę „Plotą po kreive“ (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Rezultatas yra `0.9749908725812341`. Kadangi AUC svyruoja nuo 0 iki 1, norite didelio rezultato, nes modelis, kuris 100% tiksliai prognozuoja, turės AUC lygią 1; šiuo atveju modelis yra _gana geras_.

Ateities pamokose apie klasifikacijas sužinosite, kaip iteruoti, kad pagerintumėte savo modelio rezultatus. Bet kol kas sveikiname! Jūs baigėte šias regresijos pamokas!

---
## 🚀Iššūkis

Logistinėje regresijoje yra daug ką išnagrinėti! Tačiau geriausias būdas mokytis yra eksperimentuoti. Suraskite duomenų rinkinį, kuris tinka tokio tipo analizei, ir sukurkite modelį su juo. Ką išmokote? patarimas: pabandykite [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) ieškoti įdomių duomenų rinkinių.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Perskaitykite pirmuosius kelis [šio Stanfordo dokumento](https://web.stanford.edu/~jurafsky/slp3/5.pdf) puslapius apie praktinius logistinės regresijos panaudojimus. Pagalvokite apie užduotis, kurios geriau tinka vienam ar kitam regresijos tipui, kuriuos studijavome iki šiol. Kas veiktų geriausiai?

## Užduotis

[Pakartokite šią regresiją](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.