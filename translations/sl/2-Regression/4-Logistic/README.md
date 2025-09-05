<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T11:35:35+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "sl"
}
-->
# Logistična regresija za napovedovanje kategorij

![Infografika: Logistična vs. linearna regresija](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ta lekcija je na voljo tudi v R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Uvod

V tej zadnji lekciji o regresiji, eni izmed osnovnih _klasičnih_ tehnik strojnega učenja, si bomo ogledali logistično regresijo. To tehniko bi uporabili za odkrivanje vzorcev za napovedovanje binarnih kategorij. Ali je ta sladkarija čokolada ali ne? Ali je ta bolezen nalezljiva ali ne? Ali bo ta stranka izbrala ta izdelek ali ne?

V tej lekciji boste spoznali:

- Novo knjižnico za vizualizacijo podatkov
- Tehnike logistične regresije

✅ Poglobite svoje razumevanje dela s to vrsto regresije v tem [učnem modulu](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Predpogoji

Ker smo že delali s podatki o bučah, smo dovolj seznanjeni, da opazimo, da obstaja ena binarna kategorija, s katero lahko delamo: `Barva`.

Zgradimo model logistične regresije, da napovemo, _kakšne barve bo določena buča_ (oranžna 🎃 ali bela 👻), glede na nekatere spremenljivke.

> Zakaj govorimo o binarni klasifikaciji v lekciji o regresiji? Izključno zaradi jezikovne priročnosti, saj je logistična regresija [pravzaprav metoda klasifikacije](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), čeprav temelji na linearni osnovi. O drugih načinih klasifikacije podatkov se boste naučili v naslednji skupini lekcij.

## Določite vprašanje

Za naše namene bomo to izrazili kot binarno: 'Bela' ali 'Ne bela'. V našem naboru podatkov obstaja tudi kategorija 'črtasta', vendar je primerov te kategorije malo, zato je ne bomo uporabili. Ta kategorija tako ali tako izgine, ko odstranimo manjkajoče vrednosti iz nabora podatkov.

> 🎃 Zabavno dejstvo: bele buče včasih imenujemo 'duhove buče'. Niso zelo enostavne za rezljanje, zato niso tako priljubljene kot oranžne, vendar so videti kul! Tako bi lahko naše vprašanje preoblikovali tudi kot: 'Duh' ali 'Ne duh'. 👻

## O logistični regresiji

Logistična regresija se od linearne regresije, ki ste jo spoznali prej, razlikuje v nekaj pomembnih vidikih.

[![Strojno učenje za začetnike - Razumevanje logistične regresije za klasifikacijo](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "Strojno učenje za začetnike - Razumevanje logistične regresije za klasifikacijo")

> 🎥 Kliknite zgornjo sliko za kratek video pregled logistične regresije.

### Binarna klasifikacija

Logistična regresija ne ponuja enakih funkcij kot linearna regresija. Prva ponuja napoved binarne kategorije ("bela ali ne bela"), medtem ko je druga sposobna napovedovati neprekinjene vrednosti, na primer glede na izvor buče in čas žetve, _koliko se bo njena cena zvišala_.

![Model klasifikacije buč](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Druge klasifikacije

Obstajajo tudi druge vrste logistične regresije, vključno z multinomno in ordinalno:

- **Multinomna**, ki vključuje več kot eno kategorijo - "Oranžna, Bela in Črtasta".
- **Ordinalna**, ki vključuje urejene kategorije, uporabna, če želimo logično urediti naše rezultate, kot so buče, ki so razvrščene po končnem številu velikosti (mini, sm, med, lg, xl, xxl).

![Multinomna vs. ordinalna regresija](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Spremenljivke NI NUJNO, da so korelirane

Se spomnite, kako je linearna regresija bolje delovala z bolj koreliranimi spremenljivkami? Logistična regresija je nasprotje - spremenljivke ni nujno, da se ujemajo. To ustreza tem podatkom, ki imajo nekoliko šibke korelacije.

### Potrebujete veliko čistih podatkov

Logistična regresija bo dala natančnejše rezultate, če uporabite več podatkov; naš majhen nabor podatkov ni optimalen za to nalogo, zato to upoštevajte.

[![Strojno učenje za začetnike - Analiza in priprava podatkov za logistično regresijo](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "Strojno učenje za začetnike - Analiza in priprava podatkov za logistično regresijo")

✅ Razmislite o vrstah podatkov, ki bi bile primerne za logistično regresijo.

## Naloga - uredite podatke

Najprej nekoliko očistite podatke, odstranite manjkajoče vrednosti in izberite le nekatere stolpce:

1. Dodajte naslednjo kodo:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Vedno lahko pokukate v svoj novi podatkovni okvir:

    ```python
    pumpkins.info
    ```

### Vizualizacija - kategorni graf

Do zdaj ste znova naložili [začetni zvezek](../../../../2-Regression/4-Logistic/notebook.ipynb) s podatki o bučah in jih očistili, da ste ohranili nabor podatkov, ki vsebuje nekaj spremenljivk, vključno z `Barvo`. Vizualizirajmo podatkovni okvir v zvezku z uporabo druge knjižnice: [Seaborn](https://seaborn.pydata.org/index.html), ki temelji na Matplotlibu, ki smo ga uporabljali prej.

Seaborn ponuja nekaj zanimivih načinov za vizualizacijo vaših podatkov. Na primer, lahko primerjate porazdelitve podatkov za vsako `Sorto` in `Barvo` v kategorni graf.

1. Ustvarite tak graf z uporabo funkcije `catplot`, uporabite naše podatke o bučah `pumpkins` in določite barvno preslikavo za vsako kategorijo buč (oranžna ali bela):

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

    ![Mreža vizualiziranih podatkov](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Z opazovanjem podatkov lahko vidite, kako se podatki o barvi nanašajo na sorto.

    ✅ Glede na ta kategorni graf, kakšne zanimive raziskave si lahko zamislite?

### Predobdelava podatkov: kodiranje značilnosti in oznak
Naš nabor podatkov o bučah vsebuje nizovne vrednosti za vse svoje stolpce. Delo s kategorijskimi podatki je za ljudi intuitivno, za stroje pa ne. Algoritmi strojnega učenja dobro delujejo s številkami. Zato je kodiranje zelo pomemben korak v fazi predobdelave podatkov, saj nam omogoča pretvorbo kategorijskih podatkov v številčne podatke, ne da bi pri tem izgubili kakršne koli informacije. Dobro kodiranje vodi k gradnji dobrega modela.

Za kodiranje značilnosti obstajata dve glavni vrsti kodirnikov:

1. Ordinalni kodirnik: dobro se prilega ordinalnim spremenljivkam, ki so kategorijske spremenljivke, kjer njihovi podatki sledijo logičnemu vrstnemu redu, kot je stolpec `Item Size` v našem naboru podatkov. Ustvari preslikavo, tako da je vsaka kategorija predstavljena s številko, ki je vrstni red kategorije v stolpcu.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Kategorni kodirnik: dobro se prilega nominalnim spremenljivkam, ki so kategorijske spremenljivke, kjer njihovi podatki ne sledijo logičnemu vrstnemu redu, kot so vse značilnosti, razen `Item Size` v našem naboru podatkov. To je kodiranje z eno vročo vrednostjo, kar pomeni, da je vsaka kategorija predstavljena z binarnim stolpcem: kodirana spremenljivka je enaka 1, če buča pripada tej sorti, in 0 sicer.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Nato se `ColumnTransformer` uporabi za združitev več kodirnikov v en korak in njihovo uporabo na ustreznih stolpcih.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
Po drugi strani pa za kodiranje oznake uporabimo razred `LabelEncoder` knjižnice scikit-learn, ki je pripomoček za normalizacijo oznak, tako da vsebujejo le vrednosti med 0 in n_classes-1 (tukaj 0 in 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Ko smo kodirali značilnosti in oznako, jih lahko združimo v nov podatkovni okvir `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Kakšne so prednosti uporabe ordinalnega kodirnika za stolpec `Item Size`?

### Analizirajte odnose med spremenljivkami

Zdaj, ko smo predhodno obdelali naše podatke, lahko analiziramo odnose med značilnostmi in oznako, da dobimo idejo o tem, kako dobro bo model lahko napovedal oznako glede na značilnosti.
Najboljši način za izvedbo te vrste analize je risanje podatkov. Ponovno bomo uporabili funkcijo `catplot` knjižnice Seaborn, da vizualiziramo odnose med `Item Size`, `Variety` in `Color` v kategorni graf. Za boljše risanje podatkov bomo uporabili kodiran stolpec `Item Size` in nekodiran stolpec `Variety`.

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
![Kategorni graf vizualiziranih podatkov](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Uporabite graf 'swarm'

Ker je `Color` binarna kategorija (Bela ali Ne bela), potrebuje '[specializiran pristop](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) za vizualizacijo'. Obstajajo drugi načini za vizualizacijo odnosa te kategorije z drugimi spremenljivkami.

Spremenljivke lahko vizualizirate vzporedno z grafi knjižnice Seaborn.

1. Poskusite graf 'swarm', da prikažete porazdelitev vrednosti:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Swarm graf vizualiziranih podatkov](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Opozorilo**: zgornja koda lahko ustvari opozorilo, saj Seaborn ne uspe predstaviti takšne količine podatkovnih točk v grafu swarm. Možna rešitev je zmanjšanje velikosti označevalca z uporabo parametra 'size'. Vendar bodite pozorni, da to vpliva na berljivost grafa.

> **🧮 Pokaži mi matematiko**
>
> Logistična regresija temelji na konceptu 'maksimalne verjetnosti' z uporabo [sigmoidnih funkcij](https://wikipedia.org/wiki/Sigmoid_function). 'Sigmoidna funkcija' na grafu izgleda kot oblika črke 'S'. Vzame vrednost in jo preslika nekam med 0 in 1. Njena krivulja se imenuje tudi 'logistična krivulja'. Njena formula izgleda takole:
>
> ![logistična funkcija](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> kjer se sredinska točka sigmoidne funkcije nahaja na x-ovi točki 0, L je največja vrednost krivulje, k pa je strmina krivulje. Če je rezultat funkcije večji od 0,5, bo oznaka dodeljena razredu '1' binarne izbire. Če ne, bo razvrščena kot '0'.

## Zgradite svoj model

Gradnja modela za iskanje teh binarnih klasifikacij je presenetljivo enostavna v Scikit-learn.

[![Strojno učenje za začetnike - Logistična regresija za klasifikacijo podatkov](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "Strojno učenje za začetnike - Logistična regresija za klasifikacijo podatkov")

> 🎥 Kliknite zgornjo sliko za kratek video pregled gradnje modela linearne regresije.

1. Izberite spremenljivke, ki jih želite uporabiti v svojem klasifikacijskem modelu, in razdelite učne ter testne nize z uporabo `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Zdaj lahko trenirate svoj model z uporabo `fit()` z učnimi podatki in natisnete njegov rezultat:

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

    Oglejte si oceno svojega modela. Ni slabo, glede na to, da imate le približno 1000 vrstic podatkov:

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

## Boljše razumevanje prek matrike zmede

Medtem ko lahko dobite poročilo o oceni modela [pogoji](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) z natisom zgornjih elementov, boste morda lažje razumeli svoj model z uporabo [matrike zmede](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix), ki nam pomaga razumeti, kako model deluje.

> 🎓 '[Matrika zmede](https://wikipedia.org/wiki/Confusion_matrix)' (ali 'matrika napak') je tabela, ki izraža resnične in napačne pozitivne ter negativne rezultate vašega modela, s čimer ocenjuje natančnost napovedi.

1. Za uporabo matrike zmede pokličite `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Oglejte si matriko zmede svojega modela:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

V Scikit-learn matrike zmede vrstice (os 0) predstavljajo dejanske oznake, stolpci (os 1) pa napovedane oznake.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Kaj se dogaja tukaj? Recimo, da je naš model pozvan, da klasificira buče med dvema binarnima kategorijama, kategorijo 'bela' in kategorijo 'ne bela'.

- Če vaš model napove bučo kot ne belo in v resnici pripada kategoriji 'ne bela', to imenujemo resnični negativni rezultat, prikazan z zgornjo levo številko.
- Če vaš model napove bučo kot belo in v resnici pripada kategoriji 'ne bela', to imenujemo napačni negativni rezultat, prikazan z spodnjo levo številko.
- Če vaš model napove bučo kot ne belo in v resnici pripada kategoriji 'bela', to imenujemo napačni pozitivni rezultat, prikazan z zgornjo desno številko.
- Če vaš model napove bučo kot belo in v resnici pripada kategoriji 'bela', to imenujemo resnični pozitivni rezultat, prikazan z spodnjo desno številko.

Kot ste morda uganili, je zaželeno imeti večje število resničnih pozitivnih in resničnih negativnih rezultatov ter manjše število napačnih pozitivnih in napačnih negativnih rezultatov, kar pomeni, da model deluje bolje.
Kako se matrika zmede povezuje s natančnostjo in priklicem? Spomnite se, da je poročilo o klasifikaciji, ki je bilo natisnjeno zgoraj, pokazalo natančnost (0,85) in priklic (0,67).

Natančnost = tp / (tp + fp) = 22 / (22 + 4) = 0,8461538461538461

Priklic = tp / (tp + fn) = 22 / (22 + 11) = 0,6666666666666666

✅ V: Glede na matriko zmede, kako se je model odrezal? O: Ni slabo; obstaja kar nekaj pravih negativnih primerov, vendar tudi nekaj lažnih negativnih primerov.

Ponovno si poglejmo izraze, ki smo jih videli prej, s pomočjo matrike zmede in njenega preslikavanja TP/TN ter FP/FN:

🎓 Natančnost: TP/(TP + FP) Delež relevantnih primerov med pridobljenimi primeri (npr. katere oznake so bile dobro označene).

🎓 Priklic: TP/(TP + FN) Delež relevantnih primerov, ki so bili pridobljeni, ne glede na to, ali so bili dobro označeni ali ne.

🎓 f1-ocena: (2 * natančnost * priklic)/(natančnost + priklic) Tehtano povprečje natančnosti in priklica, pri čemer je najboljša ocena 1, najslabša pa 0.

🎓 Podpora: Število pojavitev vsake pridobljene oznake.

🎓 Točnost: (TP + TN)/(TP + TN + FP + FN) Delež oznak, ki so bile pravilno napovedane za vzorec.

🎓 Makro povprečje: Izračun neuteženega povprečja metrik za vsako oznako, ne upoštevajoč neravnovesje oznak.

🎓 Tehtano povprečje: Izračun povprečja metrik za vsako oznako, pri čemer se upošteva neravnovesje oznak z njihovo težo glede na podporo (število pravih primerov za vsako oznako).

✅ Ali lahko ugotovite, katero metriko bi morali spremljati, če želite, da vaš model zmanjša število lažnih negativnih primerov?

## Vizualizacija ROC krivulje tega modela

[![ML za začetnike - Analiza zmogljivosti logistične regresije z ROC krivuljami](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML za začetnike - Analiza zmogljivosti logistične regresije z ROC krivuljami")

> 🎥 Kliknite zgornjo sliko za kratek video pregled ROC krivulj.

Naredimo še eno vizualizacijo, da si ogledamo tako imenovano 'ROC' krivuljo:

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

S pomočjo Matplotliba narišite [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) ali ROC krivuljo modela. ROC krivulje se pogosto uporabljajo za pregled rezultatov klasifikatorja glede na njegove prave in lažne pozitivne primere. "ROC krivulje običajno prikazujejo stopnjo pravih pozitivnih primerov na Y osi in stopnjo lažnih pozitivnih primerov na X osi." Zato sta strmina krivulje in prostor med sredinsko črto ter krivuljo pomembna: želite krivuljo, ki hitro gre navzgor in čez črto. V našem primeru so na začetku prisotni lažni pozitivni primeri, nato pa se črta pravilno dvigne in gre čez.

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Na koncu uporabite Scikit-learnov [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) za izračun dejanskega 'Področja pod krivuljo' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Rezultat je `0.9749908725812341`. Ker se AUC giblje med 0 in 1, si želite visok rezultat, saj bo model, ki je 100 % pravilen v svojih napovedih, imel AUC 1; v tem primeru je model _kar dober_.

V prihodnjih lekcijah o klasifikacijah se boste naučili, kako iterirati za izboljšanje rezultatov modela. Za zdaj pa čestitke! Zaključili ste te lekcije o regresiji!

---
## 🚀Izziv

Logistična regresija skriva še veliko več! Najboljši način za učenje pa je eksperimentiranje. Poiščite podatkovni niz, ki je primeren za tovrstno analizo, in z njim zgradite model. Kaj ste se naučili? namig: poskusite [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) za zanimive podatkovne nize.

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Preberite prve strani [tega dokumenta s Stanforda](https://web.stanford.edu/~jurafsky/slp3/5.pdf) o nekaterih praktičnih uporabah logistične regresije. Razmislite o nalogah, ki so bolj primerne za eno ali drugo vrsto regresijskih nalog, ki smo jih preučevali do zdaj. Kaj bi delovalo najbolje?

## Naloga 

[Ponovno preizkusite to regresijo](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas opozarjamo, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki bi izhajale iz uporabe tega prevoda.