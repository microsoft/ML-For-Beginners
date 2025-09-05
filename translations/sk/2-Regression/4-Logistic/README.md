<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T15:16:31+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "sk"
}
-->
# Logistická regresia na predpovedanie kategórií

![Infografika: Logistická vs. lineárna regresia](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

> ### [Táto lekcia je dostupná aj v R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Úvod

V tejto poslednej lekcii o regresii, jednej zo základných _klasických_ techník strojového učenia, sa pozrieme na logistickú regresiu. Túto techniku by ste použili na objavenie vzorcov na predpovedanie binárnych kategórií. Je táto cukrovinka čokoládová alebo nie? Je táto choroba nákazlivá alebo nie? Vyberie si tento zákazník tento produkt alebo nie?

V tejto lekcii sa naučíte:

- Novú knižnicu na vizualizáciu dát
- Techniky logistickej regresie

✅ Prehĺbte si svoje znalosti o práci s týmto typom regresie v tomto [učebnom module](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Predpoklady

Po práci s dátami o tekviciach sme už dostatočne oboznámení s tým, že existuje jedna binárna kategória, s ktorou môžeme pracovať: `Farba`.

Postavme model logistickej regresie na predpovedanie toho, na základe niektorých premenných, _akú farbu bude mať daná tekvica_ (oranžová 🎃 alebo biela 👻).

> Prečo hovoríme o binárnej klasifikácii v lekcii o regresii? Len z jazykového pohodlia, pretože logistická regresia je [v skutočnosti metóda klasifikácie](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), hoci založená na lineárnom prístupe. O ďalších spôsoboch klasifikácie dát sa dozviete v nasledujúcej skupine lekcií.

## Definovanie otázky

Pre naše účely to vyjadríme ako binárnu otázku: 'Biela' alebo 'Nie biela'. V našej dátovej sade je tiež kategória 'pruhovaná', ale má málo záznamov, takže ju nebudeme používať. Aj tak zmizne, keď odstránime nulové hodnoty z dátovej sady.

> 🎃 Zaujímavý fakt: biele tekvice niekedy nazývame 'duchové' tekvice. Nie sú veľmi ľahké na vyrezávanie, takže nie sú tak populárne ako oranžové, ale vyzerajú zaujímavo! Takže by sme mohli našu otázku preformulovať ako: 'Duch' alebo 'Nie duch'. 👻

## O logistickej regresii

Logistická regresia sa líši od lineárnej regresie, ktorú ste sa naučili predtým, v niekoľkých dôležitých ohľadoch.

[![Strojové učenie pre začiatočníkov - Pochopenie logistickej regresie pre klasifikáciu](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "Strojové učenie pre začiatočníkov - Pochopenie logistickej regresie pre klasifikáciu")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o logistickej regresii.

### Binárna klasifikácia

Logistická regresia neponúka rovnaké funkcie ako lineárna regresia. Prvá ponúka predpoveď o binárnej kategórii ("biela alebo nie biela"), zatiaľ čo druhá je schopná predpovedať kontinuálne hodnoty, napríklad na základe pôvodu tekvice a času zberu, _ako veľmi sa zvýši jej cena_.

![Model klasifikácie tekvíc](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Iné typy klasifikácií

Existujú aj iné typy logistickej regresie, vrátane multinomiálnej a ordinálnej:

- **Multinomiálna**, ktorá zahŕňa viac ako jednu kategóriu - "Oranžová, Biela a Pruhovaná".
- **Ordinálna**, ktorá zahŕňa usporiadané kategórie, užitočné, ak by sme chceli usporiadať naše výsledky logicky, ako naše tekvice, ktoré sú usporiadané podľa konečného počtu veľkostí (mini, sm, med, lg, xl, xxl).

![Multinomiálna vs ordinálna regresia](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Premenné NEMUSIA korelovať

Pamätáte si, ako lineárna regresia fungovala lepšie s viac korelovanými premennými? Logistická regresia je opakom - premenné nemusia byť v súlade. To funguje pre tieto dáta, ktoré majú pomerne slabé korelácie.

### Potrebujete veľa čistých dát

Logistická regresia poskytne presnejšie výsledky, ak použijete viac dát; naša malá dátová sada nie je optimálna pre túto úlohu, takže to majte na pamäti.

[![Strojové učenie pre začiatočníkov - Analýza a príprava dát pre logistickú regresiu](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "Strojové učenie pre začiatočníkov - Analýza a príprava dát pre logistickú regresiu")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o príprave dát pre lineárnu regresiu.

✅ Premýšľajte o typoch dát, ktoré by sa hodili pre logistickú regresiu.

## Cvičenie - upravte dáta

Najprv trochu upravte dáta, odstráňte nulové hodnoty a vyberte len niektoré stĺpce:

1. Pridajte nasledujúci kód:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Vždy sa môžete pozrieť na svoj nový dataframe:

    ```python
    pumpkins.info
    ```

### Vizualizácia - kategóriálny graf

Teraz ste načítali [štartovací notebook](../../../../2-Regression/4-Logistic/notebook.ipynb) s dátami o tekviciach a upravili ho tak, aby obsahoval dátovú sadu s niekoľkými premennými vrátane `Farba`. Vizualizujme dataframe v notebooku pomocou inej knižnice: [Seaborn](https://seaborn.pydata.org/index.html), ktorá je postavená na Matplotlib, ktorý sme použili skôr.

Seaborn ponúka niekoľko zaujímavých spôsobov vizualizácie vašich dát. Napríklad môžete porovnať distribúcie dát pre každú `Variety` a `Farba` v kategóriálnom grafe.

1. Vytvorte takýto graf pomocou funkcie `catplot`, použite naše dáta o tekviciach `pumpkins` a špecifikujte farebné mapovanie pre každú kategóriu tekvíc (oranžová alebo biela):

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

    ![Mriežka vizualizovaných dát](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Pozorovaním dát môžete vidieť, ako sa dáta o farbe vzťahujú na odrodu.

    ✅ Na základe tohto kategóriálneho grafu, aké zaujímavé skúmania si dokážete predstaviť?

### Predspracovanie dát: kódovanie vlastností a štítkov
Naša dátová sada o tekviciach obsahuje textové hodnoty vo všetkých svojich stĺpcoch. Práca s kategóriálnymi dátami je intuitívna pre ľudí, ale nie pre stroje. Algoritmy strojového učenia fungujú dobre s číslami. Preto je kódovanie veľmi dôležitým krokom vo fáze predspracovania dát, pretože nám umožňuje premeniť kategóriálne dáta na číselné dáta bez straty informácií. Dobré kódovanie vedie k vytvoreniu dobrého modelu.

Pre kódovanie vlastností existujú dva hlavné typy kóderov:

1. Ordinálny kóder: hodí sa pre ordinálne premenné, ktoré sú kategóriálne premenné, kde ich dáta nasledujú logické usporiadanie, ako stĺpec `Item Size` v našej dátovej sade. Vytvára mapovanie, kde každá kategória je reprezentovaná číslom, ktoré je poradie kategórie v stĺpci.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Kategóriálny kóder: hodí sa pre nominálne premenné, ktoré sú kategóriálne premenné, kde ich dáta nenasledujú logické usporiadanie, ako všetky vlastnosti odlišné od `Item Size` v našej dátovej sade. Ide o kódovanie typu one-hot, čo znamená, že každá kategória je reprezentovaná binárnym stĺpcom: kódovaná premenná je rovná 1, ak tekvica patrí do tejto odrody, a 0 inak.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Potom sa na kombináciu viacerých kóderov do jedného kroku a ich aplikáciu na príslušné stĺpce používa `ColumnTransformer`.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
Na druhej strane, na kódovanie štítku používame triedu `LabelEncoder` zo scikit-learn, ktorá je pomocnou triedou na normalizáciu štítkov tak, aby obsahovali iba hodnoty medzi 0 a n_classes-1 (tu 0 a 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Keď sme zakódovali vlastnosti a štítok, môžeme ich zlúčiť do nového dataframe `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Aké sú výhody použitia ordinálneho kódera pre stĺpec `Item Size`?

### Analýza vzťahov medzi premennými

Teraz, keď sme predspracovali naše dáta, môžeme analyzovať vzťahy medzi vlastnosťami a štítkom, aby sme získali predstavu o tom, ako dobre bude model schopný predpovedať štítok na základe vlastností.
Najlepší spôsob, ako vykonať tento druh analýzy, je vizualizácia dát. Opäť použijeme funkciu `catplot` zo Seaborn na vizualizáciu vzťahov medzi `Item Size`, `Variety` a `Farba` v kategóriálnom grafe. Na lepšiu vizualizáciu dát použijeme zakódovaný stĺpec `Item Size` a nezakódovaný stĺpec `Variety`.

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
![Kategóriálny graf vizualizovaných dát](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Použitie swarm grafu

Keďže Farba je binárna kategória (Biela alebo Nie), potrebuje 'špecializovaný prístup [k vizualizácii](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar)'. Existujú aj iné spôsoby vizualizácie vzťahu tejto kategórie s inými premennými.

Premenné môžete vizualizovať vedľa seba pomocou grafov Seaborn.

1. Skúste 'swarm' graf na zobrazenie distribúcie hodnôt:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Swarm graf vizualizovaných dát](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Pozor**: vyššie uvedený kód môže generovať varovanie, pretože Seaborn nedokáže reprezentovať také množstvo dátových bodov v swarm grafe. Možným riešením je zmenšenie veľkosti značky pomocou parametra 'size'. Buďte však opatrní, pretože to ovplyvňuje čitateľnosť grafu.

> **🧮 Ukážte mi matematiku**
>
> Logistická regresia sa opiera o koncept 'maximálnej pravdepodobnosti' pomocou [sigmoidných funkcií](https://wikipedia.org/wiki/Sigmoid_function). 'Sigmoidná funkcia' na grafe vyzerá ako tvar 'S'. Berie hodnotu a mapuje ju na niečo medzi 0 a 1. Jej krivka sa tiež nazýva 'logistická krivka'. Jej vzorec vyzerá takto:
>
> ![logistická funkcia](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> kde stred sigmoidnej funkcie sa nachádza na 0 bode x, L je maximálna hodnota krivky a k je strmosť krivky. Ak je výsledok funkcie viac ako 0.5, daný štítok bude priradený do triedy '1' binárnej voľby. Ak nie, bude klasifikovaný ako '0'.

## Vytvorte svoj model

Vytvorenie modelu na nájdenie týchto binárnych klasifikácií je prekvapivo jednoduché v Scikit-learn.

[![Strojové učenie pre začiatočníkov - Logistická regresia pre klasifikáciu dát](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "Strojové učenie pre začiatočníkov - Logistická regresia pre klasifikáciu dát")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o vytváraní modelu lineárnej regresie.

1. Vyberte premenné, ktoré chcete použiť vo svojom klasifikačnom modeli, a rozdeľte tréningové a testovacie sady pomocou `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Teraz môžete trénovať svoj model, zavolaním `fit()` s vašimi tréningovými dátami, a vytlačiť jeho výsledok:

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

    Pozrite sa na skóre vášho modelu. Nie je to zlé, vzhľadom na to, že máte len asi 1000 riadkov dát:

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

## Lepšie pochopenie pomocou matice zmätku

Zatiaľ čo môžete získať správu o skóre [termíny](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) vytlačením vyššie uvedených položiek, môžete svoj model lepšie pochopiť pomocou [matice zmätku](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix), ktorá nám pomáha pochopiť, ako model funguje.

> 🎓 '[Matica zmätku](https://wikipedia.org/wiki/Confusion_matrix)' (alebo 'matica chýb') je tabuľka, ktorá vyjadruje skutočné vs. falošné pozitíva a negatíva vášho modelu, čím hodnotí presnosť predpovedí.

1. Na použitie matice zmätku zavolajte `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Pozrite sa na maticu zmätku vášho modelu:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

V Scikit-learn, riadky (os 0) sú skutočné štítky a stĺpce (os 1) sú predpovedané štítky.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Čo sa tu deje? Povedzme, že náš model je požiadaný klasifikovať tekvice medzi dvoma binárnymi kategóriami, kategóriou 'biela' a kategóriou 'nie biela'.

- Ak váš model predpovedá tekvicu ako nie bielu a v skutočnosti patrí do kategórie 'nie biela', nazývame to pravý negatívny, zobrazený horným ľavým číslom.
- Ak váš model predpovedá tekvicu ako bielu a v skutočnosti patrí do kategórie 'nie biela', nazývame to falošný negatívny, zobrazený dolným ľavým číslom.
- Ak váš model predpovedá tekvicu ako nie bielu a v skutočnosti patrí do kategórie 'biela', nazývame to falošný pozitívny, zobrazený horným pravým číslom.
- Ak váš model predpovedá tekvicu ako bielu a v skutočnosti patrí do kategórie 'biela', nazývame to pravý pozitívny, zobrazený dolným pravým číslom.

Ako ste mohli uhádnuť, je preferované mať väčší počet pravých pozitívnych a pravých negatívnych a nižší počet falošných pozitívnych a falošných negatívnych, čo naznačuje, že model funguje lepšie.
Ako súvisí matica zámien s presnosťou a odvolaním? Pamätajte, že klasifikačná správa uvedená vyššie ukázala presnosť (0,85) a odvolanie (0,67).

Presnosť = tp / (tp + fp) = 22 / (22 + 4) = 0,8461538461538461

Odvolanie = tp / (tp + fn) = 22 / (22 + 11) = 0,6666666666666666

✅ Otázka: Podľa matice zámien, ako si model viedol? Odpoveď: Nie zle; je tu dobrý počet správne negatívnych, ale aj niekoľko nesprávne negatívnych.

Poďme si znova prejsť pojmy, ktoré sme videli skôr, s pomocou mapovania TP/TN a FP/FN v matici zámien:

🎓 Presnosť: TP/(TP + FP) Podiel relevantných prípadov medzi získanými prípadmi (napr. ktoré štítky boli správne označené)

🎓 Odvolanie: TP/(TP + FN) Podiel relevantných prípadov, ktoré boli získané, či už správne označené alebo nie

🎓 f1-skóre: (2 * presnosť * odvolanie)/(presnosť + odvolanie) Vážený priemer presnosti a odvolania, pričom najlepšie je 1 a najhoršie 0

🎓 Podpora: Počet výskytov každého získaného štítku

🎓 Presnosť: (TP + TN)/(TP + TN + FP + FN) Percento štítkov predpovedaných správne pre vzorku.

🎓 Makro priemer: Výpočet nevyváženého priemeru metrík pre každý štítok, bez ohľadu na nerovnováhu štítkov.

🎓 Vážený priemer: Výpočet priemeru metrík pre každý štítok, pričom sa berie do úvahy nerovnováha štítkov vážením podľa ich podpory (počet skutočných prípadov pre každý štítok).

✅ Dokážete si predstaviť, ktorú metriku by ste mali sledovať, ak chcete, aby váš model znížil počet nesprávne negatívnych?

## Vizualizácia ROC krivky tohto modelu

[![ML pre začiatočníkov - Analýza výkonu logistickej regresie pomocou ROC kriviek](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML pre začiatočníkov - Analýza výkonu logistickej regresie pomocou ROC kriviek")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad ROC kriviek

Poďme urobiť ešte jednu vizualizáciu, aby sme videli tzv. 'ROC' krivku:

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

Pomocou Matplotlibu vykreslite [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) alebo ROC modelu. ROC krivky sa často používajú na zobrazenie výstupu klasifikátora z hľadiska jeho správne vs. nesprávne pozitívnych. "ROC krivky zvyčajne zobrazujú mieru správne pozitívnych na osi Y a mieru nesprávne pozitívnych na osi X." Preto záleží na strmosti krivky a priestore medzi stredovou čiarou a krivkou: chcete krivku, ktorá rýchlo stúpa a prechádza nad čiaru. V našom prípade sú na začiatku nesprávne pozitívne, a potom krivka správne stúpa a prechádza nad čiaru:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Nakoniec použite Scikit-learn [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) na výpočet skutočnej 'Plochy pod krivkou' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Výsledok je `0.9749908725812341`. Keďže AUC sa pohybuje od 0 do 1, chcete vysoké skóre, pretože model, ktorý je 100% správny vo svojich predpovediach, bude mať AUC 1; v tomto prípade je model _celkom dobrý_. 

V budúcich lekciách o klasifikáciách sa naučíte, ako iterovať na zlepšenie skóre vášho modelu. Ale zatiaľ gratulujeme! Dokončili ste tieto lekcie o regresii!

---
## 🚀Výzva

Logistická regresia má oveľa viac, čo sa dá preskúmať! Ale najlepší spôsob, ako sa učiť, je experimentovať. Nájdite dataset, ktorý sa hodí na tento typ analýzy, a vytvorte s ním model. Čo ste sa naučili? tip: skúste [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) pre zaujímavé datasety.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Prečítajte si prvé stránky [tohto dokumentu zo Stanfordu](https://web.stanford.edu/~jurafsky/slp3/5.pdf) o niektorých praktických využitiach logistickej regresie. Premýšľajte o úlohách, ktoré sú lepšie vhodné pre jeden alebo druhý typ regresných úloh, ktoré sme doteraz študovali. Čo by fungovalo najlepšie?

## Zadanie 

[Opakovanie tejto regresie](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.