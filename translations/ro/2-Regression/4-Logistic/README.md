<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T15:18:34+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "ro"
}
-->
# Regresie logistică pentru a prezice categorii

![Infografic despre regresia logistică vs. regresia liniară](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

> ### [Această lecție este disponibilă și în R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Introducere

În această ultimă lecție despre regresie, una dintre tehnicile clasice de bază ale ML, vom analiza regresia logistică. Această tehnică este utilizată pentru a descoperi modele care prezic categorii binare. Este această bomboană ciocolată sau nu? Este această boală contagioasă sau nu? Va alege acest client produsul sau nu?

În această lecție, vei învăța:

- O nouă bibliotecă pentru vizualizarea datelor
- Tehnici pentru regresia logistică

✅ Aprofundează înțelegerea lucrului cu acest tip de regresie în acest [modul de învățare](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Prerechizite

După ce am lucrat cu datele despre dovleci, suntem suficient de familiarizați cu ele pentru a realiza că există o categorie binară cu care putem lucra: `Color`.

Să construim un model de regresie logistică pentru a prezice, pe baza unor variabile, _ce culoare este probabil să aibă un dovleac_ (portocaliu 🎃 sau alb 👻).

> De ce discutăm despre clasificarea binară într-o lecție despre regresie? Doar din comoditate lingvistică, deoarece regresia logistică este [de fapt o metodă de clasificare](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), deși bazată pe regresie liniară. Învață despre alte metode de clasificare a datelor în următorul grup de lecții.

## Definirea întrebării

Pentru scopurile noastre, vom exprima aceasta ca o binară: 'Alb' sau 'Nu Alb'. Există și o categorie 'dungat' în setul nostru de date, dar există puține instanțe ale acesteia, așa că nu o vom folosi. Oricum dispare odată ce eliminăm valorile nule din setul de date.

> 🎃 Fapt amuzant: uneori numim dovlecii albi 'dovleci fantomă'. Nu sunt foarte ușor de sculptat, așa că nu sunt la fel de populari ca cei portocalii, dar arată interesant! Așadar, am putea reformula întrebarea noastră astfel: 'Fantoma' sau 'Nu Fantoma'. 👻

## Despre regresia logistică

Regresia logistică diferă de regresia liniară, pe care ai învățat-o anterior, în câteva moduri importante.

[![ML pentru începători - Înțelegerea regresiei logistice pentru clasificarea în învățarea automată](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML pentru începători - Înțelegerea regresiei logistice pentru clasificarea în învățarea automată")

> 🎥 Fă clic pe imaginea de mai sus pentru un scurt videoclip despre regresia logistică.

### Clasificare binară

Regresia logistică nu oferă aceleași caracteristici ca regresia liniară. Prima oferă o predicție despre o categorie binară ("alb sau nu alb"), în timp ce cea de-a doua este capabilă să prezică valori continue, de exemplu, având în vedere originea unui dovleac și momentul recoltării, _cât de mult va crește prețul său_.

![Model de clasificare a dovlecilor](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Alte clasificări

Există alte tipuri de regresie logistică, inclusiv multinomială și ordonată:

- **Multinomială**, care implică mai multe categorii - "Portocaliu, Alb și Dungat".
- **Ordonată**, care implică categorii ordonate, utilă dacă dorim să ordonăm rezultatele logic, cum ar fi dovlecii noștri care sunt ordonați după un număr finit de dimensiuni (mini, mic, mediu, mare, XL, XXL).

![Regresie multinomială vs ordonată](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Variabilele NU trebuie să fie corelate

Îți amintești cum regresia liniară funcționa mai bine cu variabile mai corelate? Regresia logistică este opusul - variabilele nu trebuie să fie aliniate. Acest lucru funcționează pentru aceste date care au corelații destul de slabe.

### Ai nevoie de multe date curate

Regresia logistică va oferi rezultate mai precise dacă folosești mai multe date; setul nostru mic de date nu este optim pentru această sarcină, așa că ține cont de acest lucru.

[![ML pentru începători - Analiza și pregătirea datelor pentru regresia logistică](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML pentru începători - Analiza și pregătirea datelor pentru regresia logistică")

✅ Gândește-te la tipurile de date care s-ar potrivi bine regresiei logistice.

## Exercițiu - curățarea datelor

Mai întâi, curăță puțin datele, eliminând valorile nule și selectând doar câteva coloane:

1. Adaugă următorul cod:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Poți oricând să arunci o privire asupra noului tău dataframe:

    ```python
    pumpkins.info
    ```

### Vizualizare - grafic categorial

Până acum ai încărcat [notebook-ul de început](../../../../2-Regression/4-Logistic/notebook.ipynb) cu datele despre dovleci din nou și le-ai curățat astfel încât să păstrezi un set de date care conține câteva variabile, inclusiv `Color`. Să vizualizăm dataframe-ul în notebook folosind o bibliotecă diferită: [Seaborn](https://seaborn.pydata.org/index.html), care este construită pe Matplotlib pe care l-am folosit anterior.

Seaborn oferă câteva modalități interesante de a vizualiza datele tale. De exemplu, poți compara distribuțiile datelor pentru fiecare `Variety` și `Color` într-un grafic categorial.

1. Creează un astfel de grafic folosind funcția `catplot`, utilizând datele noastre despre dovleci `pumpkins` și specificând o mapare de culori pentru fiecare categorie de dovleci (portocaliu sau alb):

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

    ![O grilă de date vizualizate](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Observând datele, poți vedea cum datele despre culoare se raportează la varietate.

    ✅ Având acest grafic categorial, ce explorări interesante poți imagina?

### Pre-procesarea datelor: codificarea caracteristicilor și etichetelor

Setul nostru de date despre dovleci conține valori de tip string pentru toate coloanele sale. Lucrul cu date categoriale este intuitiv pentru oameni, dar nu pentru mașini. Algoritmii de învățare automată funcționează bine cu numere. De aceea, codificarea este un pas foarte important în faza de pre-procesare a datelor, deoarece ne permite să transformăm datele categoriale în date numerice, fără a pierde informații. O codificare bună duce la construirea unui model bun.

Pentru codificarea caracteristicilor există două tipuri principale de codificatori:

1. Codificator ordinal: se potrivește bine pentru variabile ordinale, care sunt variabile categoriale unde datele lor urmează o ordine logică, cum ar fi coloana `Item Size` din setul nostru de date. Creează o mapare astfel încât fiecare categorie să fie reprezentată de un număr, care este ordinea categoriei în coloană.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Codificator categorial: se potrivește bine pentru variabile nominale, care sunt variabile categoriale unde datele lor nu urmează o ordine logică, cum ar fi toate caracteristicile diferite de `Item Size` din setul nostru de date. Este o codificare one-hot, ceea ce înseamnă că fiecare categorie este reprezentată de o coloană binară: variabila codificată este egală cu 1 dacă dovleacul aparține acelei varietăți și 0 altfel.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

Apoi, `ColumnTransformer` este utilizat pentru a combina mai mulți codificatori într-un singur pas și a-i aplica coloanelor corespunzătoare.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

Pe de altă parte, pentru a codifica eticheta, folosim clasa `LabelEncoder` din scikit-learn, care este o clasă utilitară pentru a normaliza etichetele astfel încât să conțină doar valori între 0 și n_classes-1 (aici, 0 și 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

După ce am codificat caracteristicile și eticheta, le putem îmbina într-un nou dataframe `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ Care sunt avantajele utilizării unui codificator ordinal pentru coloana `Item Size`?

### Analiza relațiilor dintre variabile

Acum că am pre-procesat datele, putem analiza relațiile dintre caracteristici și etichetă pentru a înțelege cât de bine va putea modelul să prezică eticheta pe baza caracteristicilor. 

Cea mai bună modalitate de a efectua acest tip de analiză este să plotăm datele. Vom folosi din nou funcția `catplot` din Seaborn pentru a vizualiza relațiile dintre `Item Size`, `Variety` și `Color` într-un grafic categorial. Pentru a plota mai bine datele, vom folosi coloana codificată `Item Size` și coloana necodificată `Variety`.

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

![Un grafic categorial al datelor vizualizate](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Utilizarea unui grafic swarm

Deoarece `Color` este o categorie binară (Alb sau Nu), necesită 'o [abordare specializată](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) pentru vizualizare'. Există alte modalități de a vizualiza relația acestei categorii cu alte variabile.

Poți vizualiza variabilele una lângă alta cu grafice Seaborn.

1. Încearcă un grafic 'swarm' pentru a arăta distribuția valorilor:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Un grafic swarm al datelor vizualizate](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Atenție**: codul de mai sus poate genera un avertisment, deoarece Seaborn nu reușește să reprezinte o cantitate atât de mare de puncte de date într-un grafic swarm. O soluție posibilă este reducerea dimensiunii markerului, utilizând parametrul 'size'. Totuși, fii conștient că acest lucru afectează lizibilitatea graficului.

> **🧮 Arată-mi matematica**
>
> Regresia logistică se bazează pe conceptul de 'maximum likelihood' folosind [funcții sigmoid](https://wikipedia.org/wiki/Sigmoid_function). O 'Funcție Sigmoid' pe un grafic arată ca o formă de 'S'. Aceasta ia o valoare și o mapează undeva între 0 și 1. Curba sa este numită și 'curbă logistică'. Formula sa arată astfel:
>
> ![funcția logistică](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> unde punctul de mijloc al sigmoidului se află la punctul 0 al lui x, L este valoarea maximă a curbei, iar k este panta curbei. Dacă rezultatul funcției este mai mare de 0.5, eticheta în cauză va fi atribuită clasei '1' din alegerea binară. Dacă nu, va fi clasificată ca '0'.

## Construiește modelul tău

Construirea unui model pentru a găsi aceste clasificări binare este surprinzător de simplă în Scikit-learn.

[![ML pentru începători - Regresia logistică pentru clasificarea datelor](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML pentru începători - Regresia logistică pentru clasificarea datelor")

> 🎥 Fă clic pe imaginea de mai sus pentru un scurt videoclip despre construirea unui model de regresie liniară.

1. Selectează variabilele pe care vrei să le folosești în modelul tău de clasificare și împarte seturile de antrenament și testare apelând `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Acum poți antrena modelul tău, apelând `fit()` cu datele de antrenament și afișând rezultatul:

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

    Aruncă o privire asupra scorului modelului tău. Nu este rău, având în vedere că ai doar aproximativ 1000 de rânduri de date:

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

## Înțelegere mai bună printr-o matrice de confuzie

Deși poți obține un raport de scor [termeni](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) prin imprimarea elementelor de mai sus, s-ar putea să înțelegi modelul mai ușor utilizând o [matrice de confuzie](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) pentru a ne ajuta să înțelegem cum performează modelul.

> 🎓 O '[matrice de confuzie](https://wikipedia.org/wiki/Confusion_matrix)' (sau 'matrice de eroare') este un tabel care exprimă pozitivele și negativele adevărate vs. false ale modelului tău, evaluând astfel acuratețea predicțiilor.

1. Pentru a utiliza o matrice de confuzie, apelează `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Aruncă o privire asupra matricei de confuzie a modelului tău:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

În Scikit-learn, rândurile (axa 0) sunt etichetele reale, iar coloanele (axa 1) sunt etichetele prezise.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Ce se întâmplă aici? Să presupunem că modelul nostru este solicitat să clasifice dovlecii între două categorii binare, categoria 'alb' și categoria 'nu alb'.

- Dacă modelul tău prezice un dovleac ca fiind nu alb și acesta aparține categoriei 'nu alb' în realitate, îl numim negativ adevărat, indicat de numărul din stânga sus.
- Dacă modelul tău prezice un dovleac ca fiind alb și acesta aparține categoriei 'nu alb' în realitate, îl numim negativ fals, indicat de numărul din stânga jos.
- Dacă modelul tău prezice un dovleac ca fiind nu alb și acesta aparține categoriei 'alb' în realitate, îl numim pozitiv fals, indicat de numărul din dreapta sus.
- Dacă modelul tău prezice un dovleac ca fiind alb și acesta aparține categoriei 'alb' în realitate, îl numim pozitiv adevărat, indicat de numărul din dreapta jos.

După cum probabil ai ghicit, este preferabil să ai un număr mai mare de pozitive adevărate și negative adevărate și un număr mai mic de pozitive false și negative false, ceea ce implică faptul că modelul performează mai bine.
Cum se leagă matricea de confuzie de precizie și recall? Ține minte, raportul de clasificare afișat mai sus a arătat o precizie de 0.85 și un recall de 0.67.

Precizie = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Recall = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Î: Conform matricei de confuzie, cum s-a descurcat modelul? R: Nu rău; există un număr bun de true negatives, dar și câteva false negatives.

Să revenim la termenii pe care i-am văzut mai devreme, cu ajutorul mapării TP/TN și FP/FN din matricea de confuzie:

🎓 Precizie: TP/(TP + FP) Proporția instanțelor relevante dintre cele recuperate (de exemplu, etichetele care au fost bine etichetate)

🎓 Recall: TP/(TP + FN) Proporția instanțelor relevante care au fost recuperate, indiferent dacă au fost bine etichetate sau nu

🎓 f1-score: (2 * precizie * recall)/(precizie + recall) O medie ponderată a preciziei și recall-ului, cu cel mai bun scor fiind 1 și cel mai slab fiind 0

🎓 Support: Numărul de apariții ale fiecărei etichete recuperate

🎓 Acuratețe: (TP + TN)/(TP + TN + FP + FN) Procentul de etichete prezise corect pentru un eșantion.

🎓 Macro Avg: Calculul mediei neponderate a metricilor pentru fiecare etichetă, fără a ține cont de dezechilibrul etichetelor.

🎓 Weighted Avg: Calculul mediei metricilor pentru fiecare etichetă, ținând cont de dezechilibrul etichetelor prin ponderarea lor în funcție de support (numărul de instanțe reale pentru fiecare etichetă).

✅ Te poți gândi la ce metrică ar trebui să fii atent dacă vrei ca modelul tău să reducă numărul de false negatives?

## Vizualizează curba ROC a acestui model

[![ML pentru începători - Analiza performanței regresiei logistice cu curbe ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML pentru începători - Analiza performanței regresiei logistice cu curbe ROC")

> 🎥 Fă clic pe imaginea de mai sus pentru o prezentare video scurtă despre curbele ROC

Să facem o vizualizare suplimentară pentru a vedea așa-numita 'curbă ROC':

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

Folosind Matplotlib, plotează [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) sau ROC-ul modelului. Curbele ROC sunt adesea utilizate pentru a obține o imagine de ansamblu asupra ieșirii unui clasificator în termeni de true positives vs. false positives. "Curbele ROC prezintă de obicei rata de true positives pe axa Y și rata de false positives pe axa X." Astfel, abruptul curbei și spațiul dintre linia de mijloc și curbă contează: vrei o curbă care urcă rapid și depășește linia. În cazul nostru, există false positives la început, iar apoi linia urcă și depășește corect:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

În final, folosește API-ul [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) din Scikit-learn pentru a calcula efectiv 'Aria Sub Curbă' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Rezultatul este `0.9749908725812341`. Având în vedere că AUC variază între 0 și 1, vrei un scor mare, deoarece un model care este 100% corect în predicțiile sale va avea un AUC de 1; în acest caz, modelul este _destul de bun_.

În lecțiile viitoare despre clasificări, vei învăța cum să iterezi pentru a îmbunătăți scorurile modelului tău. Dar pentru moment, felicitări! Ai finalizat aceste lecții despre regresie!

---
## 🚀Provocare

Există mult mai multe de explorat în ceea ce privește regresia logistică! Dar cel mai bun mod de a învăța este să experimentezi. Găsește un set de date care se pretează acestui tip de analiză și construiește un model cu el. Ce ai învățat? sugestie: încearcă [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) pentru seturi de date interesante.

## [Quiz după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Studiu individual

Citește primele câteva pagini din [acest articol de la Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) despre câteva utilizări practice ale regresiei logistice. Gândește-te la sarcini care sunt mai potrivite pentru unul sau altul dintre tipurile de regresie pe care le-am studiat până acum. Ce ar funcționa cel mai bine?

## Temă

[Reia această regresie](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.