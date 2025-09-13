<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T16:26:58+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "ro"
}
-->
# Introducere în clasificare

În aceste patru lecții, vei explora un aspect fundamental al învățării automate clasice - _clasificarea_. Vom parcurge utilizarea diferitelor algoritmi de clasificare cu un set de date despre toate bucătăriile minunate din Asia și India. Sper că ți-e foame!

![doar un praf!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Sărbătorește bucătăriile pan-asiatice în aceste lecții! Imagine de [Jen Looper](https://twitter.com/jenlooper)

Clasificarea este o formă de [învățare supravegheată](https://wikipedia.org/wiki/Supervised_learning) care are multe în comun cu tehnicile de regresie. Dacă învățarea automată se referă la prezicerea valorilor sau denumirilor unor lucruri folosind seturi de date, atunci clasificarea se împarte, în general, în două grupuri: _clasificare binară_ și _clasificare multiclasă_.

[![Introducere în clasificare](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introducere în clasificare")

> 🎥 Fă clic pe imaginea de mai sus pentru un videoclip: John Guttag de la MIT introduce clasificarea

Amintește-ți:

- **Regresia liniară** te-a ajutat să prezici relațiile dintre variabile și să faci predicții precise despre unde ar putea să se încadreze un nou punct de date în raport cu acea linie. De exemplu, ai putea prezice _ce preț ar avea un dovleac în septembrie vs. decembrie_.
- **Regresia logistică** te-a ajutat să descoperi "categorii binare": la acest punct de preț, _este acest dovleac portocaliu sau nu-portocaliu_?

Clasificarea folosește diferiți algoritmi pentru a determina alte modalități de a stabili eticheta sau clasa unui punct de date. Hai să lucrăm cu acest set de date despre bucătării pentru a vedea dacă, observând un grup de ingrediente, putem determina originea sa culinară.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

> ### [Această lecție este disponibilă și în R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introducere

Clasificarea este una dintre activitățile fundamentale ale cercetătorului în învățare automată și ale specialistului în știința datelor. De la clasificarea de bază a unei valori binare ("este acest email spam sau nu?"), la clasificarea și segmentarea complexă a imaginilor folosind viziunea computerizată, este întotdeauna util să poți sorta datele în clase și să pui întrebări despre ele.

Pentru a exprima procesul într-un mod mai științific, metoda ta de clasificare creează un model predictiv care îți permite să mapezi relația dintre variabilele de intrare și variabilele de ieșire.

![clasificare binară vs. multiclasă](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Probleme binare vs. multiclasă pentru algoritmii de clasificare. Infografic de [Jen Looper](https://twitter.com/jenlooper)

Înainte de a începe procesul de curățare a datelor, vizualizarea lor și pregătirea pentru sarcinile de învățare automată, să învățăm puțin despre diferitele moduri în care învățarea automată poate fi utilizată pentru a clasifica datele.

Derivată din [statistică](https://wikipedia.org/wiki/Statistical_classification), clasificarea folosind învățarea automată clasică utilizează caracteristici, cum ar fi `smoker`, `weight` și `age`, pentru a determina _probabilitatea de a dezvolta boala X_. Ca o tehnică de învățare supravegheată similară cu exercițiile de regresie pe care le-ai realizat anterior, datele tale sunt etichetate, iar algoritmii de învățare automată folosesc aceste etichete pentru a clasifica și prezice clasele (sau 'caracteristicile') unui set de date și pentru a le atribui unui grup sau unui rezultat.

✅ Gândește-te un moment la un set de date despre bucătării. Ce ar putea răspunde un model multiclasă? Ce ar putea răspunde un model binar? Ce-ar fi dacă ai vrea să determini dacă o anumită bucătărie este probabil să folosească schinduf? Sau dacă ai vrea să vezi dacă, având un cadou constând într-o pungă de cumpărături plină cu anason stelat, anghinare, conopidă și hrean, ai putea crea un fel de mâncare tipic indian?

[![Coșuri misterioase nebune](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Coșuri misterioase nebune")

> 🎥 Fă clic pe imaginea de mai sus pentru un videoclip. Întregul concept al emisiunii 'Chopped' este 'coșul misterios', unde bucătarii trebuie să facă un fel de mâncare dintr-o alegere aleatorie de ingrediente. Cu siguranță un model de învățare automată ar fi fost de ajutor!

## Salut, 'clasificator'

Întrebarea pe care vrem să o adresăm acestui set de date despre bucătării este, de fapt, o întrebare de **clasificare multiclasă**, deoarece avem mai multe bucătării naționale potențiale cu care să lucrăm. Având un lot de ingrediente, în care dintre aceste multe clase se va încadra datele?

Scikit-learn oferă mai mulți algoritmi diferiți pentru a clasifica datele, în funcție de tipul de problemă pe care vrei să o rezolvi. În următoarele două lecții, vei învăța despre câțiva dintre acești algoritmi.

## Exercițiu - curăță și echilibrează datele

Prima sarcină, înainte de a începe acest proiect, este să cureți și să **echilibrezi** datele pentru a obține rezultate mai bune. Începe cu fișierul gol _notebook.ipynb_ din rădăcina acestui folder.

Primul lucru de instalat este [imblearn](https://imbalanced-learn.org/stable/). Acesta este un pachet Scikit-learn care îți va permite să echilibrezi mai bine datele (vei învăța mai multe despre această sarcină în curând).

1. Pentru a instala `imblearn`, rulează `pip install`, astfel:

    ```python
    pip install imblearn
    ```

1. Importează pachetele necesare pentru a importa datele și a le vizualiza, de asemenea importă `SMOTE` din `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Acum ești pregătit să imporți datele.

1. Următoarea sarcină va fi să imporți datele:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Utilizarea `read_csv()` va citi conținutul fișierului csv _cusines.csv_ și îl va plasa în variabila `df`.

1. Verifică forma datelor:

    ```python
    df.head()
    ```

   Primele cinci rânduri arată astfel:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Obține informații despre aceste date apelând `info()`:

    ```python
    df.info()
    ```

    Rezultatul tău seamănă cu:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Exercițiu - învățarea despre bucătării

Acum munca începe să devină mai interesantă. Hai să descoperim distribuția datelor, pe bucătărie.

1. Plotează datele sub formă de bare apelând `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribuția datelor despre bucătării](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Există un număr finit de bucătării, dar distribuția datelor este inegală. Poți remedia asta! Înainte de a face acest lucru, explorează puțin mai mult.

1. Află cât de multe date sunt disponibile per bucătărie și afișează-le:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    Rezultatul arată astfel:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Descoperirea ingredientelor

Acum poți să te adâncești mai mult în date și să afli care sunt ingredientele tipice pentru fiecare bucătărie. Ar trebui să elimini datele recurente care creează confuzie între bucătării, așa că hai să învățăm despre această problemă.

1. Creează o funcție `create_ingredient()` în Python pentru a crea un dataframe de ingrediente. Această funcție va începe prin eliminarea unei coloane inutile și va sorta ingredientele după numărul lor:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Acum poți folosi această funcție pentru a obține o idee despre primele zece cele mai populare ingrediente per bucătărie.

1. Apelează `create_ingredient()` și plotează rezultatul apelând `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Fă același lucru pentru datele despre bucătăria japoneză:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japoneză](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Acum pentru ingredientele chinezești:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chineză](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plotează ingredientele indiene:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. În cele din urmă, plotează ingredientele coreene:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![coreean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Acum, elimină cele mai comune ingrediente care creează confuzie între bucătării distincte, apelând `drop()`:

   Toată lumea iubește orezul, usturoiul și ghimbirul!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Echilibrarea setului de date

Acum că ai curățat datele, folosește [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Tehnica de Supraîncărcare Sintetică a Minorităților" - pentru a le echilibra.

1. Apelează `fit_resample()`, această strategie generează noi mostre prin interpolare.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Prin echilibrarea datelor, vei obține rezultate mai bune atunci când le clasifici. Gândește-te la o clasificare binară. Dacă majoritatea datelor tale aparțin unei clase, un model de învățare automată va prezice acea clasă mai frecvent, doar pentru că există mai multe date pentru ea. Echilibrarea datelor elimină acest dezechilibru.

1. Acum poți verifica numărul de etichete per ingredient:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Rezultatul tău arată astfel:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Datele sunt acum curate, echilibrate și foarte delicioase!

1. Ultimul pas este să salvezi datele echilibrate, inclusiv etichetele și caracteristicile, într-un nou dataframe care poate fi exportat într-un fișier:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Poți arunca o ultimă privire asupra datelor folosind `transformed_df.head()` și `transformed_df.info()`. Salvează o copie a acestor date pentru utilizare în lecțiile viitoare:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Acest CSV proaspăt poate fi găsit acum în folderul de date rădăcină.

---

## 🚀Provocare

Acest curriculum conține mai multe seturi de date interesante. Răsfoiește folderele `data` și vezi dacă vreunul conține seturi de date care ar fi potrivite pentru clasificare binară sau multiclasă. Ce întrebări ai pune acestui set de date?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Explorează API-ul SMOTE. Pentru ce cazuri de utilizare este cel mai potrivit? Ce probleme rezolvă?

## Temă

[Explorează metodele de clasificare](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa maternă ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.