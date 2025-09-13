<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T15:14:22+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "sw"
}
-->
# Logistic regression kutabiri makundi

![Picha ya infographic ya Logistic vs. linear regression](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somo hili linapatikana kwa R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Utangulizi

Katika somo hili la mwisho kuhusu Regression, mojawapo ya mbinu za msingi za _classic_ ML, tutachunguza Logistic Regression. Ungetumia mbinu hii kugundua mifumo ya kutabiri makundi mawili. Je, pipi hii ni chokoleti au siyo? Je, ugonjwa huu ni wa kuambukiza au siyo? Je, mteja huyu atachagua bidhaa hii au siyo?

Katika somo hili, utajifunza:

- Maktaba mpya ya kuonyesha data
- Mbinu za logistic regression

✅ Kuimarisha uelewa wako wa kufanya kazi na aina hii ya regression katika [moduli ya kujifunza](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Mahitaji ya awali

Baada ya kufanya kazi na data ya malenge, sasa tunafahamu vya kutosha kutambua kwamba kuna kundi moja la binary ambalo tunaweza kufanya kazi nalo: `Color`.

Hebu tujenge mfano wa logistic regression kutabiri, kwa kuzingatia baadhi ya vigezo, _rangi ya malenge fulani itakuwa nini_ (machungwa 🎃 au nyeupe 👻).

> Kwa nini tunazungumzia binary classification katika somo kuhusu regression? Ni kwa urahisi wa lugha tu, kwani logistic regression ni [kweli mbinu ya classification](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), ingawa ni ya msingi wa linear. Jifunze kuhusu njia nyingine za kuainisha data katika kundi la somo linalofuata.

## Eleza swali

Kwa madhumuni yetu, tutalieleza kama binary: 'Nyeupe' au 'Siyo Nyeupe'. Pia kuna kundi la 'striped' katika dataset yetu lakini kuna matukio machache ya kundi hilo, kwa hivyo hatutalitumia. Linatoweka mara tu tunapotoa thamani za null kutoka dataset, hata hivyo.

> 🎃 Ukweli wa kufurahisha, mara nyingine tunaita malenge nyeupe 'malenge ya roho'. Hayafai sana kwa kuchonga, kwa hivyo hayajapendwa kama yale ya machungwa lakini yanaonekana ya kuvutia! Kwa hivyo tunaweza pia kuunda upya swali letu kama: 'Roho' au 'Siyo Roho'. 👻

## Kuhusu logistic regression

Logistic regression inatofautiana na linear regression, ambayo ulijifunza hapo awali, kwa njia kadhaa muhimu.

[![ML kwa wanaoanza - Kuelewa Logistic Regression kwa Machine Learning Classification](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML kwa wanaoanza - Kuelewa Logistic Regression kwa Machine Learning Classification")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa video kuhusu logistic regression.

### Binary classification

Logistic regression haitoi vipengele sawa na linear regression. Ya kwanza inatoa utabiri kuhusu kundi la binary ("nyeupe au siyo nyeupe") ilhali ya pili ina uwezo wa kutabiri thamani zinazoendelea, kwa mfano kwa kuzingatia asili ya malenge na wakati wa kuvuna, _bei yake itaongezeka kiasi gani_.

![Mfano wa uainishaji wa malenge](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infographic na [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Uainishaji mwingine

Kuna aina nyingine za logistic regression, ikiwa ni pamoja na multinomial na ordinal:

- **Multinomial**, ambayo inahusisha kuwa na zaidi ya kundi moja - "Machungwa, Nyeupe, na Striped".
- **Ordinal**, ambayo inahusisha makundi yaliyo na mpangilio, yanafaa ikiwa tungependa kupanga matokeo yetu kwa mantiki, kama malenge yetu ambayo yamepangwa kwa idadi finyu ya ukubwa (mini, sm, med, lg, xl, xxl).

![Multinomial vs ordinal regression](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Vigezo HAVIHITAJI kuhusiana

Kumbuka jinsi linear regression ilifanya kazi vizuri zaidi na vigezo vilivyohusiana? Logistic regression ni kinyume - vigezo havihitaji kuhusiana. Hii inafaa kwa data hii ambayo ina uhusiano dhaifu kiasi.

### Unahitaji data nyingi safi

Logistic regression itatoa matokeo sahihi zaidi ikiwa utatumia data zaidi; dataset yetu ndogo siyo bora kwa kazi hii, kwa hivyo kumbuka hilo.

[![ML kwa wanaoanza - Uchambuzi wa Data na Maandalizi kwa Logistic Regression](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML kwa wanaoanza - Uchambuzi wa Data na Maandalizi kwa Logistic Regression")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa video kuhusu kuandaa data kwa linear regression

✅ Fikiria aina za data ambazo zinafaa kwa logistic regression

## Zoezi - safisha data

Kwanza, safisha data kidogo, ukiondoa thamani za null na kuchagua baadhi ya safu:

1. Ongeza msimbo ufuatao:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Unaweza daima kuchungulia dataframe yako mpya:

    ```python
    pumpkins.info
    ```

### Uonyeshaji - ploti ya makundi

Hadi sasa umepakia [notebook ya kuanzia](../../../../2-Regression/4-Logistic/notebook.ipynb) na data ya malenge tena na kuisafisha ili kuhifadhi dataset inayojumuisha vigezo vichache, ikiwa ni pamoja na `Color`. Hebu tuonyeshe dataframe katika notebook kwa kutumia maktaba tofauti: [Seaborn](https://seaborn.pydata.org/index.html), ambayo imejengwa juu ya Matplotlib tuliyotumia awali.

Seaborn inatoa njia nzuri za kuonyesha data yako. Kwa mfano, unaweza kulinganisha usambazaji wa data kwa kila `Variety` na `Color` katika ploti ya makundi.

1. Unda ploti kama hiyo kwa kutumia kazi ya `catplot`, ukitumia data yetu ya malenge `pumpkins`, na kubainisha ramani ya rangi kwa kila kundi la malenge (machungwa au nyeupe):

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

    ![Gridi ya data iliyoonyeshwa](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Kwa kuchunguza data, unaweza kuona jinsi data ya Color inavyohusiana na Variety.

    ✅ Kwa kuzingatia ploti hii ya makundi, ni uchunguzi gani wa kuvutia unaweza kufikiria?

### Usindikaji wa data: encoding ya vipengele na lebo
Dataset yetu ya malenge ina thamani za maandishi kwa safu zake zote. Kufanya kazi na data ya makundi ni rahisi kwa binadamu lakini siyo kwa mashine. Algorithms za machine learning hufanya kazi vizuri na namba. Ndiyo maana encoding ni hatua muhimu sana katika awamu ya usindikaji wa data, kwani inatuwezesha kubadilisha data ya makundi kuwa data ya namba, bila kupoteza taarifa yoyote. Encoding nzuri husababisha kujenga mfano mzuri.

Kwa encoding ya vipengele kuna aina mbili kuu za encoders:

1. Ordinal encoder: inafaa vizuri kwa vigezo vya ordinal, ambavyo ni vigezo vya makundi ambapo data yao inafuata mpangilio wa mantiki, kama safu ya `Item Size` katika dataset yetu. Inaunda ramani ambapo kila kundi linawakilishwa na namba, ambayo ni mpangilio wa kundi katika safu.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Categorical encoder: inafaa vizuri kwa vigezo vya nominal, ambavyo ni vigezo vya makundi ambapo data yao haifuati mpangilio wa mantiki, kama vipengele vyote tofauti na `Item Size` katika dataset yetu. Ni encoding ya one-hot, ambayo inamaanisha kwamba kila kundi linawakilishwa na safu ya binary: kipengele kilichosimbwa ni sawa na 1 ikiwa malenge ni wa Variety hiyo na 0 vinginevyo.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Kisha, `ColumnTransformer` hutumika kuunganisha encoders nyingi katika hatua moja na kuzitumia kwa safu zinazofaa.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
Kwa upande mwingine, ili kusimba lebo, tunatumia darasa la `LabelEncoder` la scikit-learn, ambalo ni darasa la zana kusaidia kuweka lebo katika hali ya kawaida ili ziwe na thamani tu kati ya 0 na n_classes-1 (hapa, 0 na 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Mara tu tunapokuwa tumesimba vipengele na lebo, tunaweza kuviunganisha katika dataframe mpya `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Ni faida gani za kutumia ordinal encoder kwa safu ya `Item Size`?

### Changanua uhusiano kati ya vigezo

Sasa kwa kuwa tumefanya usindikaji wa data yetu, tunaweza kuchanganua uhusiano kati ya vipengele na lebo ili kupata wazo la jinsi mfano utakavyoweza kutabiri lebo kwa kuzingatia vipengele.
Njia bora ya kufanya uchambuzi wa aina hii ni kuonyesha data. Tutatumia tena kazi ya Seaborn `catplot`, kuonyesha uhusiano kati ya `Item Size`,  `Variety` na `Color` katika ploti ya makundi. Ili kuonyesha data vizuri tutatumia safu ya `Item Size` iliyosimbwa na safu ya `Variety` isiyosimbwa.

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
![Ploti ya makundi ya data iliyoonyeshwa](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Tumia swarm plot

Kwa kuwa Color ni kundi la binary (Nyeupe au Siyo), inahitaji '[mbinu maalum](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) ya uonyeshaji'. Kuna njia nyingine za kuonyesha uhusiano wa kundi hili na vigezo vingine.

Unaweza kuonyesha vigezo kando kwa kando na ploti za Seaborn.

1. Jaribu ploti ya 'swarm' kuonyesha usambazaji wa thamani:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Swarm ya data iliyoonyeshwa](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Angalizo**: msimbo hapo juu unaweza kutoa onyo, kwa kuwa seaborn inashindwa kuwakilisha idadi kubwa ya datapoints katika ploti ya swarm. Suluhisho linalowezekana ni kupunguza ukubwa wa alama, kwa kutumia parameter ya 'size'. Hata hivyo, fahamu kwamba hili linaathiri usomaji wa ploti.

> **🧮 Nionyeshe Hisabati**
>
> Logistic regression inategemea dhana ya 'uwezekano wa juu zaidi' kwa kutumia [sigmoid functions](https://wikipedia.org/wiki/Sigmoid_function). 'Sigmoid Function' kwenye ploti inaonekana kama umbo la 'S'. Inachukua thamani na kuipanga mahali fulani kati ya 0 na 1. Curve yake pia inaitwa 'logistic curve'. Fomula yake inaonekana kama hii:
>
> ![logistic function](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> ambapo katikati ya sigmoid inajipata katika nukta ya 0 ya x, L ni thamani ya juu ya curve, na k ni mwinuko wa curve. Ikiwa matokeo ya function ni zaidi ya 0.5, lebo husika itapewa darasa '1' la chaguo la binary. Ikiwa siyo, itatambuliwa kama '0'.

## Jenga mfano wako

Kujenga mfano wa kupata uainishaji wa binary ni rahisi kushangaza katika Scikit-learn.

[![ML kwa wanaoanza - Logistic Regression kwa uainishaji wa data](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML kwa wanaoanza - Logistic Regression kwa uainishaji wa data")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa video kuhusu kujenga mfano wa linear regression

1. Chagua vigezo unavyotaka kutumia katika mfano wako wa uainishaji na gawanya seti za mafunzo na majaribio kwa kuita `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Sasa unaweza kufundisha mfano wako, kwa kuita `fit()` na data yako ya mafunzo, na kuchapisha matokeo yake:

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

    Angalia scoreboard ya mfano wako. Si mbaya, ukizingatia una takriban safu 1000 za data:

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

## Uelewa bora kupitia confusion matrix

Ingawa unaweza kupata ripoti ya scoreboard [maneno](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) kwa kuchapisha vitu hapo juu, unaweza kuelewa mfano wako kwa urahisi zaidi kwa kutumia [confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) kutusaidia kuelewa jinsi mfano unavyofanya kazi.

> 🎓 '[Confusion matrix](https://wikipedia.org/wiki/Confusion_matrix)' (au 'error matrix') ni jedwali linaloonyesha positives na negatives za kweli dhidi ya za uongo za mfano wako, hivyo kupima usahihi wa utabiri.

1. Ili kutumia confusion matrix, ita `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Angalia confusion matrix ya mfano wako:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Katika Scikit-learn, confusion matrices Safu (axis 0) ni lebo halisi na safu (axis 1) ni lebo zilizotabiriwa.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Nini kinaendelea hapa? Tuseme mfano wetu umeombwa kuainisha malenge kati ya makundi mawili ya binary, kundi 'nyeupe' na kundi 'siyo nyeupe'.

- Ikiwa mfano wako unatabiri malenge kama siyo nyeupe na ni wa kundi 'siyo nyeupe' kwa kweli tunaita true negative, inayoonyeshwa na namba ya juu kushoto.
- Ikiwa mfano wako unatabiri malenge kama nyeupe na ni wa kundi 'siyo nyeupe' kwa kweli tunaita false negative, inayoonyeshwa na namba ya chini kushoto. 
- Ikiwa mfano wako unatabiri malenge kama siyo nyeupe na ni wa kundi 'nyeupe' kwa kweli tunaita false positive, inayoonyeshwa na namba ya juu kulia. 
- Ikiwa mfano wako unatabiri malenge kama nyeupe na ni wa kundi 'nyeupe' kwa kweli tunaita true positive, inayoonyeshwa na namba ya chini kulia.

Kama unavyoweza kudhani ni bora kuwa na idadi kubwa ya true positives na true negatives na idadi ndogo ya false positives na false negatives, ambayo inaonyesha kwamba mfano unafanya kazi vizuri.
Je, matriki ya mkanganyiko inahusianaje na usahihi na urejeshaji? Kumbuka, ripoti ya uainishaji iliyoonyeshwa hapo juu ilionyesha usahihi (0.85) na urejeshaji (0.67).

Usahihi = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Urejeshaji = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Swali: Kulingana na matriki ya mkanganyiko, je, modeli ilifanya vipi? Jibu: Si mbaya; kuna idadi nzuri ya hasi za kweli lakini pia kuna chache hasi za uongo.

Hebu turejee maneno tuliyoyaona awali kwa msaada wa ramani ya TP/TN na FP/FN ya matriki ya mkanganyiko:

🎓 Usahihi: TP/(TP + FP) Sehemu ya matukio yanayofaa kati ya matukio yaliyopatikana (mfano, ni lebo zipi ziliwekwa alama vizuri)

🎓 Urejeshaji: TP/(TP + FN) Sehemu ya matukio yanayofaa yaliyopatikana, iwe yamewekwa alama vizuri au la

🎓 Alama ya f1: (2 * usahihi * urejeshaji)/(usahihi + urejeshaji) Wastani wa uzito wa usahihi na urejeshaji, bora ikiwa 1 na mbaya ikiwa 0

🎓 Msaada: Idadi ya matukio ya kila lebo yaliyopatikana

🎓 Usahihi: (TP + TN)/(TP + TN + FP + FN) Asilimia ya lebo zilizotabiriwa kwa usahihi kwa sampuli.

🎓 Wastani wa Macro: Hesabu ya wastani wa vipimo visivyopimwa kwa kila lebo, bila kuzingatia usawa wa lebo.

🎓 Wastani wa Uzito: Hesabu ya wastani wa vipimo kwa kila lebo, kwa kuzingatia usawa wa lebo kwa kuzipima kulingana na msaada wao (idadi ya matukio ya kweli kwa kila lebo).

✅ Je, unaweza kufikiria ni kipimo gani unachopaswa kuangalia ikiwa unataka modeli yako kupunguza idadi ya hasi za uongo?

## Kuonyesha Mchoro wa ROC wa modeli hii

[![ML kwa wanaoanza - Kuchambua Utendaji wa Logistic Regression kwa Mchoro wa ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML kwa wanaoanza - Kuchambua Utendaji wa Logistic Regression kwa Mchoro wa ROC")

> 🎥 Bofya picha hapo juu kwa muhtasari mfupi wa video kuhusu michoro ya ROC

Hebu tufanye uonyeshaji mmoja zaidi ili kuona kinachoitwa 'mchoro wa ROC':

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

Kwa kutumia Matplotlib, chora [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) au ROC ya modeli. Michoro ya ROC mara nyingi hutumika kupata mtazamo wa matokeo ya kionyeshi kwa kuzingatia chanya za kweli dhidi ya chanya za uongo. "Michoro ya ROC kwa kawaida huonyesha kiwango cha chanya za kweli kwenye mhimili wa Y, na kiwango cha chanya za uongo kwenye mhimili wa X." Kwa hivyo, mwinuko wa mchoro na nafasi kati ya mstari wa katikati na mchoro ni muhimu: unataka mchoro unaopanda haraka na kupita mstari. Katika kesi yetu, kuna chanya za uongo mwanzoni, kisha mstari unapanda na kupita vizuri:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Hatimaye, tumia [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) ya Scikit-learn kuhesabu 'Eneo Chini ya Mchoro' (AUC) halisi:

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Matokeo ni `0.9749908725812341`. Kwa kuwa AUC inatofautiana kati ya 0 na 1, unataka alama kubwa, kwa kuwa modeli ambayo ni sahihi kwa 100% katika utabiri wake itakuwa na AUC ya 1; katika kesi hii, modeli ni _nzuri sana_.

Katika masomo ya baadaye kuhusu uainishaji, utajifunza jinsi ya kurudia ili kuboresha alama za modeli yako. Lakini kwa sasa, hongera! Umehitimisha masomo haya ya regression!

---
## 🚀Changamoto

Kuna mengi zaidi ya kuchunguza kuhusu logistic regression! Lakini njia bora ya kujifunza ni kujaribu. Tafuta seti ya data inayofaa kwa aina hii ya uchambuzi na tengeneza modeli nayo. Unajifunza nini? Kidokezo: jaribu [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) kwa seti za data za kuvutia.

## [Jaribio baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Soma kurasa chache za kwanza za [karatasi hii kutoka Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) kuhusu matumizi ya vitendo ya logistic regression. Fikiria kuhusu kazi ambazo zinafaa zaidi kwa aina moja au nyingine ya kazi za regression ambazo tumejifunza hadi sasa. Ni ipi ingefanya kazi bora?

## Kazi

[Kurudia regression hii](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.