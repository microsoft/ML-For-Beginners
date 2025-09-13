<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T18:10:22+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "tl"
}
-->
# Logistic regression para sa pag-predict ng mga kategorya

![Infographic ng Logistic vs. Linear Regression](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ang araling ito ay available sa R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Panimula

Sa huling araling ito tungkol sa Regression, isa sa mga pangunahing _classic_ na teknik sa ML, tatalakayin natin ang Logistic Regression. Ginagamit ang teknik na ito upang matuklasan ang mga pattern para mag-predict ng binary categories. Ang candy ba ay chocolate o hindi? Ang sakit ba ay nakakahawa o hindi? Pipiliin ba ng customer ang produktong ito o hindi?

Sa araling ito, matututunan mo:

- Isang bagong library para sa data visualization
- Mga teknik para sa logistic regression

✅ Palalimin ang iyong kaalaman sa paggamit ng ganitong uri ng regression sa [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Kinakailangan

Dahil nagtrabaho na tayo sa pumpkin data, sapat na ang ating kaalaman upang mapansin na may isang binary category na maaari nating pag-aralan: `Color`.

Gumawa tayo ng logistic regression model upang mag-predict kung, batay sa ilang variables, _anong kulay ang malamang na maging isang pumpkin_ (orange 🎃 o white 👻).

> Bakit natin pinag-uusapan ang binary classification sa isang aralin tungkol sa regression? Para lamang sa linguistic convenience, dahil ang logistic regression ay [talagang isang classification method](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), bagamat linear-based. Matuto tungkol sa iba pang paraan ng pag-classify ng data sa susunod na grupo ng aralin.

## Tukuyin ang tanong

Para sa ating layunin, ipapahayag natin ito bilang binary: 'White' o 'Not White'. Mayroon ding 'striped' na kategorya sa ating dataset ngunit kakaunti ang instances nito, kaya hindi natin ito gagamitin. Nawawala rin ito kapag tinanggal natin ang mga null values mula sa dataset.

> 🎃 Nakakatuwang kaalaman, minsan tinatawag natin ang mga white pumpkins na 'ghost' pumpkins. Hindi sila madaling i-carve, kaya hindi sila kasing popular ng mga orange pumpkins ngunit cool silang tingnan! Kaya maaari rin nating baguhin ang tanong bilang: 'Ghost' o 'Not Ghost'. 👻

## Tungkol sa logistic regression

Ang logistic regression ay naiiba sa linear regression, na natutunan mo na dati, sa ilang mahahalagang aspeto.

[![ML para sa mga nagsisimula - Pag-unawa sa Logistic Regression para sa Machine Learning Classification](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML para sa mga nagsisimula - Pag-unawa sa Logistic Regression para sa Machine Learning Classification")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng logistic regression.

### Binary classification

Ang logistic regression ay hindi nag-aalok ng parehong features tulad ng linear regression. Ang una ay nagbibigay ng prediction tungkol sa binary category ("white o hindi white") samantalang ang huli ay may kakayahang mag-predict ng patuloy na values, halimbawa batay sa pinanggalingan ng pumpkin at oras ng pag-ani, _kung gaano tataas ang presyo nito_.

![Model ng Pumpkin Classification](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Iba pang classifications

May iba pang uri ng logistic regression, kabilang ang multinomial at ordinal:

- **Multinomial**, na may higit sa isang kategorya - "Orange, White, at Striped".
- **Ordinal**, na may ordered categories, kapaki-pakinabang kung nais nating i-order ang ating outcomes nang lohikal, tulad ng ating pumpkins na naka-order batay sa finite na bilang ng sizes (mini, sm, med, lg, xl, xxl).

![Multinomial vs ordinal regression](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Hindi kailangang mag-correlate ang mga variables

Tandaan kung paano mas gumagana ang linear regression sa mas correlated na variables? Ang logistic regression ay kabaligtaran - hindi kailangang mag-align ang mga variables. Angkop ito para sa data na may medyo mahina na correlations.

### Kailangan mo ng maraming malinis na data

Mas accurate ang resulta ng logistic regression kung mas maraming data ang ginagamit; ang maliit na dataset natin ay hindi optimal para sa task na ito, kaya tandaan ito.

[![ML para sa mga nagsisimula - Data Analysis at Preparation para sa Logistic Regression](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML para sa mga nagsisimula - Data Analysis at Preparation para sa Logistic Regression")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng paghahanda ng data para sa linear regression

✅ Pag-isipan ang mga uri ng data na angkop para sa logistic regression

## Ehersisyo - ayusin ang data

Una, linisin ang data nang kaunti, tanggalin ang mga null values at piliin lamang ang ilang mga column:

1. Idagdag ang sumusunod na code:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Maaari mong tingnan ang iyong bagong dataframe:

    ```python
    pumpkins.info
    ```

### Visualization - categorical plot

Sa puntong ito, na-load mo na ang [starter notebook](../../../../2-Regression/4-Logistic/notebook.ipynb) gamit ang pumpkin data at nilinis ito upang mapanatili ang dataset na naglalaman ng ilang variables, kabilang ang `Color`. I-visualize natin ang dataframe sa notebook gamit ang ibang library: [Seaborn](https://seaborn.pydata.org/index.html), na nakabatay sa Matplotlib na ginamit natin dati.

Nag-aalok ang Seaborn ng mga magagandang paraan upang i-visualize ang iyong data. Halimbawa, maaari mong ikumpara ang distributions ng data para sa bawat `Variety` at `Color` sa isang categorical plot.

1. Gumawa ng ganitong plot gamit ang `catplot` function, gamit ang pumpkin data `pumpkins`, at tukuyin ang color mapping para sa bawat pumpkin category (orange o white):

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

    ![Grid ng visualized data](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Sa pag-obserba ng data, makikita mo kung paano nauugnay ang Color data sa Variety.

    ✅ Batay sa categorical plot na ito, ano ang mga kawili-wiling explorations na maaari mong maisip?

### Data pre-processing: feature at label encoding

Ang dataset ng pumpkins ay naglalaman ng string values para sa lahat ng mga column nito. Ang paggamit ng categorical data ay intuitive para sa mga tao ngunit hindi para sa mga makina. Ang machine learning algorithms ay mas mahusay gumana sa mga numero. Kaya't ang encoding ay isang napakahalagang hakbang sa data pre-processing phase, dahil pinapayagan tayo nitong gawing numerical data ang categorical data, nang hindi nawawala ang anumang impormasyon. Ang mahusay na encoding ay nagreresulta sa paggawa ng mahusay na modelo.

Para sa feature encoding, may dalawang pangunahing uri ng encoders:

1. Ordinal encoder: angkop ito para sa ordinal variables, na mga categorical variables kung saan ang kanilang data ay sumusunod sa lohikal na pagkakasunod-sunod, tulad ng `Item Size` column sa ating dataset. Gumagawa ito ng mapping kung saan ang bawat kategorya ay kinakatawan ng isang numero, na siyang pagkakasunod-sunod ng kategorya sa column.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Categorical encoder: angkop ito para sa nominal variables, na mga categorical variables kung saan ang kanilang data ay hindi sumusunod sa lohikal na pagkakasunod-sunod, tulad ng lahat ng features maliban sa `Item Size` sa ating dataset. Ito ay isang one-hot encoding, na nangangahulugang ang bawat kategorya ay kinakatawan ng isang binary column: ang encoded variable ay katumbas ng 1 kung ang pumpkin ay kabilang sa Variety na iyon at 0 kung hindi.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

Pagkatapos, ginagamit ang `ColumnTransformer` upang pagsamahin ang maraming encoders sa isang hakbang at i-apply ang mga ito sa tamang mga column.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

Sa kabilang banda, para i-encode ang label, ginagamit natin ang scikit-learn `LabelEncoder` class, na isang utility class upang gawing normal ang labels upang maglaman lamang ng mga values sa pagitan ng 0 at n_classes-1 (dito, 0 at 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

Kapag na-encode na natin ang features at label, maaari natin itong pagsamahin sa isang bagong dataframe `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ Ano ang mga benepisyo ng paggamit ng ordinal encoder para sa `Item Size` column?

### Suriin ang relasyon sa pagitan ng mga variables

Ngayon na na-pre-process na natin ang data, maaari nating suriin ang relasyon sa pagitan ng features at label upang magkaroon ng ideya kung gaano kahusay ang modelo sa pag-predict ng label batay sa features. Ang pinakamahusay na paraan upang gawin ang ganitong uri ng pagsusuri ay ang pag-plot ng data. Gagamitin natin muli ang Seaborn `catplot` function, upang i-visualize ang relasyon sa pagitan ng `Item Size`, `Variety`, at `Color` sa isang categorical plot. Para mas maayos ang pag-plot ng data, gagamitin natin ang encoded `Item Size` column at ang unencoded `Variety` column.

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

![Catplot ng visualized data](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Gumamit ng swarm plot

Dahil ang Color ay isang binary category (White o Not), kailangan nito ng 'isang [specialized approach](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) sa visualization'. May iba pang paraan upang i-visualize ang relasyon ng kategoryang ito sa ibang variables.

Maaari mong i-visualize ang mga variables nang magkatabi gamit ang Seaborn plots.

1. Subukan ang 'swarm' plot upang ipakita ang distribution ng values:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Swarm ng visualized data](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Mag-ingat**: ang code sa itaas ay maaaring mag-generate ng warning, dahil hindi maipapakita ng seaborn ang ganitong dami ng datapoints sa isang swarm plot. Ang posibleng solusyon ay bawasan ang laki ng marker, gamit ang 'size' parameter. Gayunpaman, tandaan na maaapektuhan nito ang readability ng plot.

> **🧮 Ipakita ang Math**
>
> Ang logistic regression ay nakabatay sa konsepto ng 'maximum likelihood' gamit ang [sigmoid functions](https://wikipedia.org/wiki/Sigmoid_function). Ang 'Sigmoid Function' sa isang plot ay mukhang hugis 'S'. Kinukuha nito ang isang value at ina-map ito sa pagitan ng 0 at 1. Ang curve nito ay tinatawag ding 'logistic curve'. Ang formula nito ay ganito:
>
> ![logistic function](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> kung saan ang midpoint ng sigmoid ay nasa x's 0 point, L ang maximum value ng curve, at k ang steepness ng curve. Kung ang resulta ng function ay higit sa 0.5, ang label ay bibigyan ng class '1' ng binary choice. Kung hindi, ito ay ikakategorya bilang '0'.

## Bumuo ng iyong modelo

Ang paggawa ng modelo upang mahanap ang mga binary classification ay nakakagulat na simple sa Scikit-learn.

[![ML para sa mga nagsisimula - Logistic Regression para sa classification ng data](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML para sa mga nagsisimula - Logistic Regression para sa classification ng data")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng paggawa ng linear regression model

1. Piliin ang mga variables na nais mong gamitin sa iyong classification model at hatiin ang training at test sets gamit ang `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Ngayon maaari mong i-train ang iyong modelo, gamit ang `fit()` sa iyong training data, at i-print ang resulta nito:

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

    Tingnan ang scoreboard ng iyong modelo. Hindi masama, isinasaalang-alang na mayroon ka lamang humigit-kumulang 1000 rows ng data:

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

## Mas mahusay na pag-unawa gamit ang confusion matrix

Habang maaari kang makakuha ng scoreboard report [terms](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) sa pamamagitan ng pag-print ng mga items sa itaas, mas mauunawaan mo ang iyong modelo gamit ang [confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) upang matulungan tayong maunawaan kung paano gumagana ang modelo.

> 🎓 Ang '[confusion matrix](https://wikipedia.org/wiki/Confusion_matrix)' (o 'error matrix') ay isang table na nagpapakita ng true vs. false positives at negatives ng iyong modelo, kaya't nasusukat ang accuracy ng predictions.

1. Upang gumamit ng confusion matrix, tawagin ang `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Tingnan ang confusion matrix ng iyong modelo:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Sa Scikit-learn, ang Rows (axis 0) ay actual labels at ang columns (axis 1) ay predicted labels.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Ano ang nangyayari dito? Sabihin nating ang modelo natin ay tinanong upang i-classify ang pumpkins sa pagitan ng dalawang binary categories, category 'white' at category 'not-white'.

- Kung ang modelo mo ay nag-predict ng pumpkin bilang hindi white at ito ay kabilang sa category 'not-white' sa realidad, tinatawag natin itong true negative, na ipinapakita ng numero sa kaliwang itaas.
- Kung ang modelo mo ay nag-predict ng pumpkin bilang white at ito ay kabilang sa category 'not-white' sa realidad, tinatawag natin itong false negative, na ipinapakita ng numero sa kaliwang ibaba.
- Kung ang modelo mo ay nag-predict ng pumpkin bilang hindi white at ito ay kabilang sa category 'white' sa realidad, tinatawag natin itong false positive, na ipinapakita ng numero sa kanang itaas.
- Kung ang modelo mo ay nag-predict ng pumpkin bilang white at ito ay kabilang sa category 'white' sa realidad, tinatawag natin itong true positive, na ipinapakita ng numero sa kanang ibaba.

Tulad ng inaasahan mo, mas mainam na magkaroon ng mas malaking bilang ng true positives at true negatives at mas mababang bilang ng false positives at false negatives, na nagpapahiwatig na mas mahusay ang performance ng modelo.
Paano nauugnay ang confusion matrix sa precision at recall? Tandaan, ang classification report na ipinakita sa itaas ay nagpakita ng precision (0.85) at recall (0.67).

Precision = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Recall = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Q: Ayon sa confusion matrix, paano ang performance ng modelo?  
A: Hindi masama; may magandang bilang ng true negatives ngunit mayroon ding ilang false negatives.

Balikan natin ang mga terminong nakita natin kanina gamit ang mapping ng TP/TN at FP/FN sa confusion matrix:

🎓 Precision: TP/(TP + FP)  
Ang bahagi ng mga tamang instance sa mga nakuha na instance (hal. aling mga label ang maayos na na-label)

🎓 Recall: TP/(TP + FN)  
Ang bahagi ng mga tamang instance na nakuha, maayos man ang pagkaka-label o hindi

🎓 f1-score: (2 * precision * recall)/(precision + recall)  
Isang weighted average ng precision at recall, kung saan ang pinakamaganda ay 1 at ang pinakamasama ay 0

🎓 Support:  
Ang bilang ng mga paglitaw ng bawat label na nakuha

🎓 Accuracy: (TP + TN)/(TP + TN + FP + FN)  
Ang porsyento ng mga label na tama ang prediksyon para sa isang sample.

🎓 Macro Avg:  
Ang kalkulasyon ng unweighted mean metrics para sa bawat label, hindi isinasaalang-alang ang imbalance ng label.

🎓 Weighted Avg:  
Ang kalkulasyon ng mean metrics para sa bawat label, isinasaalang-alang ang imbalance ng label sa pamamagitan ng pag-weight base sa kanilang support (ang bilang ng tamang instance para sa bawat label).

✅ Maaari mo bang isipin kung aling metric ang dapat mong bantayan kung gusto mong bawasan ang bilang ng false negatives?

## I-visualize ang ROC curve ng modelong ito

[![ML para sa mga nagsisimula - Pagsusuri ng Performance ng Logistic Regression gamit ang ROC Curves](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML para sa mga nagsisimula - Pagsusuri ng Performance ng Logistic Regression gamit ang ROC Curves")

> 🎥 I-click ang imahe sa itaas para sa isang maikling video overview ng ROC curves

Gumawa tayo ng isa pang visualization upang makita ang tinatawag na 'ROC' curve:

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

Gamit ang Matplotlib, i-plot ang [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) o ROC ng modelo. Ang ROC curves ay madalas gamitin upang makita ang output ng isang classifier sa mga aspeto ng true vs. false positives. "Ang ROC curves ay karaniwang may true positive rate sa Y axis, at false positive rate sa X axis." Kaya, mahalaga ang steepness ng curve at ang espasyo sa pagitan ng midpoint line at ng curve: gusto mo ng curve na mabilis na tumataas at lumalampas sa linya. Sa ating kaso, may false positives sa simula, at pagkatapos ay ang linya ay tumataas at lumalampas nang maayos:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Sa huli, gamitin ang [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) ng Scikit-learn upang kalkulahin ang aktwal na 'Area Under the Curve' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```  
Ang resulta ay `0.9749908725812341`. Dahil ang AUC ay nasa saklaw na 0 hanggang 1, gusto mo ng mataas na score, dahil ang isang modelo na 100% tama sa mga prediksyon nito ay magkakaroon ng AUC na 1; sa kasong ito, ang modelo ay _medyo magaling_.

Sa mga susunod na aralin tungkol sa classifications, matututo ka kung paano mag-iterate upang mapabuti ang mga score ng iyong modelo. Ngunit sa ngayon, binabati kita! Natapos mo na ang mga regression lessons!

---
## 🚀Hamunin

Marami pang dapat tuklasin tungkol sa logistic regression! Ngunit ang pinakamagandang paraan upang matuto ay ang mag-eksperimento. Maghanap ng dataset na angkop para sa ganitong uri ng pagsusuri at bumuo ng modelo gamit ito. Ano ang natutunan mo? Tip: subukan ang [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) para sa mga kawili-wiling dataset.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review at Pag-aaral sa Sarili

Basahin ang unang ilang pahina ng [papel na ito mula sa Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) tungkol sa ilang praktikal na gamit ng logistic regression. Pag-isipan ang mga gawain na mas angkop para sa isa o sa ibang uri ng regression na pinag-aralan natin hanggang sa puntong ito. Ano ang mas angkop?

## Takdang-Aralin

[Subukang muli ang regression na ito](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na pinagmulan. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.