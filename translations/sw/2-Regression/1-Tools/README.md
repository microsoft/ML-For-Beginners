<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T15:20:12+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "sw"
}
-->
# Anza na Python na Scikit-learn kwa mifano ya regression

![Muhtasari wa regressions katika sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote na [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Jaribio la awali la somo](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somo hili linapatikana kwa R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Utangulizi

Katika masomo haya manne, utajifunza jinsi ya kujenga mifano ya regression. Tutajadili matumizi yake hivi karibuni. Lakini kabla ya kuanza, hakikisha una zana sahihi za kuanza mchakato!

Katika somo hili, utajifunza jinsi ya:

- Kuseti kompyuta yako kwa kazi za kujifunza mashine za ndani.
- Kufanya kazi na Jupyter notebooks.
- Kutumia Scikit-learn, ikiwa ni pamoja na usakinishaji.
- Kuchunguza regression ya mstari kupitia zoezi la vitendo.

## Usakinishaji na usanidi

[![ML kwa wanaoanza - Sanidi zana zako tayari kujenga mifano ya Kujifunza Mashine](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML kwa wanaoanza - Sanidi zana zako tayari kujenga mifano ya Kujifunza Mashine")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya jinsi ya kusanidi kompyuta yako kwa ML.

1. **Sakinisha Python**. Hakikisha kuwa [Python](https://www.python.org/downloads/) imewekwa kwenye kompyuta yako. Utatumia Python kwa kazi nyingi za sayansi ya data na kujifunza mashine. Mfumo mwingi wa kompyuta tayari una usakinishaji wa Python. Kuna [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) zinazopatikana pia, ili kurahisisha usanidi kwa watumiaji wengine.

   Baadhi ya matumizi ya Python yanahitaji toleo moja la programu, wakati mengine yanahitaji toleo tofauti. Kwa sababu hii, ni muhimu kufanya kazi ndani ya [mazingira ya kawaida](https://docs.python.org/3/library/venv.html).

2. **Sakinisha Visual Studio Code**. Hakikisha kuwa Visual Studio Code imewekwa kwenye kompyuta yako. Fuata maelekezo haya ili [kusakinisha Visual Studio Code](https://code.visualstudio.com/) kwa usakinishaji wa msingi. Utatumia Python ndani ya Visual Studio Code katika kozi hii, kwa hivyo unaweza kutaka kujifunza jinsi ya [kusanidi Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) kwa maendeleo ya Python.

   > Jifunze Python kwa kupitia mkusanyiko huu wa [moduli za kujifunza](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Sanidi Python na Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Sanidi Python na Visual Studio Code")
   >
   > 🎥 Bonyeza picha hapo juu kwa video: kutumia Python ndani ya VS Code.

3. **Sakinisha Scikit-learn**, kwa kufuata [maelekezo haya](https://scikit-learn.org/stable/install.html). Kwa kuwa unahitaji kuhakikisha unatumia Python 3, inashauriwa utumie mazingira ya kawaida. Kumbuka, ikiwa unasakinisha maktaba hii kwenye Mac ya M1, kuna maelekezo maalum kwenye ukurasa uliohusishwa hapo juu.

4. **Sakinisha Jupyter Notebook**. Utahitaji [kusakinisha kifurushi cha Jupyter](https://pypi.org/project/jupyter/).

## Mazingira yako ya kuandika ML

Utatumia **notebooks** kuendeleza msimbo wako wa Python na kuunda mifano ya kujifunza mashine. Aina hii ya faili ni zana ya kawaida kwa wanasayansi wa data, na zinaweza kutambulika kwa kiambishi au kiendelezi `.ipynb`.

Notebooks ni mazingira ya maingiliano yanayomruhusu msanidi programu kuandika msimbo na kuongeza maelezo na kuandika nyaraka kuzunguka msimbo, jambo ambalo ni muhimu sana kwa miradi ya majaribio au utafiti.

[![ML kwa wanaoanza - Sanidi Jupyter Notebooks kuanza kujenga mifano ya regression](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML kwa wanaoanza - Sanidi Jupyter Notebooks kuanza kujenga mifano ya regression")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya kufanya zoezi hili.

### Zoezi - kufanya kazi na notebook

Katika folda hii, utapata faili _notebook.ipynb_.

1. Fungua _notebook.ipynb_ ndani ya Visual Studio Code.

   Seva ya Jupyter itaanza na Python 3+ imeanzishwa. Utapata maeneo ya notebook ambayo yanaweza `kuendeshwa`, vipande vya msimbo. Unaweza kuendesha kipande cha msimbo kwa kuchagua ikoni inayofanana na kitufe cha kucheza.

2. Chagua ikoni ya `md` na ongeza kidogo cha markdown, na maandishi yafuatayo **# Karibu kwenye notebook yako**.

   Kisha, ongeza msimbo wa Python.

3. Andika **print('hello notebook')** kwenye kipande cha msimbo.
4. Chagua mshale ili kuendesha msimbo.

   Unapaswa kuona taarifa iliyochapishwa:

    ```output
    hello notebook
    ```

![VS Code na notebook imefunguliwa](../../../../2-Regression/1-Tools/images/notebook.jpg)

Unaweza kuchanganya msimbo wako na maoni ili kujidokumentisha notebook.

✅ Fikiria kwa dakika moja jinsi mazingira ya kazi ya msanidi programu wa wavuti yanavyotofautiana na yale ya mwanasayansi wa data.

## Kuanzisha na Scikit-learn

Sasa kwa kuwa Python imewekwa katika mazingira yako ya ndani, na unajisikia vizuri na Jupyter notebooks, hebu tujifunze vizuri na Scikit-learn (tamka `sci` kama `science`). Scikit-learn inatoa [API ya kina](https://scikit-learn.org/stable/modules/classes.html#api-ref) kusaidia kufanya kazi za ML.

Kulingana na [tovuti yao](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn ni maktaba ya kujifunza mashine ya chanzo huria inayounga mkono kujifunza kwa usimamizi na bila usimamizi. Pia inatoa zana mbalimbali za kufaa mifano, usindikaji wa data, uteuzi wa mifano na tathmini, na huduma nyingine nyingi."

Katika kozi hii, utatumia Scikit-learn na zana nyingine kujenga mifano ya kujifunza mashine kufanya kile tunachokiita 'kazi za kujifunza mashine za jadi'. Tumeepuka makusudi mitandao ya neva na kujifunza kwa kina, kwani zinashughulikiwa vyema katika mtaala wetu ujao wa 'AI kwa Wanaoanza'.

Scikit-learn inafanya iwe rahisi kujenga mifano na kuipima kwa matumizi. Inalenga hasa kutumia data ya nambari na ina datasets kadhaa zilizotayarishwa tayari kwa matumizi kama zana za kujifunza. Pia inajumuisha mifano iliyojengwa tayari kwa wanafunzi kujaribu. Hebu tuchunguze mchakato wa kupakia data iliyopakiwa awali na kutumia estimator iliyojengwa kwa mfano wa kwanza wa ML na Scikit-learn na data ya msingi.

## Zoezi - notebook yako ya kwanza ya Scikit-learn

> Mafunzo haya yamechochewa na [mfano wa regression ya mstari](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) kwenye tovuti ya Scikit-learn.

[![ML kwa wanaoanza - Mradi wako wa Kwanza wa Regression ya Mstari katika Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML kwa wanaoanza - Mradi wako wa Kwanza wa Regression ya Mstari katika Python")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya kufanya zoezi hili.

Katika faili _notebook.ipynb_ lililohusishwa na somo hili, futa seli zote kwa kubonyeza ikoni ya 'takataka'.

Katika sehemu hii, utatumia dataset ndogo kuhusu ugonjwa wa kisukari ambayo imejengwa ndani ya Scikit-learn kwa madhumuni ya kujifunza. Fikiria kwamba unataka kujaribu matibabu kwa wagonjwa wa kisukari. Mifano ya Kujifunza Mashine inaweza kusaidia kuamua ni wagonjwa gani watakaoitikia matibabu vizuri zaidi, kulingana na mchanganyiko wa vigezo. Hata mfano wa regression ya msingi, unapowekwa kwenye grafu, unaweza kuonyesha taarifa kuhusu vigezo ambavyo vinaweza kusaidia kupanga majaribio yako ya kliniki ya kinadharia.

✅ Kuna aina nyingi za mbinu za regression, na ni ipi unayochagua inategemea jibu unalotafuta. Ikiwa unataka kutabiri urefu unaowezekana wa mtu wa umri fulani, ungetumia regression ya mstari, kwani unatafuta **thamani ya nambari**. Ikiwa unavutiwa na kugundua kama aina ya chakula inapaswa kuzingatiwa kuwa ya mboga au la, unatafuta **ugawaji wa kategoria** kwa hivyo ungetumia regression ya logistic. Utajifunza zaidi kuhusu regression ya logistic baadaye. Fikiria kidogo kuhusu maswali unayoweza kuuliza data, na ni mbinu ipi kati ya hizi itakuwa sahihi zaidi.

Hebu tuanze kazi hii.

### Ingiza maktaba

Kwa kazi hii tutaleta maktaba kadhaa:

- **matplotlib**. Ni [chombo cha kuchora grafu](https://matplotlib.org/) na tutakitumia kuunda grafu ya mstari.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) ni maktaba muhimu kwa kushughulikia data ya nambari katika Python.
- **sklearn**. Hii ni maktaba ya [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Ingiza maktaba kadhaa kusaidia kazi zako.

1. Ongeza imports kwa kuandika msimbo ufuatao:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Hapo juu unaleta `matplotlib`, `numpy` na unaleta `datasets`, `linear_model` na `model_selection` kutoka `sklearn`. `model_selection` hutumika kugawanya data katika seti za mafunzo na majaribio.

### Dataset ya kisukari

Dataset ya [kisukari](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) iliyojengwa ndani ina sampuli 442 za data kuhusu kisukari, na vigezo 10 vya sifa, baadhi ya ambazo ni pamoja na:

- umri: umri kwa miaka
- bmi: index ya uzito wa mwili
- bp: shinikizo la damu la wastani
- s1 tc: T-Cells (aina ya seli nyeupe za damu)

✅ Dataset hii inajumuisha dhana ya 'jinsia' kama kigezo cha sifa muhimu kwa utafiti kuhusu kisukari. Dataset nyingi za matibabu zinajumuisha aina hii ya uainishaji wa binary. Fikiria kidogo kuhusu jinsi uainishaji kama huu unaweza kuwatenga sehemu fulani za idadi ya watu kutoka kwa matibabu.

Sasa, pakia data ya X na y.

> 🎓 Kumbuka, hii ni kujifunza kwa usimamizi, na tunahitaji lengo lililopewa jina 'y'.

Katika seli mpya ya msimbo, pakia dataset ya kisukari kwa kuita `load_diabetes()`. Ingizo `return_X_y=True` linaashiria kwamba `X` itakuwa matrix ya data, na `y` itakuwa lengo la regression.

1. Ongeza amri za kuchapisha ili kuonyesha umbo la matrix ya data na kipengele chake cha kwanza:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Unachopata kama jibu ni tuple. Unachofanya ni kugawa thamani mbili za kwanza za tuple kwa `X` na `y` mtawalia. Jifunze zaidi [kuhusu tuples](https://wikipedia.org/wiki/Tuple).

    Unaweza kuona kwamba data hii ina vitu 442 vilivyopangwa katika arrays za vipengele 10:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Fikiria kidogo kuhusu uhusiano kati ya data na lengo la regression. Regression ya mstari inatabiri uhusiano kati ya kipengele X na kigezo cha lengo y. Je, unaweza kupata [lengo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) kwa dataset ya kisukari katika nyaraka? Dataset hii inaonyesha nini, ikizingatiwa lengo?

2. Kisha, chagua sehemu ya dataset hii ili kuweka grafu kwa kuchagua safu ya 3 ya dataset. Unaweza kufanya hivi kwa kutumia operator `:` kuchagua safu zote, na kisha kuchagua safu ya 3 kwa kutumia index (2). Unaweza pia kupanga upya data kuwa array ya 2D - kama inavyohitajika kwa kuweka grafu - kwa kutumia `reshape(n_rows, n_columns)`. Ikiwa moja ya parameter ni -1, kipimo kinacholingana kinahesabiwa kiotomatiki.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Wakati wowote, chapisha data ili kuangalia umbo lake.

3. Sasa kwa kuwa una data tayari kuwekwa kwenye grafu, unaweza kuona kama mashine inaweza kusaidia kuamua mgawanyiko wa kimantiki kati ya nambari katika dataset hii. Ili kufanya hivyo, unahitaji kugawanya data (X) na lengo (y) katika seti za majaribio na mafunzo. Scikit-learn ina njia rahisi ya kufanya hivi; unaweza kugawanya data yako ya majaribio katika sehemu fulani.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Sasa uko tayari kufundisha mfano wako! Pakia mfano wa regression ya mstari na uufundishe na seti zako za mafunzo za X na y kwa kutumia `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` ni kazi utakayoiona katika maktaba nyingi za ML kama TensorFlow.

5. Kisha, tengeneza utabiri kwa kutumia data ya majaribio, kwa kutumia kazi `predict()`. Hii itatumika kuchora mstari kati ya vikundi vya data.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Sasa ni wakati wa kuonyesha data kwenye grafu. Matplotlib ni chombo muhimu sana kwa kazi hii. Tengeneza grafu ya alama za X na y za majaribio, na tumia utabiri kuchora mstari mahali panapofaa zaidi, kati ya vikundi vya data vya mfano.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![grafu ya alama za data kuhusu kisukari](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Fikiria kidogo kuhusu kinachoendelea hapa. Mstari wa moja kwa moja unapita katikati ya nukta nyingi ndogo za data, lakini unafanya nini hasa? Je, unaweza kuona jinsi unavyoweza kutumia mstari huu kutabiri mahali ambapo data mpya, ambayo haijawahi kuonekana, inapaswa kutoshea kuhusiana na mhimili wa y wa mchoro? Jaribu kuelezea kwa maneno matumizi ya vitendo ya mfano huu.

Hongera, umeunda mfano wako wa kwanza wa regression ya mstari, umetengeneza utabiri kwa kutumia, na kuonyesha katika mchoro!

---
## 🚀Changamoto

Chora variable tofauti kutoka kwa dataset hii. Kidokezo: hariri mstari huu: `X = X[:,2]`. Ukiangalia lengo la dataset hii, unaweza kugundua nini kuhusu maendeleo ya ugonjwa wa kisukari?

## [Jaribio baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Katika mafunzo haya, ulifanya kazi na regression rahisi ya mstari, badala ya regression ya univariate au multiple. Soma kidogo kuhusu tofauti kati ya mbinu hizi, au angalia [video hii](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Soma zaidi kuhusu dhana ya regression na fikiria ni aina gani za maswali yanayoweza kujibiwa kwa mbinu hii. Chukua [mafunzo haya](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) ili kuongeza uelewa wako.

## Kazi

[Dataset tofauti](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.