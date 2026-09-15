# Anza na Python na Scikit-learn kwa mifano ya regression

![Muhtasari wa regressions katika sketchnote](../../../../translated_images/sw/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote na [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Mtihani kabla ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somu hii inapatikana kwa R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Utangulizi

Katika mihadhara hii minne, utagundua jinsi ya kujenga mifano ya regression. Tutajadili kwa ajili ya nini hivi karibuni. Lakini kabla hujafanya chochote, hakikisha una zana sahihi tayari kuanza mchakato!

Katika somu hii, utajifunza jinsi ya:

- Kusanidi kompyuta yako kwa kazi za kujifunza kwa mashine za ndani.
- Kufanya kazi na Jupyter Notebooks.
- Kutumia Scikit-learn, ikiwa ni pamoja na usakinishaji.
- Kuchunguza regression ya mstari kwa mazoezi ya vitendo.

## Usakinishaji na usanidi

[![ML kwa waanzilishi - Andaa zana zako tayari kujenga Mifano ya Kujifunza kwa Mashine](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML kwa waanzilishi - Andaa zana zako tayari kujenga Mifano ya Kujifunza kwa Mashine")

> 🎥 Bonyeza picha hapo juu kwa video fupi inayoelezea jinsi ya kusanidi kompyuta yako kwa ML.

1. **Sakinisha Python**. Hakikisha kuwa [Python](https://www.python.org/downloads/) imesakinishwa kwenye kompyuta yako. Utatumia Python kwa kazi nyingi za sayansi ya data na kujifunza kwa mashine. Mifumo mingi ya kompyuta tayari ina usakinishaji wa Python. Kuna [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) muhimu pia, kusaidia kwa urahisi wa usanidi kwa baadhi ya watumiaji.

   Baadhi ya matumizi ya Python, hata hivyo, yanahitaji toleo moja la programu, wakati mengine yanahitaji toleo tofauti. Kwa sababu hiyo, ni muhimu kufanya kazi ndani ya [mazingira pepe](https://docs.python.org/3/library/venv.html).

2. **Sakinisha Visual Studio Code**. Hakikisha unayo Visual Studio Code imesakinishwa kwenye kompyuta yako. Fuata maelekezo haya ya [kusakinisha Visual Studio Code](https://code.visualstudio.com/) kwa usakinishaji wa msingi. Utatumia Python ndani ya Visual Studio Code katika kozi hii, kwa hivyo unaweza kutaka kujifunza jinsi ya [kusanidi Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) kwa maendeleo ya Python.

   > Jifunze stadi za Python kwa kufanya kupitia mkusanyiko huu wa [Moduli za Kujifunza](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Sanidi Python na Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Sanidi Python na Visual Studio Code")
   >
   > 🎥 Bonyeza picha hapo juu kwa video: kutumia Python ndani ya VS Code.

3. **Sakinisha Scikit-learn**, kwa kufuata [maelekezo haya](https://scikit-learn.org/stable/install.html). Kwa kuwa unataka kuhakikisha unatumia Python 3, inashauriwa kutumia mazingira pepe. Kumbuka, ikiwa unasakinisha maktaba hii kwenye Mac ya M1, kuna maelekezo maalum kwenye ukurasa ulioainishwa hapo juu.

1. **Sakinisha Jupyter Notebook**. Utahitaji [kusakinisha kifurushi cha Jupyter](https://pypi.org/project/jupyter/).

## Mazingira yako ya uandishi wa ML

Utatumia **notebooks** kuendeleza msimbo wako wa Python na kuunda mifano ya kujifunza kwa mashine. Aina hii ya faili ni zana ya kawaida kwa wanasayansi wa data, na inaweza kutambuliwa kwa kiambishi au kiplugini chake `.ipynb`.

Notebooks ni mazingira ya maingiliano yanayomruhusu mtengenezaji wa programu kuandika msimbo pamoja na kuongeza maelezo na kuandika nyaraka kuhusiana na msimbo, jambo ambalo ni msaada kwa miradi ya majaribio au ya utafiti.

[![ML kwa waanzilishi - Sanidi Jupyter Notebooks kuanza kujenga mifano ya regression](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML kwa waanzilishi - Sanidi Jupyter Notebooks kuanza kujenga mifano ya regression")

> 🎥 Bonyeza picha hapo juu kwa video fupi inayoelezea mazoezi haya.

### Mazoezi - fanya kazi na notebook

Katika folda hii, utapata faili _notebook.ipynb_.

1. Fungua _notebook.ipynb_ katika Visual Studio Code.

   Seva ya Jupyter itaanza na Python 3+ imechaguliwa. Utaona maeneo katika notebook ambayo yanaweza `kukimbia`, vipande vya msimbo. Unaweza kuendesha blokki ya msimbo, kwa kuchagua ikoni inayofanana na kitufe cha kuendesha.

1. Chagua ikoni ya `md` na ongeza kidogo cha markup, na maandishi yafuatayo **# Karibu kwenye notebook yako**.

   Baadaye, ongeza msimbo wa Python.

1. Andika **print('hello notebook')** katika blokki ya msimbo.
1. Chagua mshale kuendesha msimbo.

   Unapaswa kuona taarifa iliyochapishwa:

    ```output
    hello notebook
    ```

![VS Code na notebook wazi](../../../../translated_images/sw/notebook.4a3ee31f396b8832.webp)

Unaweza kuingiza msimbo wako kwa maoni ili kujitayarisha nyaraka za notebook.

✅ Fikiria kwa dakika mmoja jinsi mazingira ya kazi ya mtaalamu wa wavuti yanavyotofautiana na yale ya mtaalamu wa sayansi ya data.

## Anza kutumia Scikit-learn

Sasa Python imewekwa kwenye mazingira yako ya ndani, na umezoea Jupyter Notebooks, hebu tufurahie kwa usawa Scikit-learn (itisemwe `sci` kama katika `science`). Scikit-learn hutoa [API pana](https://scikit-learn.org/stable/modules/classes.html#api-ref) kusaidia kufanya kazi za ML.

Kulingana na [tovuti yao](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn ni maktaba ya kujifunza kwa mashine iliyo wazi inayounga mkono kujifunza kwa usimamizi na bila usimamizi. Pia hutoa zana mbalimbali za kufitisha modeli, maandalizi ya data, kuchagua na kutathmini modeli, na huduma nyingine nyingi."

Katika kozi hii, utatumia Scikit-learn na zana nyingine kujenga mifano ya kujifunza kwa mashine kufanya kazi ambazo huitwa 'kazi za jadi za kujifunza kwa mashine'. Tumepuuza kwa makusudi mitandao ya neva na kujifunza kwa kina, kwani inashughulikiwa vizuri katika mtaala wetu unaokuja wa 'AI kwa Waanzilishi'.

Scikit-learn hufanya iwe rahisi kujenga mifano na kuitathmini kwa matumizi. Inazingatia data ya nambari hasa na ina seti kadhaa za data tayari kwa matumizi kama zana za kujifunza. Pia ina mifano iliyojengwa tayari kwa wanafunzi kujaribu. Hebu tuchunguze mchakato wa kupakia data iliyopakiwa na kutumia makadirio yaliyojengwa kuunda mfano wako wa kwanza wa ML na Scikit-learn kwa data rahisi.

## Mazoezi - notebook yako ya kwanza ya Scikit-learn

> Mafunzo haya yamechochewa na [mfano wa regression ya mstari](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) kwenye tovuti ya Scikit-learn.


[![ML kwa waanzilishi - Mradi wako wa Kwanza wa Linear Regression kwa Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML kwa waanzilishi - Mradi wako wa Kwanza wa Linear Regression kwa Python")

> 🎥 Bonyeza picha hapo juu kwa video fupi inayoelezea mazoezi haya.

Katika faili la _notebook.ipynb_ lililohusiana na somu hii, futa seli zote kwa kubonyeza ikoni ya 'makopo ya takataka'.

Katika sehemu hii, utafanya kazi na dataset ndogo kuhusu ugonjwa wa kisukari uliyojengwa ndani ya Scikit-learn kwa madhumuni ya kujifunza. Fikiria ungependa kujaribu tiba kwa wagonjwa wa kisukari. Mifano ya Kujifunza kwa Mashine inaweza kusaidia kubainisha wagonjwa ambao wangeweza kuathirika vyema na tiba, kulingana na mchanganyiko wa vigezo. Hata mfano wa regression rahisi, ukiwa umeonyeshwa kwa kuona, unaweza kuonyesha habari kuhusu vigezo vitakavyosaidia kupanga majaribio yako ya kliniki.

✅ Kuna aina nyingi za mbinu za regression, na ipi unachagua inategemea jibu unalotafuta. Ikiwa ungependa kutabiri urefu wa mtu fulani kwa umri fulani, utatumia regression ya mstari, kwa kuwa unatafuta **thamani ya nambari**. Ikiwa unavutiwa kugundua kama aina ya chakula inapaswa kuchukuliwa kama vegan au la, unatafuta **ugawaji wa kategoria** kwa hiyo utatumia logistic regression. Utajifunza zaidi kuhusu logistic regression baadaye. Fikiria kidogo kuhusu maswali unayoweza kuuliza data, na ni njia gani kati ya hizi ingekuwa bora zaidi.

Hebu tuanze kazi hii.

### Ingiza maktaba

Kwa kazi hii tutaunda maktaba kadhaa:

- **matplotlib**. Ni [zana ya kuchora grafu](https://matplotlib.org/) muhimu na tutaitumia kuunda mchoro wa mstari.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) ni maktaba muhimu ya kushughulikia data za nambari katika Python.
- **sklearn**. Hii ni maktaba ya [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Ingiza maktaba ili kusaidia kazi zako.

1. Ongeza ingizo kwa kuandika msimbo ufuatao:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Juu unaunda ingizo la `matplotlib`, `numpy` na unaunda ingizo la `datasets`, `linear_model` na `model_selection` kutoka `sklearn`. `model_selection` hutumika kugawa data katika seti za mafunzo na za mtihani.

### Dataset ya ugonjwa wa kisukari

Dataset ya [ugonjwa wa kisukari iliyojengwa](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) ina sampuli 442 za data kuhusu kisukari, na vigezo 10 vya sifa, baadhi ni:

- umri: umri kwa miaka
- bmi: index ya uzito wa mwili
- bp: shinikizo la damu wastani
- s1 tc: Seli T (aina ya seli nyeupe za damu)

✅ Dataset hii ina dhana ya ‘jinsia’ kama kigezo muhimu kwa utafiti kuhusu kisukari. Dataset nyingi za matibabu zina usahihishaji huu wa binary. Fikiria kidogo jinsi ugawaji kama huu unaweza kuondoa baadhi ya sehemu za watu kwenye matibabu.

Sasa, pakia data za X na y.

> 🎓 Kumbuka, hii ni kujifunza kwa usimamizi, na tunahitaji lengo liitwalo 'y'.

Katika seli mpya ya msimbo, pakia dataset ya ugonjwa wa kisukari kwa kuita `load_diabetes()`. Ingizo la `return_X_y=True` linaonyesha kuwa `X` itakuwa matriisi ya data, na `y` itakuwa lengo la regression.

1. Ongeza amri za print kuonyesha umbo la matriisi ya data na kipengele chake cha kwanza:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Kile unachokipokea kama jibu, ni tuple. Unachofanya ni kugawia maadili mawili ya kwanza ya tuple kwa `X` na `y` kwa mtiririko. Jifunze zaidi [kuhusu tuples](https://wikipedia.org/wiki/Tuple).

    Unaona data hii ina vitu 442 vilivyowekwa katika safu za vipengele 10:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Fikiria kidogo kuhusu uhusiano kati ya data na lengo la regression. Linear regression hutabiri uhusiano kati ya kigezo X na kigezo lengo y. Je, unaweza kupata [lengo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) la dataset ya ugonjwa wa kisukari katika nyaraka? Dataset hii inaonyesha nini, ikizingatia lengo hilo?

2. Kisha, chagua sehemu ya dataset hii kuchora kwa kuchagua safu ya 3 ya dataset. Unaweza kufanya hivi kwa kutumia mopereta `:` kuchagua safu zote, kisha kuchagua safu ya 3 kwa kutumia kiindiketa (2). Pia unaweza kuunda tena data kuwa array ya 2D - kama inavyohitajika kwa kuchora - kwa kutumia `reshape(n_rows, n_columns)`. Ikiwa moja ya vigezo ni -1, kipimo kinacholingana huhesabiwa kiotomatiki.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Wakati wowote, chapisha data kuona umbo lake.

3. Sasa unayo data tayari kuchorwa, unaweza kuona kama mashine inaweza kusaidia kubainisha mgawanyiko wa mantiki kati ya nambari katika dataset hii. Kufanya hivi, unahitaji kugawanya data (X) na lengo (y) katika seti za mafunzo na mtihani. Scikit-learn ina njia rahisi ya kufanya hivi; unaweza kugawanya data zako za mtihani mahali fulani.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Sasa uko tayari kufundisha mfano wako! Pakia mfano wa regression ya mstari na ufundishe kwa seti zako za mafunzo za X na y kwa kutumia `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` ni kazi utakayokuta katika maktaba nyingi za ML kama TensorFlow

5. Kisha, tengeneza utabiri kwa kutumia data za mtihani, kwa kutumia kazi ya `predict()`. Hii itatumika kuchora mstari kati ya makundi ya data

    ```python
    y_pred = model.predict(X_test)
    ```

6. Sasa ni wakati wa kuonyesha data katika mchoro. Matplotlib ni zana muhimu sana kwa kazi hii. Unda mchoro wa alama za majaribio ya X na y zote, na tumia utabiri kuchora mstari mahali panapofaa zaidi, kati ya makundi ya data ya mfano.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![mchoro wa alama unaoonyesha pointi za data kuhusu ugonjwa wa kisukari](../../../../translated_images/sw/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Fikiria kidogo kinachotokea hapa. Mstari wa moja kwa moja unaendeshwa kati ya alama ndogo za data nyingi, lakini unafanya nini hasa? Unaona jinsi unavyoweza kutumia mstari huu kutabiri mahali pointi mpya ya data isiyoonekana itafaa kuhusiana na mhimili wa y wa mchoro? Jaribu kuweka maneno matumizi halisi ya mfano huu.

Hongera, umeunda mfano wako wa kwanza wa regression ya mstari, kutengeneza utabiri nao, na kuonyesha kwenye mchoro!

---
## 🚀Changamoto

Chora kigezo tofauti kutoka kwenye dataset hii. Vidokezo: hariri mstari huu: `X = X[:,2]`. Kutokana na lengo la dataset hii, ni nini unaweza kugundua kuhusu maendeleo ya ugonjwa wa kisukari?
## [Mtihani baada ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujifunza binafsi

Katika mafunzo haya, ulifanya kazi na regression rahisi ya mstari, badala ya regression ya mstari moja au mingi. Soma kidogo kuhusu tofauti kati ya mbinu hizi, au tazama [video hii](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Soma zaidi kuhusu dhana ya regression na fikiria aina gani za maswali yanayoweza kujibiwa na mbinu hii. Chukua [mafunzo haya](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) ili kuongeza uelewa wako.

## Kazi

[Seti tofauti ya data](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Kionyozo**:
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kupata usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au upungufu wa usahihi. Hati ya asili katika lugha yake halisi inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu inayofanywa na binadamu inapendekezwa. Hatutojibu kwa kuelewa vibaya au tafsiri potofu zinazotokea kutokana na matumizi ya tafsiri hii.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->