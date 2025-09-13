<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T16:25:15+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "sw"
}
-->
# Utangulizi wa Uainishaji

Katika masomo haya manne, utachunguza kipengele muhimu cha ujifunzaji wa mashine wa kawaida - _uainishaji_. Tutapitia matumizi ya algoriti mbalimbali za uainishaji kwa kutumia seti ya data kuhusu vyakula vya kupendeza vya Asia na India. Tunatumai una njaa!

![just a pinch!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Sherehekea vyakula vya pan-Asia katika masomo haya! Picha na [Jen Looper](https://twitter.com/jenlooper)

Uainishaji ni aina ya [ujifunzaji unaosimamiwa](https://wikipedia.org/wiki/Supervised_learning) ambao una mfanano mkubwa na mbinu za regression. Ikiwa ujifunzaji wa mashine unahusu kutabiri thamani au majina ya vitu kwa kutumia seti za data, basi uainishaji kwa ujumla huangukia katika makundi mawili: _uainishaji wa binary_ na _uainishaji wa darasa nyingi_.

[![Introduction to classification](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introduction to classification")

> 🎥 Bofya picha hapo juu kwa video: John Guttag wa MIT anatambulisha uainishaji

Kumbuka:

- **Linear regression** ilikusaidia kutabiri uhusiano kati ya vigezo na kufanya utabiri sahihi kuhusu mahali ambapo data mpya ingeangukia kwa uhusiano na mstari huo. Kwa mfano, ungeweza kutabiri _bei ya malenge itakuwa kiasi gani mwezi wa Septemba dhidi ya Desemba_.
- **Logistic regression** ilikusaidia kugundua "makundi ya binary": kwa kiwango hiki cha bei, _je, malenge hili ni la rangi ya machungwa au si la machungwa_?

Uainishaji hutumia algoriti mbalimbali kuamua njia nyingine za kuainisha lebo au darasa la data. Hebu tufanye kazi na data hii ya vyakula ili kuona kama, kwa kuangalia kikundi cha viungo, tunaweza kuamua asili ya vyakula hivyo.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somo hili linapatikana kwa R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Utangulizi

Uainishaji ni mojawapo ya shughuli za msingi za mtafiti wa ujifunzaji wa mashine na mwanasayansi wa data. Kuanzia uainishaji wa msingi wa thamani ya binary ("je, barua pepe hii ni spam au si spam?"), hadi uainishaji wa picha na kugawanya kwa kutumia maono ya kompyuta, daima ni muhimu kuweza kupanga data katika madarasa na kuuliza maswali kuhusu data hiyo.

Kwa kusema mchakato kwa njia ya kisayansi zaidi, mbinu yako ya uainishaji huunda mfano wa utabiri unaokuwezesha kuonyesha uhusiano kati ya vigezo vya ingizo na vigezo vya matokeo.

![binary vs. multiclass classification](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Masuala ya binary dhidi ya darasa nyingi kwa algoriti za uainishaji kushughulikia. Infographic na [Jen Looper](https://twitter.com/jenlooper)

Kabla ya kuanza mchakato wa kusafisha data yetu, kuiona, na kujiandaa kwa kazi zetu za ML, hebu tujifunze kidogo kuhusu njia mbalimbali ambazo ujifunzaji wa mashine unaweza kutumika kuainisha data.

Ikitokana na [takwimu](https://wikipedia.org/wiki/Statistical_classification), uainishaji kwa kutumia ujifunzaji wa mashine wa kawaida hutumia vipengele, kama `smoker`, `weight`, na `age` kuamua _uwezekano wa kupata ugonjwa X_. Kama mbinu ya ujifunzaji unaosimamiwa inayofanana na mazoezi ya regression uliyofanya awali, data yako ina lebo na algoriti za ML hutumia lebo hizo kuainisha na kutabiri madarasa (au 'vipengele') vya seti ya data na kuzipangia kundi au matokeo.

✅ Chukua muda kufikiria seti ya data kuhusu vyakula. Je, mfano wa darasa nyingi ungeweza kujibu nini? Je, mfano wa binary ungeweza kujibu nini? Je, ungependa kuamua kama chakula fulani kina uwezekano wa kutumia fenugreek? Je, ungependa kuona kama, ukipokea zawadi ya mfuko wa mboga uliojaa star anise, artichokes, cauliflower, na horseradish, ungeweza kuunda sahani ya kawaida ya Kihindi?

[![Crazy mystery baskets](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Crazy mystery baskets")

> 🎥 Bofya picha hapo juu kwa video. Premisi nzima ya kipindi 'Chopped' ni 'mystery basket' ambapo wapishi wanapaswa kutengeneza sahani kutoka kwa chaguo la viungo vya nasibu. Hakika mfano wa ML ungeweza kusaidia!

## Habari 'classifier'

Swali tunalotaka kuuliza kuhusu seti ya data ya vyakula ni swali la **darasa nyingi**, kwani tuna vyakula vya kitaifa kadhaa vya kufanya kazi navyo. Ukipewa kundi la viungo, ni darasa gani kati ya haya mengi data itafaa?

Scikit-learn inatoa algoriti kadhaa tofauti za kutumia kuainisha data, kulingana na aina ya tatizo unalotaka kutatua. Katika masomo mawili yajayo, utajifunza kuhusu algoriti kadhaa kati ya hizi.

## Zoezi - safisha na uratibu data yako

Kazi ya kwanza, kabla ya kuanza mradi huu, ni kusafisha na **kuratibu** data yako ili kupata matokeo bora. Anza na faili tupu _notebook.ipynb_ katika mzizi wa folda hii.

Jambo la kwanza kusakinisha ni [imblearn](https://imbalanced-learn.org/stable/). Hii ni kifurushi cha Scikit-learn ambacho kitakuruhusu kuratibu data vizuri zaidi (utajifunza zaidi kuhusu kazi hii kwa muda mfupi).

1. Ili kusakinisha `imblearn`, endesha `pip install`, kama ifuatavyo:

    ```python
    pip install imblearn
    ```

1. Ingiza vifurushi unavyohitaji kuingiza data yako na kuiona, pia ingiza `SMOTE` kutoka `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Sasa umejiandaa kusoma na kuingiza data inayofuata.

1. Kazi inayofuata itakuwa kuingiza data:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Kutumia `read_csv()` kutasoma maudhui ya faili ya csv _cusines.csv_ na kuiweka katika kigezo `df`.

1. Angalia umbo la data:

    ```python
    df.head()
    ```

   Safu tano za kwanza zinaonekana kama hivi:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Pata maelezo kuhusu data hii kwa kuita `info()`:

    ```python
    df.info()
    ```

    Matokeo yako yanafanana na:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Zoezi - kujifunza kuhusu vyakula

Sasa kazi inaanza kuwa ya kuvutia zaidi. Hebu tujifunze kuhusu usambazaji wa data, kwa kila aina ya chakula.

1. Chora data kama baa kwa kuita `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![cuisine data distribution](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Kuna idadi ndogo ya vyakula, lakini usambazaji wa data hauko sawa. Unaweza kurekebisha hilo! Kabla ya kufanya hivyo, chunguza kidogo zaidi.

1. Tafuta ni kiasi gani cha data kinapatikana kwa kila aina ya chakula na uichapishe:

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

    matokeo yanaonekana kama hivi:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Kugundua viungo

Sasa unaweza kuchimba zaidi katika data na kujifunza ni viungo gani vya kawaida kwa kila aina ya chakula. Unapaswa kusafisha data inayojirudia ambayo inasababisha mkanganyiko kati ya vyakula, kwa hivyo hebu tujifunze kuhusu tatizo hili.

1. Unda kazi `create_ingredient()` katika Python ili kuunda fremu ya data ya viungo. Kazi hii itaanza kwa kuondoa safu isiyo ya msaada na kuchagua viungo kulingana na idadi yao:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Sasa unaweza kutumia kazi hiyo kupata wazo la viungo kumi maarufu zaidi kwa kila aina ya chakula.

1. Ita `create_ingredient()` na uchore kwa kuita `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Fanya vivyo hivyo kwa data ya Kijapani:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Sasa kwa viungo vya Kichina:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Chora viungo vya Kihindi:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Hatimaye, chora viungo vya Kikorea:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Sasa, ondoa viungo vya kawaida vinavyosababisha mkanganyiko kati ya vyakula tofauti, kwa kuita `drop()`:

   Kila mtu anapenda mchele, vitunguu saumu na tangawizi!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Ratibu seti ya data

Sasa kwa kuwa umesafisha data, tumia [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Mbinu ya Kuongeza Sampuli za Wachache kwa Njia ya Kijumlisha" - kuiratibu.

1. Ita `fit_resample()`, mkakati huu huzalisha sampuli mpya kwa njia ya uingiliaji.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Kwa kuratibu data yako, utapata matokeo bora wakati wa kuainisha. Fikiria kuhusu uainishaji wa binary. Ikiwa data yako nyingi ni ya darasa moja, mfano wa ML utatabiri darasa hilo mara nyingi zaidi, kwa sababu kuna data zaidi kwa ajili yake. Kuratibu data huchukua data iliyopotoshwa na husaidia kuondoa upotoshaji huu.

1. Sasa unaweza kuangalia idadi ya lebo kwa kila kiungo:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Matokeo yako yanaonekana kama hivi:

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

    Data ni safi, imeratibiwa, na ni tamu sana!

1. Hatua ya mwisho ni kuhifadhi data yako iliyoratibiwa, ikiwa ni pamoja na lebo na vipengele, katika fremu mpya ya data ambayo inaweza kusafirishwa kwenye faili:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Unaweza kuchukua muonekano mmoja zaidi wa data kwa kutumia `transformed_df.head()` na `transformed_df.info()`. Hifadhi nakala ya data hii kwa matumizi katika masomo ya baadaye:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    CSV hii mpya sasa inaweza kupatikana katika folda ya data ya mzizi.

---

## 🚀Changamoto

Mtaala huu una seti kadhaa za data za kuvutia. Chunguza folda za `data` na uone kama kuna yoyote inayojumuisha seti za data zinazofaa kwa uainishaji wa binary au darasa nyingi? Maswali gani ungeuliza kuhusu seti ya data hiyo?

## [Maswali ya baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Chunguza API ya SMOTE. Inafaa kutumika kwa kesi gani? Inatatua matatizo gani?

## Kazi 

[Chunguza mbinu za uainishaji](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.