<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T13:19:21+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "my"
}
-->
# အမျိုးအစားခွဲခြားမှုကိုမိတ်ဆက်ခြင်း

ဒီသင်ခန်းစာလေးခုမှာ သင်သည် အခြေခံစက်ရုပ်သင်ယူမှု၏ အရေးပါသောအချက်တစ်ခုဖြစ်သော _အမျိုးအစားခွဲခြားမှု_ ကိုလေ့လာမည်ဖြစ်သည်။ အာရှနှင့်အိန္ဒိယ၏ အံ့ဩဖွယ်အစားအစာများနှင့်ပတ်သက်သောဒေတာအချက်အလက်များကို အသုံးပြု၍ အမျိုးအစားခွဲခြားမှုအယ်လဂိုရစ်သမ်များကို အသုံးပြုခြင်းကို လေ့လာမည်ဖြစ်သည်။ အစားအသောက်အတွက်ဆာလောင်နေတယ်လို့မျှော်လင့်ပါတယ်!

![just a pinch!](../../../../4-Classification/1-Introduction/images/pinch.png)

> ဒီသင်ခန်းစာတွေမှာ အာရှအစားအစာတွေကို ကျေးဇူးတင်ပါ။ ပုံကို [Jen Looper](https://twitter.com/jenlooper) မှဖန်တီးထားသည်။

အမျိုးအစားခွဲခြားမှုသည် [supervised learning](https://wikipedia.org/wiki/Supervised_learning) ၏ အမျိုးအစားတစ်ခုဖြစ်ပြီး regression နည်းလမ်းများနှင့် ဆင်တူသောအချက်များစွာပါရှိသည်။ စက်ရုပ်သင်ယူမှုသည် ဒေတာအချက်အလက်များကို အသုံးပြု၍ တန်ဖိုးများ သို့မဟုတ် အမည်များကို ခန့်မှန်းခြင်းနှင့်ပတ်သက်သည်ဆိုပါက အမျိုးအစားခွဲခြားမှုသည် အဓိကအားဖြင့် _binary classification_ နှင့် _multiclass classification_ ဆိုသောအုပ်စုနှစ်ခုအတွင်းတွင်ပါဝင်သည်။

[![Introduction to classification](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introduction to classification")

> 🎥 အထက်ပါပုံကိုနှိပ်ပြီးဗီဒီယိုကြည့်ပါ။ MIT မှ John Guttag သည် အမျိုးအစားခွဲခြားမှုကိုမိတ်ဆက်သည်။

သတိထားပါ:

- **Linear regression** သည် အပြောင်းအလဲများအကြားဆက်နွယ်မှုများကို ခန့်မှန်းရန်နှင့် အချက်အလက်အသစ်တစ်ခုသည် အဲဒီလိုင်းနှင့်ဆက်နွယ်မှုအတွင်းမှာ ရောက်ရှိမည့်နေရာကို မှန်ကန်စွာခန့်မှန်းရန် ကူညီပေးသည်။ ဥပမာအားဖြင့် _ဖွံ့ဖြိုးမှုအချိန်အလိုက် ဖရဲသီး၏စျေးနှုန်းကို ခန့်မှန်းနိုင်သည်_။
- **Logistic regression** သည် "binary categories" ကို ရှာဖွေရာတွင် ကူညီပေးသည်။ ဥပမာအားဖြင့် _ဤစျေးနှုန်းတွင် ဖရဲသီးသည် လိမ္မော်ရောင်ဖြစ်မည်လား၊ မဖြစ်မည်လား_?

အမျိုးအစားခွဲခြားမှုသည် အချက်အလက်တစ်ခု၏ label သို့မဟုတ် class ကို သတ်မှတ်ရန် အခြားနည်းလမ်းများကို သတ်မှတ်ရန် အယ်လဂိုရစ်သမ်များကို အသုံးပြုသည်။ အစားအစာဒေတာကို အသုံးပြု၍ အဖွဲ့အစည်းတစ်ခု၏ အစိတ်အပိုင်းများကို ကြည့်ရှုခြင်းဖြင့် အစားအစာ၏မူလအမျိုးအစားကို သတ်မှတ်နိုင်မည်လားဆိုတာကို လေ့လာကြည့်ပါ။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [ဒီသင်ခန်းစာကို R မှာလည်းရနိုင်ပါတယ်!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### မိတ်ဆက်

အမျိုးအစားခွဲခြားမှုသည် စက်ရုပ်သင်ယူမှုသုတေသနရှင်နှင့် ဒေတာသိပ္ပံပညာရှင်၏ အခြေခံလုပ်ငန်းများထဲမှတစ်ခုဖြစ်သည်။ binary value ("ဤအီးမေးလ်သည် spam ဖြစ်ပါသလား၊ မဖြစ်ပါသလား") ကို အခြေခံ၍ အမျိုးအစားခွဲခြားမှုမှစ၍ computer vision ကိုအသုံးပြု၍ ရုပ်ပုံခွဲခြားမှုနှင့် segmentation အထိ၊ ဒေတာကို အမျိုးအစားများအလိုက်ခွဲခြားရန်နှင့် မေးခွန်းများမေးရန် အမြဲအသုံးဝင်သည်။

သိပ္ပံပညာဆန်သောနည်းလမ်းဖြင့် ပြောရမည်ဆိုပါက သင်၏အမျိုးအစားခွဲခြားမှုနည်းလမ်းသည် input variables နှင့် output variables အကြားဆက်နွယ်မှုကို map လုပ်ရန် ခန့်မှန်းမှုမော်ဒယ်တစ်ခုကို ဖန်တီးပေးသည်။

![binary vs. multiclass classification](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binary vs. multiclass problems အမျိုးအစားခွဲခြားမှုအယ်လဂိုရစ်သမ်များကို ကိုင်တွယ်ရန်။ Infographic by [Jen Looper](https://twitter.com/jenlooper)

ဒေတာကို သန့်စင်ခြင်း၊ visualization ပြုလုပ်ခြင်းနှင့် ML tasks များအတွက် ပြင်ဆင်ခြင်းလုပ်ငန်းစဉ်ကို စတင်မတိုင်မီ စက်ရုပ်သင်ယူမှုကို အသုံးပြု၍ ဒေတာကို အမျိုးအစားခွဲခြားရန် နည်းလမ်းများအကြောင်းကို နည်းနည်းလေ့လာကြည့်ပါ။

[statistics](https://wikipedia.org/wiki/Statistical_classification) မှ ဆင်းသက်လာသော classic machine learning ကို အသုံးပြု၍ classification သည် `smoker`, `weight`, နှင့် `age` ကဲ့သို့သော features များကို အသုံးပြု၍ _X ရောဂါဖြစ်ပွားနိုင်မှု_ ကို သတ်မှတ်ပေးသည်။ သင်မကြာသေးမီက လုပ်ဆောင်ခဲ့သော regression လေ့ကျင့်ခန်းများနှင့် ဆင်တူသော supervised learning နည်းလမ်းတစ်ခုအဖြစ် သင်၏ဒေတာသည် label လုပ်ထားပြီး ML အယ်လဂိုရစ်သမ်များသည် အဲဒီ label များကို အသုံးပြု၍ ဒေတာအချက်အလက်များ၏ အမျိုးအစားများ (သို့မဟုတ် 'features') ကို ခွဲခြားရန်နှင့် အုပ်စု သို့မဟုတ် ရလဒ်တစ်ခုသို့ သတ်မှတ်ရန် ကူညီပေးသည်။

✅ အစားအစာများနှင့်ပတ်သက်သောဒေတာအချက်အလက်တစ်ခုကို စိတ်ကူးကြည့်ပါ။ multiclass မော်ဒယ်သည် ဘာကိုဖြေရှင်းနိုင်မည်လဲ? binary မော်ဒယ်သည် ဘာကိုဖြေရှင်းနိုင်မည်လဲ? ဥပမာအားဖြင့် အစားအစာတစ်ခုသည် fenugreek ကို အသုံးပြုမည်လားမသုံးမည်လားဆိုတာကို သတ်မှတ်နိုင်မည်လား? သို့မဟုတ် star anise, artichokes, cauliflower, နှင့် horseradish တို့ပါဝင်သော grocery bag တစ်ခုကို ပေးလှူခဲ့ပါက အိန္ဒိယအစားအစာတစ်မျိုးကို ဖန်တီးနိုင်မည်လား?

[![Crazy mystery baskets](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Crazy mystery baskets")

> 🎥 အထက်ပါပုံကိုနှိပ်ပြီးဗီဒီယိုကြည့်ပါ။ 'Chopped' ဆိုသောရုပ်ရှင်၏ အဓိကအကြောင်းအရာမှာ 'mystery basket' ဖြစ်ပြီး ချက်ပြုတ်သူများသည် အလွတ်ရွေးချယ်ထားသောအစားအစာများကို အသုံးပြု၍ အစားအစာတစ်မျိုးကို ပြုလုပ်ရမည်။ ML မော်ဒယ်တစ်ခုက အကူအညီပေးနိုင်မည်ဖြစ်သည်။

## Hello 'classifier'

ဤအစားအစာဒေတာအချက်အလက်ကို သင်မေးလိုသောမေးခွန်းသည် **multiclass question** ဖြစ်သည်။ အကြောင်းမှာ အမျိုးအစားများစွာရှိပြီး အစားအစာအစုတစ်ခုသည် အဲဒီအမျိုးအစားများထဲမှ မည်သည့်အမျိုးအစားနှင့် ကိုက်ညီမည်လဲဆိုတာကို သတ်မှတ်ရန်လိုအပ်သည်။

Scikit-learn သည် အမျိုးအစားခွဲခြားရန် သင်လိုအပ်သောပြဿနာအမျိုးအစားပေါ်မူတည်၍ အမျိုးမျိုးသောအယ်လဂိုရစ်သမ်များကို ပေးသည်။ နောက်ထပ်သင်ခန်းစာနှစ်ခုတွင် သင်သည် အယ်လဂိုရစ်သမ်များအကြောင်းကို လေ့လာမည်ဖြစ်သည်။

## လေ့ကျင့်ခန်း - သင်၏ဒေတာကို သန့်စင်ပြီး balance လုပ်ပါ

ဤပရောဂျက်ကို စတင်မတိုင်မီ ပထမဦးဆုံးလုပ်ဆောင်ရမည့်အလုပ်မှာ သင်၏ဒေတာကို သန့်စင်ပြီး **balance** လုပ်ခြင်းဖြစ်သည်။ ဤဖိုလ်ဒါ၏ root တွင်ရှိသော blank _notebook.ipynb_ ဖိုင်ကို စတင်ပါ။

ပထမဦးဆုံး install လုပ်ရမည့်အရာမှာ [imblearn](https://imbalanced-learn.org/stable/) ဖြစ်သည်။ ၎င်းသည် Scikit-learn package တစ်ခုဖြစ်ပြီး ဒေတာကို ပိုမို balance လုပ်ရန် ကူညီပေးမည် (ဤအလုပ်ကို နည်းနည်းအကြောင်းသိရှိမည်)။

1. `imblearn` ကို install လုပ်ရန် `pip install` ကို အောက်ပါအတိုင်း run လုပ်ပါ:

    ```python
    pip install imblearn
    ```

1. သင်၏ဒေတာကို import လုပ်ရန်လိုအပ်သော packages များကို import လုပ်ပြီး visualize လုပ်ပါ၊ `imblearn` မှ `SMOTE` ကိုလည်း import လုပ်ပါ။

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    ယခု သင်သည် ဒေတာကို import လုပ်ရန် ပြင်ဆင်ပြီးဖြစ်သည်။

1. နောက်တစ်ခုလုပ်ဆောင်ရမည့်အလုပ်မှာ ဒေတာကို import လုပ်ခြင်းဖြစ်သည်:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   `read_csv()` ကို အသုံးပြု၍ _cusines.csv_ ဆိုသော csv ဖိုင်၏ content ကို ဖတ်ပြီး `df` variable ထဲသို့ ထည့်သွင်းပါ။

1. ဒေတာ၏ shape ကို စစ်ဆေးပါ:

    ```python
    df.head()
    ```

   ပထမဦးဆုံး rows ၅ ခုသည် အောက်ပါအတိုင်းဖြစ်သည်:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. ဤဒေတာအကြောင်းကို `info()` ကိုခေါ်၍ သိရှိပါ:

    ```python
    df.info()
    ```

    သင်၏ output သည် အောက်ပါအတိုင်းဖြစ်သည်:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## လေ့ကျင့်ခန်း - အစားအစာအမျိုးအစားများကိုလေ့လာခြင်း

ယခုအလုပ်သည် ပိုမိုစိတ်ဝင်စားဖွယ်ဖြစ်လာသည်။ ဒေတာ၏ distribution ကို အမျိုးအစားအလိုက် ရှာဖွေကြည့်ပါ။

1. `barh()` ကိုခေါ်၍ ဒေတာကို bars အဖြစ် plot လုပ်ပါ:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![cuisine data distribution](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    အမျိုးအစားများသည် အကန့်အသတ်ရှိသော်လည်း ဒေတာ၏ distribution သည် မညီမျှပါ။ သင်သည် အဲဒီကို ပြင်ဆင်နိုင်သည်! ပြင်ဆင်မတိုင်မီ နည်းနည်းလေ့လာပါ။

1. အမျိုးအစားအလိုက် ရရှိနိုင်သောဒေတာပမာဏကို ရှာဖွေပြီး print ထုတ်ပါ:

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

    output သည် အောက်ပါအတိုင်းဖြစ်သည်:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## အစားအစာပစ္စည်းများကိုရှာဖွေခြင်း

ယခု သင်သည် ဒေတာကို နက်ရှိုင်းစွာရှာဖွေပြီး အမျိုးအစားအလိုက် အစားအစာပစ္စည်းများကို သိရှိနိုင်သည်။ အမျိုးအစားများအကြား ရှုပ်ထွေးမှုကို ဖြစ်စေသော ထပ်တူဖြစ်သောဒေတာများကို ဖယ်ရှားရန်လိုအပ်သည်၊ ထို့ကြောင့်ဤပြဿနာအကြောင်းကို လေ့လာကြည့်ပါ။

1. Python တွင် `create_ingredient()` ဆိုသော function တစ်ခုကို ဖန်တီးပါ။ ဤ function သည် မအသုံးဝင်သော column တစ်ခုကို drop လုပ်ပြီး အစားအစာပစ္စည်းများကို count အလိုက် sort လုပ်ပါမည်:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   ယခု သင်သည် function ကို အသုံးပြု၍ အမျိုးအစားအလိုက် အစားအစာပစ္စည်းများ၏ ထိပ်ဆုံး ၁၀ ခုကို သိရှိနိုင်သည်။

1. `create_ingredient()` ကိုခေါ်ပြီး `barh()` ကိုခေါ်၍ plot လုပ်ပါ:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. ဂျပန်ဒေတာအတွက်လည်း အလားတူလုပ်ဆောင်ပါ:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. ယခုတစ်ခါ တရုတ်အစားအစာပစ္စည်းများအတွက်:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. အိန္ဒိယအစားအစာပစ္စည်းများကို plot လုပ်ပါ:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. နောက်ဆုံးတွင် ကိုရီးယားအစားအစာပစ္စည်းများကို plot လုပ်ပါ:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. အမျိုးအစားများအကြား ရှုပ်ထွေးမှုကို ဖြစ်စေသော အစားအစာပစ္စည်းများကို drop လုပ်ပါ။ 

   အားလုံး rice, garlic, နှင့် ginger ကိုချစ်ကြသည်!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## ဒေတာကို balance လုပ်ပါ

ယခု သင်သည် ဒေတာကို သန့်စင်ပြီးဖြစ်သည်၊ [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - ကို အသုံးပြု၍ balance လုပ်ပါ။

1. `fit_resample()` ကိုခေါ်ပါ၊ ဤနည်းလမ်းသည် interpolation ဖြင့် sample အသစ်များကို ဖန်တီးပေးသည်။

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    သင်၏ဒေတာကို balance လုပ်ခြင်းဖြင့် classification လုပ်ရာတွင် ပိုမိုကောင်းမွန်သောရလဒ်များရရှိမည်ဖြစ်သည်။ binary classification ကိုစဉ်းစားပါ။ သင်၏ဒေတာအများစုသည် class တစ်ခုဖြစ်ပါက ML မော်ဒယ်သည် အဲဒီ class ကို ပိုမိုခန့်မှန်းမည်ဖြစ်သည်၊ အကြောင်းမှာ အဲဒီ class အတွက် ဒေတာများပိုမိုရှိသောကြောင့်ဖြစ်သည်။ ဒေတာကို balance လုပ်ခြင်းသည် skewed data များကို ဖယ်ရှားပြီး ဤမညီမျှမှုကို ဖြေရှင်းပေးသည်။

1. ယခု သင်သည် label များ၏ numbers ကို ingredient အလိုက် စစ်ဆေးနိုင်သည်:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    သင်၏ output သည် အောက်ပါအတိုင်းဖြစ်သည်:

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

    ဒေတာသည် သန့်စင်ပြီး balance လုပ်ထားပြီး အရသာရှိသောအခြေအနေတွင်ရှိသည်! 

1. နောက်ဆုံးအဆင့်မှာ label များနှင့် features များပါဝင်သော balanced data ကို အသစ်သော dataframe ထဲသို့ သိမ်းဆည်းပြီး ဖိုင်အဖြစ် export လုပ်ရန်ဖြစ်သည်:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. `transformed_df.head()` နှင့် `transformed_df.info()` ကို အသုံးပြု၍ ဒေတာကို နောက်တစ်ကြိမ်ကြည့်ပါ။ ဤဒေတာကို အနာဂတ်သင်ခန်းစာများတွင် အသုံးပြုရန်အတွက် copy တစ်ခု save လုပ်ပါ:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    ဤအသစ်သော CSV ကို ယခု root data folder တွင်တွေ့နိုင်ပါသည်။

---

## 🚀Challenge

ဤသင်ခန်းစာတွင် စိတ်ဝင်စားဖွယ်ဒေတာအချက်အလက်များစွာပါဝင်သည်။ `data` folder များကို ရှာဖွေကြည့်ပြီး binary classification သို့မဟုတ် multi-class classification အတွက် သင့်လျော်သောဒေတာအချက်အလက်များပါဝင်ပါသလားဆိုတာကို စစ်ဆေးပါ။ ဤဒေတာအချက်အလက်များကို သင်မေးလိုသောမေးခွန်းများက ဘာတွေလဲ?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

SMOTE ၏ API ကိုလေ့လာပါ။ ၎င်းကို အသုံးပြုရန်အကောင်း

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မှန်ကန်မှုမရှိသောအချက်များ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူသားပညာရှင်များမှ ဘာသာပြန်ဆိုမှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ဆိုမှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပါယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။