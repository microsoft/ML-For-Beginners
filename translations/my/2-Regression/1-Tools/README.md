<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T11:43:05+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "my"
}
-->
# Python နှင့် Scikit-learn ကို အသုံးပြု၍ Regression Models တည်ဆောက်ခြင်း

![Regression များ၏ အကျဉ်းချုပ်ကို Sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [ဒီသင်ခန်းစာကို R မှာလည်းရနိုင်ပါတယ်!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## အကျဉ်းချုပ်

ဒီသင်ခန်းစာလေးများမှာ Regression Models တည်ဆောက်ပုံကို လေ့လာနိုင်ပါမယ်။ Regression Models သည် ဘာအတွက်အသုံးဝင်သလဲဆိုတာကို မကြာမီဆွေးနွေးသွားပါမယ်။ သို့သော် စတင်လုပ်ဆောင်ရန်အတွက် သင့်စက်မှာ လိုအပ်သော Tools များကို အရင်ဆုံး ပြင်ဆင်ထားဖို့ လိုအပ်ပါတယ်။

ဒီသင်ခန်းစာမှာ သင်လေ့လာနိုင်မယ့်အရာများမှာ -

- သင့်ကွန်ပျူတာကို Local Machine Learning Tasks အတွက် ပြင်ဆင်ခြင်း။
- Jupyter Notebooks ကို အသုံးပြုခြင်း။
- Scikit-learn ကို အသုံးပြုခြင်း (installation အပါအဝင်)။
- Linear Regression ကို လက်တွေ့လုပ်ဆောင်ခြင်း။

## Installations နှင့် Configurations

[![ML for beginners - Machine Learning Models တည်ဆောက်ရန် Tools များကို ပြင်ဆင်ခြင်း](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML for beginners - Machine Learning Models တည်ဆောက်ရန် Tools များကို ပြင်ဆင်ခြင်း")

> 🎥 အထက်ပါပုံကို Click လုပ်ပြီး ML အတွက် သင့်ကွန်ပျူတာကို Configure လုပ်ပုံကို ကြည့်ပါ။

1. **Python ကို Install လုပ်ပါ**။ သင့်ကွန်ပျူတာမှာ [Python](https://www.python.org/downloads/) ကို Install လုပ်ထားရှိရပါမယ်။ Python ကို Data Science နှင့် Machine Learning Tasks များအတွက် အသုံးပြုပါမယ်။ အများစုသော ကွန်ပျူတာစနစ်များမှာ Python ကို အရင်ကတည်းက Install လုပ်ထားပြီးဖြစ်ပါတယ်။ [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) များလည်း ရနိုင်ပြီး၊ အချို့သောအသုံးပြုသူများအတွက် Setup ကို ပိုမိုလွယ်ကူစေပါတယ်။

   သို့သော် Python ကို အသုံးပြုခြင်းမှာ Version များကွဲပြားမှုရှိနိုင်ပါတယ်။ အချို့သော Tasks များအတွက် Version တစ်ခုလိုအပ်ပြီး၊ အခြား Tasks များအတွက် Version တစ်ခုလိုအပ်နိုင်ပါတယ်။ ဒီအကြောင်းကြောင့် [Virtual Environment](https://docs.python.org/3/library/venv.html) တွင် အလုပ်လုပ်ခြင်းသည် အကျိုးရှိပါတယ်။

2. **Visual Studio Code ကို Install လုပ်ပါ**။ သင့်ကွန်ပျူတာမှာ Visual Studio Code ကို Install လုပ်ထားရှိရပါမယ်။ [Visual Studio Code](https://code.visualstudio.com/) ကို Install လုပ်ပုံအဆင့်ဆင့်ကို လိုက်နာပါ။ ဒီသင်ခန်းစာမှာ Python ကို Visual Studio Code မှာ အသုံးပြုမယ်၊ [Visual Studio Code ကို Python Development အတွက် Configure လုပ်ပုံ](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) ကိုလည်း လေ့လာပါ။

   > Python ကို အသုံးပြုရင်း ကျွမ်းကျင်စေဖို့ ဒီ [Learn modules](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) များကို လေ့လာပါ။
   >
   > [![Visual Studio Code မှာ Python ကို Setup လုပ်ခြင်း](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Visual Studio Code မှာ Python ကို Setup လုပ်ခြင်း")
   >
   > 🎥 အထက်ပါပုံကို Click လုပ်ပြီး VS Code မှာ Python ကို အသုံးပြုပုံကို ကြည့်ပါ။

3. **Scikit-learn ကို Install လုပ်ပါ**။ [ဒီအညွှန်းများ](https://scikit-learn.org/stable/install.html) ကို လိုက်နာပြီး Install လုပ်ပါ။ Python 3 ကို အသုံးပြုရမယ်ဆိုတာ သေချာစေဖို့ Virtual Environment ကို အသုံးပြုရန် အကြံပြုပါတယ်။ M1 Mac မှာ Library ကို Install လုပ်ရင် အထူးအညွှန်းများရှိပါတယ်။

4. **Jupyter Notebook ကို Install လုပ်ပါ**။ [Jupyter package](https://pypi.org/project/jupyter/) ကို Install လုပ်ပါ။

## သင့် ML Authoring Environment

သင် **notebooks** ကို အသုံးပြုပြီး Python Code တွေကို Develop လုပ်ပြီး Machine Learning Models တွေကို တည်ဆောက်ပါမယ်။ ဒီအမျိုးအစားဖိုင်တွေဟာ Data Scientists တွေအတွက် အများဆုံးအသုံးပြုတဲ့ Tools ဖြစ်ပြီး `.ipynb` extension နဲ့ ဖိုင်တွေကို မှတ်သားနိုင်ပါတယ်။

Notebooks တွေဟာ Interactive Environment ဖြစ်ပြီး Developer တွေကို Code ရေးခြင်း၊ Notes ထည့်ခြင်း၊ Documentation ရေးခြင်း စတဲ့ အလုပ်တွေကို လုပ်နိုင်စေပါတယ်။ အထူးသဖြင့် Experimental သို့မဟုတ် Research-oriented Projects တွေအတွက် အထောက်အကူဖြစ်ပါတယ်။

[![ML for beginners - Jupyter Notebooks ကို Setup လုပ်ပြီး Regression Models တည်ဆောက်ခြင်း](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML for beginners - Jupyter Notebooks ကို Setup လုပ်ပြီး Regression Models တည်ဆောက်ခြင်း")

> 🎥 အထက်ပါပုံကို Click လုပ်ပြီး ဒီအလုပ်ကို လုပ်ဆောင်ပုံကို ကြည့်ပါ။

### လက်တွေ့လုပ်ဆောင်မှု - Notebook ကို အသုံးပြုခြင်း

ဒီ Folder မှာ _notebook.ipynb_ ဖိုင်ကို တွေ့နိုင်ပါမယ်။

1. _notebook.ipynb_ ကို Visual Studio Code မှာ ဖွင့်ပါ။

   Jupyter Server တစ်ခု Python 3+ နဲ့ စတင်ပါမယ်။ Notebook မှာ `run` လို့ရတဲ့ Code Block တွေကို တွေ့နိုင်ပါမယ်။ Code Block တစ်ခုကို Run လုပ်ဖို့ Play Button ပုံစံ Icon ကို ရွေးပါ။

2. `md` icon ကို ရွေးပြီး Markdown အနည်းငယ်ထည့်ပါ၊ **# Welcome to your notebook** ဆိုတဲ့ စာသားကို ထည့်ပါ။

   နောက်တစ်ဆင့်မှာ Python Code ကို ထည့်ပါ။

3. Code Block မှာ **print('hello notebook')** ကို ရိုက်ပါ။
4. Arrow ကို ရွေးပြီး Code ကို Run လုပ်ပါ။

   Printed Statement ကို တွေ့ရပါမယ် -

    ```output
    hello notebook
    ```

![VS Code မှာ Notebook ဖွင့်ထားပုံ](../../../../2-Regression/1-Tools/images/notebook.jpg)

သင့် Code ကို Comments တွေနဲ့ ပေါင်းစပ်ပြီး Notebook ကို Self-document လုပ်နိုင်ပါတယ်။

✅ Web Developer ရဲ့ အလုပ်လုပ်ပုံနဲ့ Data Scientist ရဲ့ အလုပ်လုပ်ပုံက ဘယ်လိုကွာခြားနေလဲဆိုတာ အနည်းငယ်တွေးကြည့်ပါ။

## Scikit-learn ကို အသုံးပြုခြင်း

Python ကို Local Environment မှာ Setup လုပ်ပြီး၊ Jupyter Notebooks ကို အသုံးပြုရင်း ကျွမ်းကျင်လာပြီးနောက်၊ Scikit-learn ကိုလည်း ကျွမ်းကျင်စေဖို့ လိုအပ်ပါတယ်။ Scikit-learn ကို `sci` (science) လို့ အသံထွက်ပါ။ Scikit-learn မှာ ML Tasks တွေကို လုပ်ဆောင်ဖို့ [အကျယ်အဝန်း API](https://scikit-learn.org/stable/modules/classes.html#api-ref) ရှိပါတယ်။

သူတို့ရဲ့ [website](https://scikit-learn.org/stable/getting_started.html) အရ - "Scikit-learn သည် supervised learning နှင့် unsupervised learning ကို ပံ့ပိုးပေးတဲ့ open source machine learning library ဖြစ်ပါတယ်။ Model fitting, data preprocessing, model selection နှင့် evaluation အပါအဝင် အခြားသော Utilities များစွာကိုလည်း ပံ့ပိုးပေးပါတယ်။"

ဒီသင်ခန်းစာမှာ Scikit-learn နှင့် အခြား Tools များကို အသုံးပြုပြီး 'traditional machine learning' tasks တွေကို လုပ်ဆောင်ပါမယ်။ Neural Networks နှင့် Deep Learning ကို မပါဝင်စေဖို့ ရည်ရွယ်ထားပြီး၊ အဲဒီအကြောင်းအရာတွေကို 'AI for Beginners' curriculum မှာ ပိုမိုလေ့လာနိုင်ပါမယ်။

Scikit-learn သည် Models တွေကို တည်ဆောက်ပြီး အသုံးပြုဖို့ အလွယ်ကူဆုံးဖြစ်စေပါတယ်။ Numeric Data ကို အဓိကထားပြီး Learning Tools အဖြစ် အသုံးပြုနိုင်တဲ့ Dataset များစွာပါဝင်ပါတယ်။ Pre-built Models တွေကိုလည်း Students တွေ စမ်းသပ်နိုင်ပါတယ်။ Prepackaged Data ကို Load လုပ်ပြီး Built-in Estimator ကို အသုံးပြုတဲ့ ပထမဆုံး ML Model ကို လေ့လာကြည့်ပါမယ်။

## လက်တွေ့လုပ်ဆောင်မှု - ပထမဆုံး Scikit-learn Notebook

> ဒီ Tutorial ကို Scikit-learn ရဲ့ [linear regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) မှာ အခြေခံပြီး ရေးသားထားပါတယ်။

[![ML for beginners - Python မှာ ပထမဆုံး Linear Regression Project](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML for beginners - Python မှာ ပထမဆုံး Linear Regression Project")

> 🎥 အထက်ပါပုံကို Click လုပ်ပြီး ဒီအလုပ်ကို လုပ်ဆောင်ပုံကို ကြည့်ပါ။

_notebook.ipynb_ ဖိုင်မှာရှိတဲ့ Cell တွေကို 'trash can' icon ကို နှိပ်ပြီး ရှင်းလင်းပါ။

ဒီအပိုင်းမှာ Scikit-learn မှာ Learning အတွက် Built-in ဖြစ်တဲ့ Diabetes Dataset ကို အသုံးပြုပါမယ်။ Diabetic Patients တွေအတွက် Treatment ကို စမ်းသပ်ချင်တယ်လို့ စဉ်းစားပါ။ Machine Learning Models တွေက Variables တွေကို အခြေခံပြီး ဘယ်သူတွေ Treatment ကို ပိုမိုတုံ့ပြန်နိုင်မလဲဆိုတာကို သတ်မှတ်ပေးနိုင်ပါတယ်။ Visualization လုပ်ထားတဲ့ Basic Regression Model တစ်ခုက Variables တွေကို သုံးပြီး Clinical Trials တွေကို စီမံခန့်ခွဲဖို့ အထောက်အကူဖြစ်စေမယ်။

✅ Regression Methods များစွာရှိပြီး၊ သင်ရွေးချယ်ရမယ့် Method က သင်လိုချင်တဲ့ အဖြေကို အခြေခံပါတယ်။ လူတစ်ဦးရဲ့ အသက်အရွယ်ကို အခြေခံပြီး အမြင့်ကို ခန့်မှန်းချင်တယ်ဆို Linear Regression ကို အသုံးပြုပါမယ်၊ အကြောင်းက Numeric Value ကို ရှာဖွေနေပါတယ်။ အစားအစာအမျိုးအစားတစ်ခုကို Vegan ဖြစ်/မဖြစ် သတ်မှတ်ချင်တယ်ဆို Logistic Regression ကို အသုံးပြုပါမယ်၊ အကြောင်းက Category Assignment ကို ရှာဖွေနေပါတယ်။ Logistic Regression ကို နောက်ပိုင်းမှာ ပိုမိုလေ့လာနိုင်ပါမယ်။ Data ကို အခြေခံပြီး မေးခွန်းများစွာကို တွေးကြည့်ပြီး၊ ဘယ် Method က ပိုသင့်တော်မလဲဆိုတာ တွေးကြည့်ပါ။

အလုပ်ကို စတင်လိုက်ပါ။

### Libraries များ Import လုပ်ခြင်း

ဒီအလုပ်အတွက် Libraries အချို့ကို Import လုပ်ပါမယ် -

- **matplotlib**. [Graphing Tool](https://matplotlib.org/) အဖြစ် အသုံးဝင်ပြီး Line Plot တစ်ခုကို ဖန်တီးဖို့ အသုံးပြုပါမယ်။
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) သည် Python မှာ Numeric Data ကို Handle လုပ်ဖို့ အသုံးဝင်တဲ့ Library ဖြစ်ပါတယ်။
- **sklearn**. [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) Library ဖြစ်ပါတယ်။

သင့် Tasks များအတွက် အထောက်အကူဖြစ်စေမယ့် Libraries များကို Import လုပ်ပါ။

1. အောက်ပါ Code ကို ရိုက်ပါ -

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   အထက်မှာ `matplotlib`, `numpy` ကို Import လုပ်ပြီး၊ `datasets`, `linear_model` နှင့် `model_selection` ကို `sklearn` မှ Import လုပ်ထားပါတယ်။ `model_selection` ကို Data ကို Training နှင့် Test Sets အဖြစ် ခွဲခြားဖို့ အသုံးပြုပါတယ်။

### Diabetes Dataset

Built-in [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) မှာ Diabetes နှင့် ပတ်သက်တဲ့ Data Samples 442 ခုပါဝင်ပြီး Feature Variables 10 ခုပါဝင်ပါတယ်၊ အချို့မှာ -

- age: အသက် (နှစ်)
- bmi: Body Mass Index
- bp: ပျမ်းမျှ သွေးပေါင်ချိန်
- s1 tc: T-Cells (အဖြူရောင် သွေးဆဲလ်အမျိုးအစား)

✅ ဒီ Dataset မှာ Diabetes နှင့် ပတ်သက်တဲ့ Research အတွက် Feature Variable အဖြစ် 'sex' ကိုပါဝင်ထားပါတယ်။ Medical Datasets များစွာမှာ ဒီလို Binary Classification ပါဝင်ပါတယ်။ Population တစ်ခုခုကို Treatment မရနိုင်စေတဲ့ Categorization များအကြောင်း အနည်းငယ်တွေးကြည့်ပါ။

အခု X နှင့် y Data ကို Load လုပ်ပါ။

> 🎓 ဒီဟာ supervised learning ဖြစ်ပြီး၊ Named 'y' Target လိုအပ်ပါတယ်။

Code Cell အသစ်တစ်ခုမှာ `load_diabetes()` ကို ခေါ်ပြီး Diabetes Dataset ကို Load လုပ်ပါ။ Input `return_X_y=True` သည် `X` ကို Data Matrix အဖြစ်၊ `y` ကို Regression Target အဖြစ် Return ပြန်ပေးပါမယ်။

1. Data Matrix ရဲ့ Shape နဲ့ ပထမဆုံး Element ကို ပြသဖို့ Print Commands အချို့ကို ထည့်ပါ -

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Response အနေနဲ့ Tuple တစ်ခုကို ပြန်ပေးပါမယ်။ Tuple ရဲ့ ပထမဆုံးနှင့် ဒုတိယတန်ဖိုးကို `X` နှင့် `y` အဖြစ် Assign လုပ်ထားပါတယ်။ [Tuples](https://wikipedia.org/wiki/Tuple) အကြောင်းပိုမိုလေ့လာပါ။

    ဒီ Data မှာ 442 Items ရှိပြီး 10 Elements ပါဝင်တဲ့ Arrays အဖြစ် Shape ရှိပါတယ် -

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Data နဲ့ Regression Target အကြား ဆက်နွယ်မှုကို အနည်းငယ်တွေးကြည့်ပါ။ Linear Regression သည် Feature X နှင့် Target Variable y အကြား ဆက်နွယ်မှုကို ခန့်မှန်းပေးပါတယ်။ Diabetes Dataset ရဲ့ [Target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) ကို Documentation မှာ ရှာဖွေပါ။ ဒီ Dataset က ဘာကို ပြသနေသလဲ?

2. Dataset ရဲ့ တစ်စိတ်တစ်ပိုင်းကို Plot ဖို့ Dataset ရဲ့ 3rd Column ကို ရွေးပါ။ `:` Operator ကို အသုံးပြုပြီး Rows အားလုံးကို ရွေးပြီး၊ Index (2) ကို အသုံးပြုပြီး 3rd Column ကို ရွေးပါ။ Data ကို 2D Array အဖြစ် Reshape လုပ်ဖို့ `reshape(n_rows, n_columns)` ကို အသုံးပြုပါ။ Parameter တစ်ခုမှာ -1 ရှိရင်၊ Dimension ကို အလိုအလျောက်တွက်ချက်ပေးပါမယ်။

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Data ရဲ့ Shape ကို အချိန်မရွေး Print လုပ်ပြီး စစ်ဆေးပါ။

3. Data ကို Plot လုပ်ဖို့ ပြင်ဆင်ပြီးနောက်၊ ဒီ Dataset မှာ Numbers တွေကို Logical Split လုပ်ဖို့ Machine ကို အသုံးပြုနိုင်ပါမယ်။ ဒီအလုပ်ကို လုပ်ဖို့ Data (X) နှင့် Target (y) ကို Test နှင့် Training Sets အဖြစ် ခွဲခြားဖို့ လိုအပ်ပါတယ်။ Scikit-learn မှာ ဒီအလုပ်ကို လုပ်ဖို့ လွယ်ကူတဲ့ နည်းလမ်းရှိပါတယ်၊ Test Data ကို တစ်ခုခု Point မှာ Split လုပ်နိုင်ပါတယ်။

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Model ကို Training လုပ်ဖို့ ပြင်ဆင်ပါ! Linear Regression Model ကို Load လုပ်ပြီး `model.fit()` ကို အသုံးပြုပြီး X နှင့် y Training Sets များကို Training လုပ်ပါ။

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` သည် TensorFlow ကဲ့သို့သော ML Libraries များမှာ တွေ့ရနိုင်တဲ့ Function ဖြစ်ပါတယ်။

5. Test Data ကို အသုံးပြုပြီး Prediction တစ်ခုကို ဖန်တီးပါ၊ `predict()` Function ကို အသုံးပြုပါ။ ဒီဟာကို Model ရဲ့ Data Groupings အကြား Line ကို ရေးဆ
✅ ဒီမှာဘာဖြစ်နေတာလဲဆိုတာကိုအနည်းငယ်တွေးကြည့်ပါ။ တစ်ခုတည်းသောတည့်တည့်လိုင်းတစ်ခုဟာ အချက်အလက်အနည်းငယ်များကိုဖြတ်သွားနေပါတယ်၊ ဒါပေမယ့် အတိအကျဘာလုပ်နေတာလဲ? မသိသေးတဲ့ အချက်အလက်တစ်ခုဟာ ဒီလိုင်းနဲ့ပတ်သက်ပြီး y axis ပေါ်မှာဘယ်နေရာမှာရှိမလဲဆိုတာကို သင်ခန့်မှန်းနိုင်ပုံကိုမြင်နိုင်ပါသလား? ဒီမော်ဒယ်ရဲ့ အကျိုးကျေးဇူးကို လက်တွေ့အသုံးချပုံအနေနဲ့ စကားလုံးတွေနဲ့ဖော်ပြကြည့်ပါ။

အောင်မြင်ပါတယ်၊ သင့်ရဲ့ ပထမဆုံး linear regression မော်ဒယ်ကို တည်ဆောက်ပြီး၊ အဲဒီနဲ့ ခန့်မှန်းချက်တစ်ခုကိုဖန်တီးပြီး၊ plot ထဲမှာ ပြသနိုင်ခဲ့ပါပြီ!

---
## 🚀စိန်ခေါ်မှု

ဒီ dataset ထဲက အခြား variable တစ်ခုကို plot လုပ်ပါ။ အကြံပြုချက်: ဒီလိုင်းကို ပြင်ဆင်ပါ `X = X[:,2]`။ ဒီ dataset ရဲ့ target ကိုအခြေခံပြီး၊ ဆီးချိုရောဂါရဲ့ တိုးတက်မှုအခြေအနေကို ဘာတွေရှာဖွေနိုင်မလဲ?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

ဒီသင်ခန်းစာမှာ သင် simple linear regression ကို အသုံးပြုခဲ့ပြီး၊ univariate regression သို့မဟုတ် multiple linear regression မဟုတ်ပါဘူး။ ဒီနည်းလမ်းတွေကြားက ကွာခြားချက်တွေကို အနည်းငယ်ဖတ်ရှုပါ၊ ဒါမှမဟုတ် [ဒီဗီဒီယို](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef) ကိုကြည့်ပါ။

Regression ရဲ့ အယူအဆကို ပိုမိုနားလည်ရန် ဖတ်ရှုပါ၊ ဒီနည်းလမ်းနဲ့ ဖြေရှင်းနိုင်တဲ့ မေးခွန်းအမျိုးမျိုးကို တွေးကြည့်ပါ။ သင့်နားလည်မှုကို ပိုမိုတိုးတက်စေရန် [ဒီသင်ခန်းစာ](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) ကို လေ့လာပါ။

## အလုပ်ပေးစာ

[A different dataset](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ အတည်ပြုထားသော ဘာသာပြန်မှုကို အသုံးပြုရန် အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပာယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။