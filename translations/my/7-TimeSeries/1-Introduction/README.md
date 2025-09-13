<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T12:01:59+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "my"
}
-->
# အချိန်စီးဆင်းမှုခန့်မှန်းခြေကိုမိတ်ဆက်ခြင်း

![အချိန်စီးဆင်းမှုအကျဉ်းချုပ်ကို Sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

ဒီသင်ခန်းစာနဲ့နောက်ထပ်သင်ခန်းစာမှာ ML သိပ္ပံပညာရှင်တစ်ဦးရဲ့အရေးပါတဲ့ကျွမ်းကျင်မှုတစ်ခုဖြစ်တဲ့ အချိန်စီးဆင်းမှုခန့်မှန်းခြေကိုလေ့လာပါမယ်။ ဒါဟာနည်းနည်းနဲ့နည်းနည်းနောက်ကျကျသိပ်မလူသိများတဲ့အကြောင်းအရာတစ်ခုဖြစ်ပါတယ်။ အချိန်စီးဆင်းမှုခန့်မှန်းခြေဟာ 'ခရစ်စတယ်ဘောလ်' တစ်ခုလိုမျိုးဖြစ်ပြီး၊ ဈေးနှုန်းလိုမျိုးသော variable ရဲ့အတိတ်လုပ်ဆောင်မှုအပေါ်အခြေခံပြီး၊ ၎င်းရဲ့အနာဂတ်တန်ဖိုးကိုခန့်မှန်းနိုင်ပါတယ်။

[![အချိန်စီးဆင်းမှုခန့်မှန်းခြေကိုမိတ်ဆက်ခြင်း](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introduction to time series forecasting")

> 🎥 အပေါ်ကပုံကိုနှိပ်ပြီး အချိန်စီးဆင်းမှုခန့်မှန်းခြေကိုမိတ်ဆက်တဲ့ဗီဒီယိုကိုကြည့်ပါ

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

ဈေးနှုန်းသတ်မှတ်ခြင်း၊ အရောင်းအဝယ်စာရင်းနှင့်ထောက်ပံ့ရေးကိစ္စများကိုဖြေရှင်းရာတွင်တိုက်ရိုက်အသုံးချနိုင်သောကြောင့်၊ အချိန်စီးဆင်းမှုခန့်မှန်းခြေဟာစီးပွားရေးလုပ်ငန်းအတွက်တန်ဖိုးရှိတဲ့နယ်ပယ်တစ်ခုဖြစ်ပါတယ်။ အနာဂတ်လုပ်ဆောင်မှုကိုပိုမိုကောင်းမွန်စွာခန့်မှန်းနိုင်ဖို့ insights ရရှိရန်အတွက် deep learning နည်းလမ်းများကိုအသုံးပြုလာကြပေမယ့်၊ အချိန်စီးဆင်းမှုခန့်မှန်းခြေဟာ classic ML နည်းလမ်းများကနေတစ်ဆင့်အများကြီးသိရှိထားတဲ့နယ်ပယ်တစ်ခုဖြစ်နေဆဲပါ။

> Penn State ရဲ့အသုံးဝင်တဲ့အချိန်စီးဆင်းမှုသင်ခန်းစာကို [ဒီမှာ](https://online.stat.psu.edu/stat510/lesson/1) ရှာဖွေပါ

## မိတ်ဆက်

သင်ဟာ smart parking meters တစ်ခုစီကိုထိန်းသိမ်းပြီး၊ ၎င်းတို့ကိုဘယ်အချိန်မှာအသုံးပြုပြီး၊ ဘယ်လောက်ကြာကြာအသုံးပြုခဲ့တယ်ဆိုတာကိုအချိန်အလိုက် data ပေးတဲ့ array တစ်ခုရှိတယ်လို့ဆိုပါစို့။

> အတိတ်လုပ်ဆောင်မှုအပေါ်အခြေခံပြီး၊ supply နဲ့ demand ရဲ့ဥပဒေများအရ meter ရဲ့အနာဂတ်တန်ဖိုးကိုခန့်မှန်းနိုင်မယ်ဆိုရင်ရောဘယ်လိုဖြစ်မလဲ?

သင့်ရည်မှန်းချက်ကိုရရှိဖို့အတွက်ဘယ်အချိန်မှာလုပ်ဆောင်ရမလဲဆိုတာကိုတိကျစွာခန့်မှန်းခြေခြင်းဟာ အချိန်စီးဆင်းမှုခန့်မှန်းခြေကဖြေရှင်းနိုင်တဲ့အခက်အခဲတစ်ခုဖြစ်ပါတယ်။ လူတွေက parking spot ရှာနေရင်းအလုပ်များတဲ့အချိန်မှာပိုကြေးပေးရတာကိုမကြိုက်ကြပေမယ့်၊ လမ်းတွေသန့်စင်ဖို့အတွက်ဝင်ငွေထုတ်ယူဖို့အကောင်းဆုံးနည်းလမ်းတစ်ခုဖြစ်နိုင်ပါတယ်!

အချိန်စီးဆင်းမှု algorithm အမျိုးအစားတစ်ချို့ကိုလေ့လာပြီး၊ data ကိုသန့်စင်ပြီးပြင်ဆင်ဖို့ notebook တစ်ခုစတင်ကြပါစို့။ သင်ခန့်မှန်းမယ့် data ဟာ GEFCom2014 forecasting competition ကနေယူထားတာဖြစ်ပြီး၊ 2012 မှ 2014 အထိ 3 နှစ်တာအတွင်း hourly electricity load နဲ့ temperature values ပါဝင်ပါတယ်။ electricity load နဲ့ temperature ရဲ့အတိတ် pattern တွေကိုအခြေခံပြီး၊ electricity load ရဲ့အနာဂတ်တန်ဖိုးကိုခန့်မှန်းနိုင်ပါတယ်။

ဒီဥပမာမှာ၊ historical load data ကိုသာအသုံးပြုပြီး၊ တစ်ခုတည်းသောအချိန်အဆင့်ကိုခန့်မှန်းဖို့လေ့လာပါမယ်။ သို့သော်စတင်မလုပ်ခင်မှာ၊ နောက်ကွယ်မှာဖြစ်နေတဲ့အရာတွေကိုနားလည်ထားရင်အသုံးဝင်ပါတယ်။

## အဓိပ္ပါယ်အချို့

'အချိန်စီးဆင်းမှု' ဆိုတဲ့စကားလုံးကိုတွေ့ရင်၊ ၎င်းကိုအခြား context အမျိုးမျိုးမှာအသုံးပြုပုံကိုနားလည်ဖို့လိုပါတယ်။

🎓 **အချိန်စီးဆင်းမှု**

ဂဏန်းသိပ္ပံမှာ "အချိန်စီးဆင်းမှုဆိုတာ အချိန်အလိုက်စဉ်ဆက်မပြတ် data points တွေကို index လုပ်ထားတာ (သို့မဟုတ် စာရင်းပြုလုပ်ထားတာ သို့မဟုတ် graph ပုံဖော်ထားတာ) ဖြစ်ပါတယ်။ အများဆုံးအချိန်စီးဆင်းမှုဟာ အချိန်အလိုက်တန်းစီထားတဲ့အဆက်မပြတ် data points တွေဖြစ်ပါတယ်။" အချိန်စီးဆင်းမှုရဲ့ဥပမာတစ်ခုက [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series) ရဲ့နေ့စဉ်ပိတ်ချိန်တန်ဖိုးဖြစ်ပါတယ်။ အချိန်စီးဆင်းမှု plot တွေကိုအသုံးပြုခြင်းနဲ့ စာရင်းဇယားပုံစံကို signal processing, မိုးလေဝသခန့်မှန်းခြေ, ငလျင်ခန့်မှန်းခြေ, နဲ့အခြားအခွင့်အရေးတွေမှာတွေ့ရပါတယ်။

🎓 **အချိန်စီးဆင်းမှုခန့်မှန်းခြေ**

အချိန်စီးဆင်းမှုခန့်မှန်းခြေဟာ အတိတ် data ရဲ့ pattern တွေကိုအခြေခံပြီး model တစ်ခုကိုအသုံးပြုပြီးအနာဂတ်တန်ဖိုးကိုခန့်မှန်းခြင်းဖြစ်ပါတယ်။ Regression models တွေကိုအသုံးပြုပြီးအချိန်စီးဆင်းမှု data ကိုလေ့လာနိုင်ပေမယ့်၊ time indices တွေကို x variables အနေနဲ့ plot ပေါ်မှာထားပြီး၊ ဒီ data ကိုအထူး model အမျိုးအစားတွေကိုအသုံးပြုပြီးလေ့လာရတာပိုကောင်းပါတယ်။

အချိန်စီးဆင်းမှု data ဟာတန်းစီထားတဲ့ observation တွေဖြစ်ပြီး၊ linear regression နဲ့လေ့လာနိုင်တဲ့ data မဟုတ်ပါဘူး။ အများဆုံးအသုံးပြုတဲ့ model တစ်ခုက ARIMA ဖြစ်ပြီး၊ "Autoregressive Integrated Moving Average" ဆိုတဲ့အတိုကောက်ဖြစ်ပါတယ်။

[ARIMA models](https://online.stat.psu.edu/stat510/lesson/1/1.1) "series ရဲ့လက်ရှိတန်ဖိုးကို အတိတ်တန်ဖိုးတွေနဲ့ အတိတ်ခန့်မှန်းမှု error တွေကိုဆက်စပ်ထားပါတယ်။" ARIMA models တွေဟာ time-domain data ကိုလေ့လာဖို့အကောင်းဆုံးဖြစ်ပြီး၊ data ဟာအချိန်အလိုက်တန်းစီထားပါတယ်။

> ARIMA models အမျိုးအစားအများကြီးရှိပြီး၊ [ဒီမှာ](https://people.duke.edu/~rnau/411arim.htm) လေ့လာနိုင်ပါတယ်။ နောက်ထပ်သင်ခန်းစာမှာ ARIMA model တစ်ခုကိုတည်ဆောက်မယ်။

နောက်ထပ်သင်ခန်းစာမှာ [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) ကိုအသုံးပြုပြီး ARIMA model တစ်ခုကိုတည်ဆောက်ပါမယ်။ Univariate Time Series ဟာတစ်ခုတည်းသော variable ကိုအခြေခံပြီး၊ အချိန်အလိုက်တန်ဖိုးပြောင်းလဲမှုကိုလေ့လာပါတယ်။ ဒီ data ရဲ့ဥပမာတစ်ခုက Mauna Loa Observatory မှာ monthly CO2 concentration ကိုမှတ်တမ်းတင်ထားတဲ့ [ဒီ dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) ဖြစ်ပါတယ်။

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ ဒီ dataset မှာအချိန်အလိုက်ပြောင်းလဲနေတဲ့ variable ကိုဖော်ထုတ်ပါ

## အချိန်စီးဆင်းမှု data ရဲ့ဂုဏ်သတ္တိများကိုစဉ်းစားရန်

အချိန်စီးဆင်းမှု data ကိုကြည့်မယ်ဆိုရင်၊ [အချို့သောဂုဏ်သတ္တိများ](https://online.stat.psu.edu/stat510/lesson/1/1.1) ရှိနေနိုင်ပြီး၊ pattern တွေကိုပိုမိုနားလည်နိုင်ဖို့အတွက် offset လုပ်ဖို့ statistical techniques တစ်ချို့ကိုအသုံးပြုရပါမယ်။ 

ဒီအချက်အလက်တွေကိုနားလည်ဖို့လိုတဲ့အဓိက concept တွေက:

🎓 **Trends**

Trend တွေဟာအချိန်အလိုက်တိုးတက်မှုနဲ့ကျဆင်းမှုတွေကိုတိုင်းတာနိုင်ပါတယ်။ [ပိုမိုလေ့လာရန်](https://machinelearningmastery.com/time-series-trends-in-python)

🎓 **[Seasonality](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Seasonality ဆိုတာပုံမှန်အချိန်ကာလအတွင်းဖြစ်ပေါ်တဲ့အတက်အကျတွေကိုဆိုလိုပါတယ်။ ဥပမာအားဖြင့် အရောင်းအဝယ်ပုံစံတွေကိုသက်ရောက်စေတဲ့ပွဲတော်ကာလတွေ။

🎓 **Outliers**

Outliers ဆိုတာ data ရဲ့ပုံမှန် variance ကနေဝေးနေတဲ့ data point တွေဖြစ်ပါတယ်။

🎓 **Long-run cycle**

Seasonality ကိုမထည့်သွင်းပဲ၊ data ဟာအချိန်ကြာမြင့်တဲ့ cycle ကိုပြသနိုင်ပါတယ်။

🎓 **Constant variance**

အချိန်အလိုက် data တစ်ချို့ဟာ constant fluctuations ကိုပြသနိုင်ပါတယ်။

🎓 **Abrupt changes**

Data ဟာရုတ်တရက်ပြောင်းလဲမှုကိုပြသနိုင်ပြီး၊ နောက်ထပ် analysis လိုအပ်နိုင်ပါတယ်။

✅ [ဒီဥပမာ plot](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) ကိုကြည့်ပြီး၊ အထက်မှာဖော်ပြထားတဲ့ဂုဏ်သတ္တိတွေကိုရှာဖွေပါ။

![In-game currency spend](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## လေ့ကျင့်ခန်း - power usage data ကိုစတင်ခြင်း

အတိတ် usage ကိုအခြေခံပြီး၊ အနာဂတ် power usage ကိုခန့်မှန်းမယ့်အချိန်စီးဆင်းမှု model တစ်ခုကိုစတင်ဖန်တီးပါ။

> ဒီဥပမာရဲ့ data ဟာ GEFCom2014 forecasting competition ကနေယူထားတာဖြစ်ပြီး၊ 2012 မှ 2014 အထိ 3 နှစ်တာအတွင်း hourly electricity load နဲ့ temperature values ပါဝင်ပါတယ်။
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. ဒီသင်ခန်းစာရဲ့ `working` folder မှာ _notebook.ipynb_ ဖိုင်ကိုဖွင့်ပါ။ Data ကို load နဲ့ visualize လုပ်ဖို့လိုအပ်တဲ့ libraries တွေကိုထည့်ပါ။

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    သင့် environment ကို set up လုပ်ပြီး၊ data ကို download လုပ်ဖို့ `common` folder ထဲက files တွေကိုအသုံးပြုနေပါတယ်။

2. နောက်တစ်ဆင့်မှာ၊ `load_data()` နဲ့ `head()` ကိုခေါ်ပြီး data ကို dataframe အနေနဲ့ကြည့်ပါ။

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    သင့်မှာ date နဲ့ load ကိုကိုယ်စားပြုတဲ့ column နှစ်ခုရှိပါတယ်။

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. အခုတော့ data ကို `plot()` ကိုခေါ်ပြီး plot လုပ်ပါ။

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![energy plot](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. 2014 ခုနှစ် ဇူလိုင်လပထမအပတ်ကို plot လုပ်ပါ။ `energy` ကို `[from date]: [to date]` ပုံစံအနေနဲ့ input ပေးပါ။

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![july](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    အလှပဆုံး plot တစ်ခု! ဒီ plot တွေကိုကြည့်ပြီး၊ အထက်မှာဖော်ပြထားတဲ့ဂုဏ်သတ္တိတွေကိုရှာဖွေပါ။ Data ကို visualize လုပ်ခြင်းကနေဘာတွေကိုသုံးသပ်နိုင်မလဲ?

နောက်ထပ်သင်ခန်းစာမှာ ARIMA model တစ်ခုကိုဖန်တီးပြီး forecast လုပ်ပါမယ်။

---

## 🚀Challenge

အချိန်စီးဆင်းမှုခန့်မှန်းခြေကအကျိုးရှိမယ့်စက်မှုလုပ်ငန်းနဲ့နယ်ပယ်တွေကိုစာရင်းပြုစုပါ။ အနုပညာ၊ စီးပွားရေးသိပ္ပံ (Econometrics)၊ သဘာဝပတ်ဝန်းကျင် (Ecology)၊ လက်လီရောင်းဝယ်ရေး၊ စက်မှုလုပ်ငန်း၊ ငွေကြေးနယ်ပယ်တွေမှာဒီနည်းလမ်းတွေကိုအသုံးချနိုင်မယ့် application တွေကိုစဉ်းစားနိုင်ပါသလား?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

ဒီမှာမဖော်ပြထားပေမယ့်၊ neural networks တွေကိုအချိန်စီးဆင်းမှုခန့်မှန်းခြေ classic နည်းလမ်းတွေကိုတိုးတက်စေဖို့ sometimes အသုံးပြုပါတယ်။ [ဒီဆောင်းပါး](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412) မှာပိုမိုလေ့လာပါ။

## Assignment

[အချိန်စီးဆင်းမှု data တွေကိုပိုမို visualize လုပ်ပါ](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို ကျေးဇူးပြု၍ သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်ရန် လိုအပ်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူက ဘာသာပြန်ဆောင်ရွက်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပါယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။