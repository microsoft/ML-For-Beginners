<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T11:56:13+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "my"
}
-->
# ARIMA ဖြင့် အချိန်စီးရီးခန့်မှန်းခြေ

ယခင်သင်ခန်းစာတွင်၊ အချိန်စီးရီးခန့်မှန်းခြေကို အနည်းငယ်လေ့လာပြီး အချိန်ကာလအတွင်း လျှပ်စစ်သုံးစွဲမှုအတက်အကျကို ဖော်ပြသည့် ဒေတာစဉ်ကို တင်သွင်းခဲ့ပါသည်။

[![ARIMA အကြောင်းမိတ်ဆက်](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA အကြောင်းမိတ်ဆက်")

> 🎥 အထက်ပါပုံကို နှိပ်ပြီး ဗီဒီယိုကြည့်ပါ: ARIMA မော်ဒယ်များအကြောင်း အကျဉ်းချုပ်။ ဤနမူနာကို R တွင် ပြုလုပ်ထားသော်လည်း၊ အယူအဆများသည် အထွေထွေသဘောတူပါသည်။

## [သင်ခန်းစာမတိုင်မီ မေးခွန်း](https://ff-quizzes.netlify.app/en/ml/)

## မိတ်ဆက်

ဤသင်ခန်းစာတွင်၊ [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) ဖြင့် မော်ဒယ်များတည်ဆောက်ရန် နည်းလမ်းတစ်ခုကို ရှာဖွေပါမည်။ ARIMA မော်ဒယ်များသည် [non-stationarity](https://wikipedia.org/wiki/Stationary_process) ပြသသည့် ဒေတာများကို အထူးသင့်လျော်သည်။

## အထွေထွေအယူအဆများ

ARIMA ကို အသုံးပြုနိုင်ရန် အောက်ပါအယူအဆများကို သိရှိထားရန် လိုအပ်ပါသည်-

- 🎓 **Stationarity**: စာရင်းဇယားအရ၊ stationarity သည် အချိန်အလိုက် ရွှေ့လျားသည့်အခါ မပြောင်းလဲသော ဒေတာဖြန့်ဖြူးမှုကို ဆိုလိုသည်။ Non-stationary ဒေတာသည် အဆင့်အတက်အကျကြောင့် အတက်အကျပြသပြီး၊ ခန့်မှန်းရန်အတွက် ပြောင်းလဲမှုများ လိုအပ်သည်။ ဥပမာအားဖြင့် ရာသီဥတုအခြေအနေသည် ဒေတာတွင် အတက်အကျများကို ဖြစ်ပေါ်စေပြီး 'seasonal-differencing' လုပ်ငန်းစဉ်ဖြင့် ဖယ်ရှားနိုင်သည်။

- 🎓 **[Differencing](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**: Differencing သည် non-stationary ဒေတာကို stationarity ဖြစ်စေရန် ပြောင်းလဲခြင်းဖြစ်ပြီး၊ non-constant trend ကို ဖယ်ရှားသည်။ "Differencing သည် အချိန်စီးရီး၏ အဆင့်အတက်အကျများကို ဖယ်ရှားပြီး၊ trend နှင့် seasonality ကို ဖယ်ရှားကာ အချိန်စီးရီး၏ mean ကို တည်ငြိမ်စေသည်။" [Shixiong et al မှ စာတမ်း](https://arxiv.org/abs/1904.07632)

## အချိန်စီးရီးတွင် ARIMA ၏ အရေးပါမှု

ARIMA ၏ အစိတ်အပိုင်းများကို ဖွင့်ရှင်းပြီး၊ အချိန်စီးရီးကို မော်ဒယ်တည်ဆောက်ရန်နှင့် ခန့်မှန်းရန် အကူအညီပေးပုံကို နားလည်ပါမည်။

- **AR - AutoRegressive**: Autoregressive မော်ဒယ်များသည် အတိတ်ဒေတာများကို 'နောက်' ပြန်ကြည့်ကာ ခန့်မှန်းချက်များ ပြုလုပ်သည်။ ဥပမာအားဖြင့် ခဲတံရောင်းအား၏ လစဉ်ဒေတာသည် 'evolving variable' အဖြစ် သတ်မှတ်နိုင်သည်။ [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrated**: ARIMA ၏ 'I' သည် [integrated](https://wikipedia.org/wiki/Order_of_integration) aspect ကို ဆိုလိုသည်။ Differencing လုပ်ငန်းစဉ်များကို အသုံးပြုကာ non-stationarity ကို ဖယ်ရှားသည်။

- **MA - Moving Average**: [Moving-average](https://wikipedia.org/wiki/Moving-average_model) aspect သည် lag များ၏ လက်ရှိနှင့် အတိတ်တန်ဖိုးများကို ကြည့်ကာ output variable ကို သတ်မှတ်သည်။

အကျဉ်းချုပ်: ARIMA သည် အချိန်စီးရီးဒေတာ၏ အထူးပုံစံနှင့် အနီးစပ်ဆုံးဖြစ်စေရန် မော်ဒယ်တည်ဆောက်ရန် အသုံးပြုသည်။

## လေ့ကျင့်ခန်း - ARIMA မော်ဒယ်တည်ဆောက်ခြင်း

ဤသင်ခန်းစာ၏ [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) ဖိုလ်ဒါကို ဖွင့်ပြီး [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) ဖိုင်ကို ရှာပါ။

1. `statsmodels` Python library ကို notebook တွင် run လုပ်ပါ။ ARIMA မော်ဒယ်များအတွက် လိုအပ်ပါသည်။

1. လိုအပ်သော libraries များကို load လုပ်ပါ။

1. ဒေတာကို `/data/energy.csv` ဖိုင်မှ Pandas dataframe သို့ load လုပ်ပြီး ကြည့်ပါ:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. 2012 ခုနှစ် ဇန်နဝါရီမှ 2014 ခုနှစ် ဒီဇင်ဘာအထိရှိသော လျှပ်စစ်သုံးစွဲမှုဒေတာအား plot လုပ်ပါ:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ယခု မော်ဒယ်တည်ဆောက်ရန် စတင်ပါ။

### Training နှင့် Testing ဒေတာစဉ်များ ဖန်တီးခြင်း

ဒေတာကို load လုပ်ပြီးပါက၊ train နှင့် test sets သို့ ခွဲခြားပါ။ Train set တွင် မော်ဒယ်ကို လေ့ကျင့်ပြီး၊ test set တွင် accuracy ကို စစ်ဆေးပါမည်။

1. 2014 ခုနှစ် စက်တင်ဘာ 1 မှ အောက်တိုဘာ 31 အထိ train set အဖြစ် သတ်မှတ်ပါ။ Test set သည် 2014 ခုနှစ် နိုဝင်ဘာ 1 မှ ဒီဇင်ဘာ 31 အထိ ဖြစ်ပါမည်:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    ဒေတာသည် ရာသီဥတုအတက်အကျများကို ပြသသော်လည်း၊ နောက်ဆုံးရက်များနှင့် ဆင်တူသည်။

1. ခွဲခြားထားသော train နှင့် test sets ကို visualization ပြုလုပ်ပါ:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![training and testing data](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    > မှတ်ချက်: ARIMA မော်ဒယ် fitting လုပ်စဉ်တွင် in-sample validation ကို အသုံးပြုသဖြင့် validation data ကို မထည့်ပါ။

### Training အတွက် ဒေတာကို ပြင်ဆင်ခြင်း

Training အတွက် ဒေတာကို filter နှင့် scale ပြုလုပ်ပါ။ 

1. Train နှင့် test sets အတွက် လိုအပ်သော column များကို filter လုပ်ပါ:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

1. ဒေတာကို (0, 1) interval အတွင်း scale ပြုလုပ်ပါ:

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Original ဒေတာနှင့် scaled ဒေတာကို ကြည့်ပါ:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Original ဒေတာ

    ![scaled](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Scaled ဒေတာ

### ARIMA ကို အကောင်အထည်ဖော်ခြင်း

`statsmodels` library ကို အသုံးပြုကာ ARIMA ကို အကောင်အထည်ဖော်ပါ။

1. Horizon value ကို သတ်မှတ်ပါ။ ဥပမာအားဖြင့် 3 နာရီ:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

1. Parameters များကို manual ရွေးချယ်ပါ:

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    > 🎓 Parameters များသည် seasonality, trend, နှင့် noise ကို မော်ဒယ်တည်ဆောက်ရန် အရေးပါသည်။

### မော်ဒယ်ကို အကဲဖြတ်ခြင်း

Walk-forward validation ကို အသုံးပြုကာ မော်ဒယ်ကို အကဲဖြတ်ပါ။

1. Test data point များကို horizon step အလိုက် ဖန်တီးပါ:

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

1. Sliding window approach ဖြင့် prediction ပြုလုပ်ပါ:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

1. Prediction နှင့် actual load ကို နှိုင်းယှဉ်ပါ:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    > Output: Prediction နှင့် actual load တို့၏ တိကျမှုကို စစ်ဆေးပါ။

### မော်ဒယ်၏ တိကျမှုကို စစ်ဆေးခြင်း

Mean absolute percentage error (MAPE) ကို အသုံးပြုကာ မော်ဒယ်၏ တိကျမှုကို စစ်ဆေးပါ။
> **🧮 သင်္ချာကို ပြပါ**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) ကို အထက်ပါဖော်ပြထားသော ဖော်မြူလာအတိုင်း အချိုးအစားအဖြစ် သတ်မှတ်ပြီး ခန့်မှန်းချက်တိကျမှုကို ပြရန် အသုံးပြုသည်။ အမှန်တကယ်ဖြစ်ရပ်နှင့် ခန့်မှန်းချက်အကြားကွာဟချက်ကို အမှန်တကယ်ဖြစ်ရပ်ဖြင့် ခွဲခြားသည်။
>
> "ဤတွက်ချက်မှုတွင် အပြည့်အဝတန်ဖိုးကို အချိန်အပိုင်းအခြားတိုင်းတွင် ခန့်မှန်းထားသော အချက်အလက်များအတွက် စုစုပေါင်းတွက်ချက်ပြီး ခန့်မှန်းထားသော အချက်အလက်အရေအတွက် n ဖြင့် ခွဲခြားသည်။" [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. ကုဒ်ဖြင့် ဆွဲချက်ကို ဖော်ပြပါ။

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. တစ်ဆင့် MAPE ကိုတွက်ချက်ပါ။

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    တစ်ဆင့်ခန့်မှန်းချက် MAPE:  0.5570581332313952 %

1. အဆင့်များစွာခန့်မှန်းချက် MAPE ကို ပုံနှိပ်ပါ။

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    နိမ့်သောနံပါတ်က အကောင်းဆုံးဖြစ်သည်။ MAPE 10 ရှိသောခန့်မှန်းချက်တစ်ခုသည် 10% မှားနေသည်ဟု ဆိုလိုသည်။

1. သို့သော် အမြဲတမ်းလိုလို၊ ဒီလိုတိကျမှုတိုင်းတာမှုကို မြင်ရလွယ်အောင် ပုံဆွဲကြည့်ရင် ပိုမိုကောင်းမွန်သည်။

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![အချိန်စီးရီးမော်ဒယ်တစ်ခု](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 တိကျမှုကောင်းမွန်သော မော်ဒယ်ကို ဖော်ပြထားသည့် အလွန်ကောင်းမွန်သော ပုံတစ်ပုံ။ အလုပ်ကောင်းပါတယ်!

---

## 🚀စိန်ခေါ်မှု

အချိန်စီးရီးမော်ဒယ်တစ်ခု၏ တိကျမှုကို စမ်းသပ်နိုင်သော နည်းလမ်းများကို လေ့လာပါ။ ဒီသင်ခန်းစာတွင် MAPE ကိုသာ ထိတွေ့ခဲ့ကြပေမယ့် သင်အသုံးပြုနိုင်သည့် အခြားနည်းလမ်းများ ရှိပါသလား။ ၎င်းတို့ကို သုတေသနပြုပြီး မှတ်ချက်ထည့်ပါ။ အသုံးဝင်သော စာရွက်စာတမ်းကို [ဒီမှာ](https://otexts.com/fpp2/accuracy.html) တွေ့နိုင်ပါသည်။

## [သင်ခန်းစာပြီးနောက် မေးခွန်း](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

ဒီသင်ခန်းစာသည် ARIMA ဖြင့် အချိန်စီးရီးခန့်မှန်းခြင်း၏ အခြေခံအချက်များကိုသာ ထိတွေ့သည်။ [ဒီ repository](https://microsoft.github.io/forecasting/) နှင့် ၎င်း၏ မော်ဒယ်အမျိုးအစားများကို လေ့လာခြင်းဖြင့် အချိန်စီးရီးမော်ဒယ်များ တည်ဆောက်ရန် အခြားနည်းလမ်းများကို သင်ယူရန် အချိန်ယူပါ။

## လုပ်ငန်းတာဝန်

[ARIMA မော်ဒယ်အသစ်](assignment.md)

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတရ အရင်းအမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပရော်ဖက်ရှင်နယ် ဘာသာပြန်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွဲအချော်များ သို့မဟုတ် အနားလွဲမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။