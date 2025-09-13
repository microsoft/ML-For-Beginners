<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-06T06:28:10+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "ne"
}
-->
# ARIMA सँग समय श्रृंखला पूर्वानुमान

अघिल्लो पाठमा, तपाईंले समय श्रृंखला पूर्वानुमानको बारेमा केही सिक्नुभयो र विद्युत लोडको समय अवधिमा हुने उतार-चढ़ाव देखाउने डेटासेट लोड गर्नुभयो।

[![ARIMA को परिचय](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA को परिचय")

> 🎥 माथिको तस्बिरमा क्लिक गर्नुहोस्: ARIMA मोडेलहरूको छोटो परिचय। यो उदाहरण R मा गरिएको छ, तर अवधारणाहरू सबै ठाउँमा लागू हुन्छन्।

## [पाठपूर्व प्रश्नोत्तरी](https://ff-quizzes.netlify.app/en/ml/)

## परिचय

यस पाठमा, तपाईं [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) प्रयोग गरेर मोडेल निर्माण गर्ने एक विशेष तरिका पत्ता लगाउनुहुनेछ। ARIMA मोडेलहरू विशेष गरी [गैर-स्टेशनरी](https://wikipedia.org/wiki/Stationary_process) डाटालाई फिट गर्न उपयुक्त हुन्छन्।

## सामान्य अवधारणाहरू

ARIMA सँग काम गर्न सक्षम हुन, तपाईंले केही अवधारणाहरू बुझ्न आवश्यक छ:

- 🎓 **स्टेशनरिटी**। सांख्यिकीय सन्दर्भमा, स्टेशनरिटी भनेको त्यस्तो डाटा हो जसको वितरण समयसँगै परिवर्तन हुँदैन। गैर-स्टेशनरी डाटाले ट्रेन्डका कारण उतार-चढ़ाव देखाउँछ, जसलाई विश्लेषण गर्न परिवर्तन गर्नुपर्छ। उदाहरणका लागि, मौसमीपनले डाटामा उतार-चढ़ाव ल्याउन सक्छ, जसलाई 'मौसमी-डिफरेन्सिङ' प्रक्रियाबाट हटाउन सकिन्छ।

- 🎓 **[डिफरेन्सिङ](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**। सांख्यिकीय सन्दर्भमा, डिफरेन्सिङ भनेको गैर-स्टेशनरी डाटालाई स्टेशनरी बनाउन ट्रेन्ड हटाउने प्रक्रिया हो। "डिफरेन्सिङले समय श्रृंखलाको स्तरमा हुने परिवर्तनहरू हटाउँछ, ट्रेन्ड र मौसमीपनलाई समाप्त गर्छ र यसरी श्रृंखलाको औसतलाई स्थिर बनाउँछ।" [Shixiong et al को पेपर](https://arxiv.org/abs/1904.07632)

## समय श्रृंखलाको सन्दर्भमा ARIMA

ARIMA का भागहरूलाई विस्तारमा बुझौं ताकि यसले समय श्रृंखला मोडेल गर्न र भविष्यवाणी गर्न कसरी मद्दत गर्छ भन्ने थाहा पाउन सकियोस्।

- **AR - अटोरेग्रेसिभको लागि**। अटोरेग्रेसिभ मोडेलहरूले, नामले नै जनाएझैं, समयको 'पछाडि' हेर्छन् र डाटाका अघिल्ला मानहरू विश्लेषण गरेर तिनका बारेमा अनुमान लगाउँछन्। यी अघिल्ला मानहरूलाई 'लाग्स' भनिन्छ। उदाहरणका लागि, मासिक पेन्सिल बिक्री देखाउने डाटा। प्रत्येक महिनाको बिक्री कुललाई डाटासेटमा 'विकसित हुने भेरिएबल' मानिन्छ। यो मोडेल यसरी निर्माण गरिन्छ कि "चासोको विकसित हुने भेरिएबललाई यसको आफ्नै पछिल्ला (अर्थात्, अघिल्ला) मानहरूमा रिग्रेस गरिन्छ।" [विकिपीडिया](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - इन्टिग्रेटेडको लागि**। समान 'ARMA' मोडेलहरूको विपरीत, ARIMA मा 'I' ले यसको *[इन्टिग्रेटेड](https://wikipedia.org/wiki/Order_of_integration)* पक्षलाई जनाउँछ। गैर-स्टेशनरीपन हटाउन डिफरेन्सिङ चरणहरू लागू गर्दा डाटालाई 'इन्टिग्रेटेड' गरिन्छ।

- **MA - मुभिङ एभरेजको लागि**। यस मोडेलको [मुभिङ-एभरेज](https://wikipedia.org/wiki/Moving-average_model) पक्षले हालका र अघिल्ला लाग्सका मानहरूलाई अवलोकन गरेर निर्धारण गरिने उत्पादन भेरिएबललाई जनाउँछ।

मुख्य कुरा: ARIMA समय श्रृंखला डाटाको विशेष स्वरूपलाई सकेसम्म नजिकबाट फिट गर्न प्रयोग गरिन्छ।

## अभ्यास - ARIMA मोडेल निर्माण गर्नुहोस्

यस पाठको [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) फोल्डर खोल्नुहोस् र [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) फाइल फेला पार्नुहोस्।

1. `statsmodels` Python लाइब्रेरी लोड गर्न नोटबुक चलाउनुहोस्; तपाईंलाई ARIMA मोडेलहरूको लागि यो आवश्यक हुनेछ।

1. आवश्यक लाइब्रेरीहरू लोड गर्नुहोस्।

1. अब, डाटा प्लट गर्न उपयोगी थप लाइब्रेरीहरू लोड गर्नुहोस्:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. `/data/energy.csv` फाइलबाट डाटा Pandas डाटाफ्रेममा लोड गर्नुहोस् र हेर्नुहोस्:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. जनवरी 2012 देखि डिसेम्बर 2014 सम्मको सबै उपलब्ध ऊर्जा डाटा प्लट गर्नुहोस्। कुनै आश्चर्य हुनु हुँदैन किनभने हामीले यो डाटा अघिल्लो पाठमा देखिसकेका छौं:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    अब, मोडेल निर्माण गरौं!

### प्रशिक्षण र परीक्षण डाटासेटहरू सिर्जना गर्नुहोस्

अब तपाईंको डाटा लोड भइसकेको छ, त्यसलाई प्रशिक्षण र परीक्षण सेटहरूमा विभाजन गर्नुहोस्। तपाईं आफ्नो मोडेललाई प्रशिक्षण सेटमा प्रशिक्षण गर्नुहुनेछ। सामान्यतया, मोडेलले प्रशिक्षण पूरा गरेपछि, तपाईं यसको सटीकता परीक्षण सेट प्रयोग गरेर मूल्याङ्कन गर्नुहुनेछ। तपाईंले सुनिश्चित गर्नुपर्छ कि परीक्षण सेटले प्रशिक्षण सेटभन्दा पछिल्लो समय अवधि समेट्छ ताकि मोडेलले भविष्यको समय अवधिबाट जानकारी प्राप्त नगरोस्।

1. सेप्टेम्बर 1 देखि अक्टोबर 31, 2014 सम्मको दुई महिनाको अवधि प्रशिक्षण सेटलाई छुट्याउनुहोस्। परीक्षण सेटमा नोभेम्बर 1 देखि डिसेम्बर 31, 2014 सम्मको दुई महिनाको अवधि समावेश हुनेछ:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    यो डाटाले दैनिक ऊर्जा खपतलाई प्रतिबिम्बित गर्ने भएकाले, त्यहाँ बलियो मौसमी ढाँचा छ, तर खपत हालका दिनहरूको खपतसँग सबैभन्दा मिल्दोजुल्दो छ।

1. भिन्नताहरूलाई दृश्यात्मक बनाउनुहोस्:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![प्रशिक्षण र परीक्षण डाटा](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    त्यसैले, डाटालाई प्रशिक्षण दिनको लागि तुलनात्मक रूपमा सानो समयको झ्याल प्रयोग गर्नु पर्याप्त हुनुपर्छ।

    > नोट: ARIMA मोडेल फिट गर्न प्रयोग गरिने फंक्शनले फिटिङको क्रममा इन-स्याम्पल मान्यकरण प्रयोग गर्ने भएकाले, हामी मान्यकरण डाटालाई छोड्नेछौं।

### प्रशिक्षणको लागि डाटा तयार गर्नुहोस्

अब, तपाईंले आफ्नो डाटालाई फिल्टर र स्केल गरेर प्रशिक्षणको लागि तयार गर्न आवश्यक छ। तपाईंको डाटासेटलाई आवश्यक समय अवधिहरू र स्तम्भहरू मात्र समावेश गर्न फिल्टर गर्नुहोस्, र डाटालाई (0, 1) को अन्तरालमा प्रक्षेपण गर्न स्केल गर्नुहोस्।

1. मूल डाटासेटलाई माथि उल्लिखित समय अवधिहरू प्रति सेट र आवश्यक स्तम्भ 'load' र मिति मात्र समावेश गर्न फिल्टर गर्नुहोस्:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    तपाईं डाटाको आकार देख्न सक्नुहुन्छ:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. डाटालाई (0, 1) को दायरामा स्केल गर्नुहोस्।

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. मूल डाटा र स्केल गरिएको डाटालाई दृश्यात्मक बनाउनुहोस्:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![मूल](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > मूल डाटा

    ![स्केल गरिएको](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > स्केल गरिएको डाटा

1. अब तपाईंले स्केल गरिएको डाटालाई क्यालिब्रेट गर्नुभएको छ, तपाईं परीक्षण डाटालाई स्केल गर्न सक्नुहुन्छ:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA कार्यान्वयन गर्नुहोस्

अब ARIMA कार्यान्वयन गर्ने समय आएको छ! तपाईंले पहिले इन्स्टल गर्नुभएको `statsmodels` लाइब्रेरी प्रयोग गर्नुहोस्।

अब तपाईंले केही चरणहरू पालना गर्नुपर्छ:

   1. `SARIMAX()` कल गरेर र मोडेलका प्यारामिटरहरू: p, d, र q प्यारामिटरहरू, र P, D, र Q प्यारामिटरहरू पास गरेर मोडेल परिभाषित गर्नुहोस्।
   2. `fit()` फंक्शन कल गरेर प्रशिक्षण डाटाको लागि मोडेल तयार गर्नुहोस्।
   3. `forecast()` फंक्शन कल गरेर र पूर्वानुमान गर्न चरणहरूको संख्या (होराइजन) निर्दिष्ट गरेर भविष्यवाणी गर्नुहोस्।

> 🎓 यी सबै प्यारामिटरहरू केका लागि हुन्? ARIMA मोडेलमा 3 वटा प्यारामिटरहरू हुन्छन् जसले समय श्रृंखलाको प्रमुख पक्षहरू मोडेल गर्न मद्दत गर्छ: मौसमीपन, ट्रेन्ड, र आवाज। यी प्यारामिटरहरू हुन्:

`p`: मोडेलको अटो-रेग्रेसिभ पक्षसँग सम्बन्धित प्यारामिटर, जसले *अघिल्ला* मानहरू समावेश गर्दछ।  
`d`: मोडेलको इन्टिग्रेटेड भागसँग सम्बन्धित प्यारामिटर, जसले समय श्रृंखलामा *डिफरेन्सिङ* (🎓 माथि डिफरेन्सिङ सम्झनुहोस् 👆?) को मात्रा प्रभावित गर्छ।  
`q`: मोडेलको मुभिङ-एभरेज भागसँग सम्बन्धित प्यारामिटर।  

> नोट: यदि तपाईंको डाटामा मौसमी पक्ष छ - जुन यसमा छ - , हामी मौसमी ARIMA मोडेल (SARIMA) प्रयोग गर्छौं। यस अवस्थामा तपाईंले अर्को सेटका प्यारामिटरहरू प्रयोग गर्नुपर्छ: `P`, `D`, र `Q` जसले `p`, `d`, र `q` जस्तै सम्बन्धहरू वर्णन गर्छन्, तर मोडेलका मौसमी घटकहरूसँग सम्बन्धित हुन्छन्।

1. आफ्नो मनपर्ने होराइजन मान सेट गरेर सुरु गर्नुहोस्। 3 घण्टा प्रयास गरौं:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMA मोडेलका प्यारामिटरहरूको लागि उत्तम मानहरू चयन गर्नु चुनौतीपूर्ण हुन सक्छ किनभने यो केही हदसम्म व्यक्तिपरक र समय खपत गर्ने हुन्छ। तपाईं [`pyramid` लाइब्रेरी](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) बाट `auto_arima()` फंक्शन प्रयोग गर्न विचार गर्न सक्नुहुन्छ।

1. अहिलेका लागि राम्रो मोडेल फेला पार्न केही म्यानुअल चयनहरू प्रयास गर्नुहोस्।

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    परिणामहरूको तालिका प्रिन्ट हुन्छ।

तपाईंले आफ्नो पहिलो मोडेल निर्माण गर्नुभयो! अब हामी यसलाई मूल्याङ्कन गर्ने तरिका फेला पार्न आवश्यक छ।

### आफ्नो मोडेल मूल्याङ्कन गर्नुहोस्

तपाईंको मोडेल मूल्याङ्कन गर्न, तथाकथित `walk forward` मान्यकरण प्रदर्शन गर्न सक्नुहुन्छ। व्यवहारमा, समय श्रृंखला मोडेलहरू प्रत्येक पटक नयाँ डाटा उपलब्ध हुँदा पुन: प्रशिक्षण गरिन्छ। यसले मोडेललाई प्रत्येक समय चरणमा उत्तम पूर्वानुमान गर्न अनुमति दिन्छ।

यस प्रविधि प्रयोग गरेर समय श्रृंखलाको सुरुमा, प्रशिक्षण डाटासेटमा मोडेल प्रशिक्षण गर्नुहोस्। त्यसपछि अर्को समय चरणमा भविष्यवाणी गर्नुहोस्। भविष्यवाणीलाई ज्ञात मानसँग तुलना गरेर मूल्याङ्कन गरिन्छ। त्यसपछि प्रशिक्षण सेटलाई ज्ञात मान समावेश गर्न विस्तार गरिन्छ र प्रक्रिया दोहोरिन्छ।

> नोट: तपाईंले प्रशिक्षण सेटको झ्याललाई स्थिर राख्नुपर्छ ताकि प्रत्येक पटक तपाईंले प्रशिक्षण सेटमा नयाँ अवलोकन थप्दा, सेटको सुरुबाट अवलोकन हटाउनुहोस्।

यो प्रक्रिया मोडेलले व्यवहारमा कसरी प्रदर्शन गर्नेछ भन्ने थप बलियो अनुमान प्रदान गर्दछ। यद्यपि, यसले यति धेरै मोडेलहरू सिर्जना गर्ने कम्प्युटेसन लागतमा आउँछ। यदि डाटा सानो छ वा मोडेल सरल छ भने यो स्वीकार्य छ, तर स्केलमा समस्या हुन सक्छ।

Walk-forward मान्यकरण समय श्रृंखला मोडेल मूल्याङ्कनको सुनौलो मानक हो र तपाईंका आफ्नै परियोजनाहरूका लागि सिफारिस गरिन्छ।

1. प्रत्येक HORIZON चरणको लागि परीक्षण डाटा पोइन्ट सिर्जना गर्नुहोस्।

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |  
    | ---------- | -------- | ---- | ------ | ------ |  
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |  
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |  
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |  
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |  
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |  

    डाटालाई यसको होराइजन पोइन्ट अनुसार तेर्सो रूपमा सिफ्ट गरिएको छ।

1. परीक्षण डाटामा स्लाइडिङ विन्डो दृष्टिकोण प्रयोग गरेर भविष्यवाणीहरू गर्नुहोस्:

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

    तपाईं प्रशिक्षण भइरहेको हेर्न सक्नुहुन्छ:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. भविष्यवाणीलाई वास्तविक लोडसँग तुलना गर्नुहोस्:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    आउटपुट  
    |     |            | timestamp | h   | prediction | actual   |  
    | --- | ---------- | --------- | --- | ---------- | -------- |  
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |  
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |  
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |  
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |  
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |  

    घण्टाको डाटाको भविष्यवाणीलाई वास्तविक लोडसँग तुलना गर्नुहोस्। यो कत्तिको सटीक छ?

### मोडेलको सटीकता जाँच गर्नुहोस्

तपाईंको मोडेलको सटीकता जाँच गर्न यसको सबै भविष्यवाणीहरूको औसत प्रतिशत त्रुटि (MAPE) परीक्षण गर्नुहोस्।
> **🧮 गणना देखाउनुहोस्**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) भविष्यवाणीको सटीकता देखाउन प्रयोग गरिन्छ, जुन माथिको सूत्रद्वारा परिभाषित अनुपात हो। वास्तविक र भविष्यवाणी गरिएको बीचको भिन्नतालाई वास्तविक मानबाट भाग लगाइन्छ। 
>
> "यस गणनामा लिइएको परिमाणको पूर्ण मान प्रत्येक भविष्यवाणी गरिएको समय बिन्दुका लागि जोडिन्छ र फिट गरिएका बिन्दुहरूको संख्या n बाट भाग लगाइन्छ।" [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. कोडमा समीकरण व्यक्त गर्नुहोस्:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. एक चरणको MAPE गणना गर्नुहोस्:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    एक चरणको पूर्वानुमान MAPE:  0.5570581332313952 %

1. बहु-चरणको पूर्वानुमान MAPE प्रिन्ट गर्नुहोस्:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    राम्रो कम संख्या उत्तम हुन्छ: विचार गर्नुहोस् कि 10 को MAPE भएको पूर्वानुमान 10% ले गलत छ।

1. तर सधैंझैं, यस्तो सटीकता मापनलाई दृश्यात्मक रूपमा हेर्न सजिलो हुन्छ, त्यसैले यसलाई प्लट गरौं:

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

    ![एक समय श्रृंखला मोडेल](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 एकदम राम्रो प्लट, जसले राम्रो सटीकता भएको मोडेल देखाउँछ। राम्रो काम!

---

## 🚀चुनौती

समय श्रृंखला मोडेलको सटीकता परीक्षण गर्ने तरिकाहरूमा गहिराइमा जानुहोस्। यस पाठमा हामी MAPE को बारेमा कुरा गर्छौं, तर के अन्य विधिहरू छन् जुन तपाईं प्रयोग गर्न सक्नुहुन्छ? तिनीहरूको अनुसन्धान गर्नुहोस् र तिनीहरूलाई व्याख्या गर्नुहोस्। [यहाँ](https://otexts.com/fpp2/accuracy.html) एउटा उपयोगी दस्तावेज फेला पार्न सकिन्छ।

## [पाठ-पछिको प्रश्नोत्तरी](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म-अध्ययन

यो पाठले ARIMA सँग समय श्रृंखला पूर्वानुमानको केवल आधारभूत कुराहरू समेट्छ। [यस रिपोजिटरी](https://microsoft.github.io/forecasting/) र यसको विभिन्न मोडेल प्रकारहरूमा गहिराइमा जान समय लिनुहोस् अन्य तरिकाहरू सिक्न समय श्रृंखला मोडेलहरू निर्माण गर्न।

## असाइनमेन्ट

[नयाँ ARIMA मोडेल](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरी अनुवाद गरिएको हो। हामी यथासम्भव सटीकता सुनिश्चित गर्न प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादहरूमा त्रुटि वा अशुद्धता हुन सक्छ। यसको मूल भाषामा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्त्वपूर्ण जानकारीका लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।