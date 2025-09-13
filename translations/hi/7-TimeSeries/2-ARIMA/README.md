<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T10:17:13+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "hi"
}
-->
# ARIMA के साथ समय श्रृंखला पूर्वानुमान

पिछले पाठ में, आपने समय श्रृंखला पूर्वानुमान के बारे में थोड़ा सीखा और एक डेटा सेट लोड किया जो एक समय अवधि के दौरान विद्युत भार में उतार-चढ़ाव दिखाता है।

[![ARIMA का परिचय](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA का परिचय")

> 🎥 ऊपर दी गई छवि पर क्लिक करें: ARIMA मॉडल का संक्षिप्त परिचय। उदाहरण R में किया गया है, लेकिन अवधारणाएं सार्वभौमिक हैं।

## [प्री-लेक्चर क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## परिचय

इस पाठ में, आप [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) के साथ मॉडल बनाने का एक विशिष्ट तरीका जानेंगे। ARIMA मॉडल विशेष रूप से [गैर-स्थिरता](https://wikipedia.org/wiki/Stationary_process) दिखाने वाले डेटा को फिट करने के लिए उपयुक्त हैं।

## सामान्य अवधारणाएं

ARIMA के साथ काम करने के लिए, आपको कुछ अवधारणाओं के बारे में जानना होगा:

- 🎓 **स्थिरता**। सांख्यिकीय संदर्भ में, स्थिरता उस डेटा को संदर्भित करती है जिसका वितरण समय में स्थानांतरित होने पर नहीं बदलता। गैर-स्थिर डेटा, फिर, रुझानों के कारण उतार-चढ़ाव दिखाता है जिसे विश्लेषण करने के लिए बदलना आवश्यक है। उदाहरण के लिए, मौसमीता डेटा में उतार-चढ़ाव ला सकती है और इसे 'मौसमी-डिफरेंसिंग' की प्रक्रिया द्वारा समाप्त किया जा सकता है।

- 🎓 **[डिफरेंसिंग](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**। सांख्यिकीय संदर्भ में, डिफरेंसिंग डेटा को स्थिर बनाने के लिए इसे बदलने की प्रक्रिया को संदर्भित करता है, जिससे इसके गैर-स्थिर रुझान को हटाया जाता है। "डिफरेंसिंग समय श्रृंखला के स्तर में बदलाव को हटाता है, रुझान और मौसमीता को समाप्त करता है और परिणामस्वरूप समय श्रृंखला के औसत को स्थिर करता है।" [Shixiong et al द्वारा पेपर](https://arxiv.org/abs/1904.07632)

## समय श्रृंखला के संदर्भ में ARIMA

ARIMA के भागों को बेहतर ढंग से समझने के लिए आइए इसे विस्तार से देखें कि यह हमें समय श्रृंखला को मॉडल करने और इसके खिलाफ पूर्वानुमान बनाने में कैसे मदद करता है।

- **AR - ऑटोरेग्रेसिव के लिए**। जैसा कि नाम से पता चलता है, ऑटोरेग्रेसिव मॉडल समय में 'पीछे' देखते हैं ताकि आपके डेटा में पिछले मानों का विश्लेषण किया जा सके और उनके बारे में धारणाएं बनाई जा सकें। इन पिछले मानों को 'लैग्स' कहा जाता है। एक उदाहरण होगा डेटा जो पेंसिल की मासिक बिक्री दिखाता है। प्रत्येक महीने की बिक्री कुल को डेटा सेट में एक 'विकसित चर' माना जाएगा। यह मॉडल इस प्रकार बनाया गया है कि "रुचि का विकसित चर अपने स्वयं के लैग्ड (यानी, पिछले) मानों पर पुन: स्थापित होता है।" [विकिपीडिया](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - इंटीग्रेटेड के लिए**। समान 'ARMA' मॉडल के विपरीत, ARIMA में 'I' इसके *[इंटीग्रेटेड](https://wikipedia.org/wiki/Order_of_integration)* पहलू को संदर्भित करता है। गैर-स्थिरता को समाप्त करने के लिए डिफरेंसिंग चरण लागू होने पर डेटा 'इंटीग्रेटेड' होता है।

- **MA - मूविंग एवरेज के लिए**। इस मॉडल का [मूविंग-एवरेज](https://wikipedia.org/wiki/Moving-average_model) पहलू आउटपुट वेरिएबल को संदर्भित करता है जो लैग्स के वर्तमान और पिछले मानों को देखकर निर्धारित किया जाता है।

सारांश: ARIMA का उपयोग समय श्रृंखला डेटा के विशेष रूप को यथासंभव करीब से फिट करने के लिए किया जाता है।

## अभ्यास - ARIMA मॉडल बनाएं

इस पाठ में [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) फ़ोल्डर खोलें और [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) फ़ाइल खोजें।

1. नोटबुक चलाएं ताकि `statsmodels` Python लाइब्रेरी लोड हो सके; आपको ARIMA मॉडल के लिए इसकी आवश्यकता होगी।

1. आवश्यक लाइब्रेरी लोड करें।

1. अब, डेटा को प्लॉट करने के लिए उपयोगी कई और लाइब्रेरी लोड करें:

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

1. `/data/energy.csv` फ़ाइल से डेटा को एक Pandas डेटा फ्रेम में लोड करें और इसे देखें:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. जनवरी 2012 से दिसंबर 2014 तक उपलब्ध ऊर्जा डेटा को प्लॉट करें। इसमें कोई आश्चर्य नहीं होना चाहिए क्योंकि हमने यह डेटा पिछले पाठ में देखा था:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    अब, आइए एक मॉडल बनाएं!

### प्रशिक्षण और परीक्षण डेटा सेट बनाएं

अब आपका डेटा लोड हो गया है, इसलिए आप इसे ट्रेन और टेस्ट सेट में विभाजित कर सकते हैं। आप अपने मॉडल को ट्रेन सेट पर प्रशिक्षित करेंगे। हमेशा की तरह, मॉडल के प्रशिक्षण समाप्त होने के बाद, आप टेस्ट सेट का उपयोग करके इसकी सटीकता का मूल्यांकन करेंगे। आपको यह सुनिश्चित करना होगा कि टेस्ट सेट ट्रेनिंग सेट की तुलना में समय की बाद की अवधि को कवर करता है ताकि यह सुनिश्चित हो सके कि मॉडल भविष्य की समय अवधि से जानकारी प्राप्त न करे।

1. सितंबर 1 से अक्टूबर 31, 2014 तक की दो महीने की अवधि को ट्रेनिंग सेट में आवंटित करें। टेस्ट सेट में नवंबर 1 से दिसंबर 31, 2014 तक की दो महीने की अवधि शामिल होगी:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    चूंकि यह डेटा ऊर्जा की दैनिक खपत को दर्शाता है, इसमें एक मजबूत मौसमी पैटर्न है, लेकिन खपत हाल के दिनों की खपत के समान है।

1. अंतर को विज़ुअलाइज़ करें:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![ट्रेनिंग और टेस्टिंग डेटा](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    इसलिए, डेटा को प्रशिक्षित करने के लिए अपेक्षाकृत छोटे समय विंडो का उपयोग पर्याप्त होना चाहिए।

    > नोट: चूंकि हम ARIMA मॉडल को फिट करने के लिए उपयोग किए जाने वाले फ़ंक्शन में फिटिंग के दौरान इन-सैंपल मान्यता का उपयोग करते हैं, हम मान्यता डेटा को छोड़ देंगे।

### प्रशिक्षण के लिए डेटा तैयार करें

अब, आपको डेटा को फ़िल्टर और स्केल करके प्रशिक्षण के लिए तैयार करना होगा। अपने डेटा सेट को केवल आवश्यक समय अवधि और कॉलम शामिल करने के लिए फ़िल्टर करें, और डेटा को 0,1 के अंतराल में प्रोजेक्ट करने के लिए स्केल करें।

1. मूल डेटा सेट को केवल उपरोक्त समय अवधि प्रति सेट और केवल आवश्यक कॉलम 'लोड' और तारीख को शामिल करने के लिए फ़िल्टर करें:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    आप डेटा का आकार देख सकते हैं:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. डेटा को (0, 1) की सीमा में स्केल करें।

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. मूल बनाम स्केल किए गए डेटा को विज़ुअलाइज़ करें:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![मूल](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > मूल डेटा

    ![स्केल किया गया](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > स्केल किया गया डेटा

1. अब जब आपने स्केल किए गए डेटा को कैलिब्रेट कर लिया है, तो आप टेस्ट डेटा को स्केल कर सकते हैं:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA लागू करें

अब ARIMA को लागू करने का समय है! आप अब `statsmodels` लाइब्रेरी का उपयोग करेंगे जिसे आपने पहले इंस्टॉल किया था।

अब आपको कई चरणों का पालन करना होगा:

   1. मॉडल को परिभाषित करें `SARIMAX()` को कॉल करके और मॉडल पैरामीटर: p, d, और q पैरामीटर, और P, D, और Q पैरामीटर पास करें।
   2. ट्रेनिंग डेटा के लिए मॉडल तैयार करें `fit()` फ़ंक्शन को कॉल करके।
   3. भविष्यवाणी करें `forecast()` फ़ंक्शन को कॉल करके और पूर्वानुमान के लिए चरणों की संख्या (होराइजन) निर्दिष्ट करें।

> 🎓 ये सभी पैरामीटर किस लिए हैं? ARIMA मॉडल में 3 पैरामीटर होते हैं जो समय श्रृंखला के प्रमुख पहलुओं को मॉडल करने में मदद करते हैं: मौसमीता, रुझान, और शोर। ये पैरामीटर हैं:

`p`: मॉडल के ऑटो-रेग्रेसिव पहलू से जुड़ा पैरामीटर, जो *पिछले* मानों को शामिल करता है।
`d`: मॉडल के इंटीग्रेटेड भाग से जुड़ा पैरामीटर, जो समय श्रृंखला पर लागू *डिफरेंसिंग* (🎓 डिफरेंसिंग याद है 👆?) की मात्रा को प्रभावित करता है।
`q`: मॉडल के मूविंग-एवरेज भाग से जुड़ा पैरामीटर।

> नोट: यदि आपके डेटा में मौसमी पहलू है - जैसा कि इस डेटा में है - तो हम मौसमी ARIMA मॉडल (SARIMA) का उपयोग करते हैं। इस मामले में आपको पैरामीटर का एक और सेट उपयोग करना होगा: `P`, `D`, और `Q` जो `p`, `d`, और `q` के समान संघों का वर्णन करते हैं, लेकिन मॉडल के मौसमी घटकों से संबंधित हैं।

1. अपने पसंदीदा होराइजन मान सेट करें। चलिए 3 घंटे आज़माते हैं:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMA मॉडल के पैरामीटर के लिए सर्वोत्तम मान चुनना चुनौतीपूर्ण हो सकता है क्योंकि यह कुछ हद तक व्यक्तिपरक और समय लेने वाला है। आप [`pyramid` लाइब्रेरी](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) से `auto_arima()` फ़ंक्शन का उपयोग करने पर विचार कर सकते हैं।

1. फिलहाल कुछ मैनुअल चयन आज़माएं ताकि एक अच्छा मॉडल मिल सके।

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    परिणामों की एक तालिका प्रिंट की जाती है।

आपने अपना पहला मॉडल बना लिया है! अब हमें इसे मूल्यांकन करने का तरीका खोजना होगा।

### अपने मॉडल का मूल्यांकन करें

अपने मॉडल का मूल्यांकन करने के लिए, आप तथाकथित `वॉक फॉरवर्ड` मान्यता कर सकते हैं। व्यवहार में, समय श्रृंखला मॉडल हर बार जब नया डेटा उपलब्ध होता है तो पुनः प्रशिक्षित किए जाते हैं। यह मॉडल को प्रत्येक समय चरण पर सबसे अच्छा पूर्वानुमान बनाने की अनुमति देता है।

इस तकनीक का उपयोग करते हुए समय श्रृंखला की शुरुआत में, ट्रेन डेटा सेट पर मॉडल को प्रशिक्षित करें। फिर अगले समय चरण पर एक पूर्वानुमान बनाएं। पूर्वानुमान ज्ञात मान के खिलाफ मूल्यांकन किया जाता है। फिर ट्रेनिंग सेट को ज्ञात मान को शामिल करने के लिए विस्तारित किया जाता है और प्रक्रिया को दोहराया जाता है।

> नोट: अधिक कुशल प्रशिक्षण के लिए आपको ट्रेनिंग सेट विंडो को स्थिर रखना चाहिए ताकि हर बार जब आप ट्रेनिंग सेट में एक नया अवलोकन जोड़ें, तो आप सेट की शुरुआत से अवलोकन को हटा दें।

यह प्रक्रिया मॉडल के व्यवहारिक प्रदर्शन का अधिक मजबूत अनुमान प्रदान करती है। हालांकि, इतने सारे मॉडल बनाने की गणना लागत आती है। यदि डेटा छोटा है या मॉडल सरल है तो यह स्वीकार्य है, लेकिन बड़े पैमाने पर समस्या हो सकती है।

वॉक-फॉरवर्ड मान्यता समय श्रृंखला मॉडल मूल्यांकन का स्वर्ण मानक है और इसे आपके अपने प्रोजेक्ट्स के लिए अनुशंसित किया जाता है।

1. पहले, प्रत्येक HORIZON चरण के लिए एक टेस्ट डेटा पॉइंट बनाएं।

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | लोड | लोड+1 | लोड+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    डेटा को इसके होराइजन पॉइंट के अनुसार क्षैतिज रूप से स्थानांतरित किया जाता है।

1. स्लाइडिंग विंडो दृष्टिकोण का उपयोग करके अपने टेस्ट डेटा पर पूर्वानुमान बनाएं, जो टेस्ट डेटा की लंबाई के आकार में एक लूप में हो:

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

    आप प्रशिक्षण को होते हुए देख सकते हैं:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. पूर्वानुमान को वास्तविक लोड से तुलना करें:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    आउटपुट
    |     |            | टाइमस्टैम्प | h   | पूर्वानुमान | वास्तविक   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    घंटेवार डेटा के पूर्वानुमान को देखें, वास्तविक लोड की तुलना में। यह कितना सटीक है?

### मॉडल की सटीकता जांचें

अपने मॉडल की सटीकता की जांच करें सभी पूर्वानुमानों पर इसके औसत प्रतिशत त्रुटि (MAPE) का परीक्षण करके।
> **🧮 गणना को समझें**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) का उपयोग भविष्यवाणी की सटीकता को एक अनुपात के रूप में दिखाने के लिए किया जाता है, जिसे ऊपर दिए गए सूत्र द्वारा परिभाषित किया गया है। वास्तविक और अनुमानित के बीच का अंतर वास्तविक से विभाजित किया जाता है।  
> "इस गणना में लिए गए पूर्ण मान को समय के प्रत्येक पूर्वानुमानित बिंदु के लिए जोड़ा जाता है और फिट किए गए बिंदुओं की संख्या n से विभाजित किया जाता है।" [विकिपीडिया](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. कोड में समीकरण व्यक्त करें:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. एक चरण का MAPE निकालें:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    एक चरण का पूर्वानुमान MAPE:  0.5570581332313952 %

1. बहु-चरण पूर्वानुमान MAPE प्रिंट करें:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    एक अच्छा कम संख्या बेहतर है: ध्यान दें कि यदि किसी पूर्वानुमान का MAPE 10 है, तो इसका मतलब है कि यह 10% तक गलत है।

1. लेकिन हमेशा की तरह, इस प्रकार की सटीकता माप को दृश्य रूप में देखना आसान होता है, तो चलिए इसे प्लॉट करते हैं:

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

    ![एक टाइम सीरीज़ मॉडल](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 एक बहुत ही अच्छा प्लॉट, जो एक अच्छे सटीकता वाले मॉडल को दिखाता है। बहुत बढ़िया!

---

## 🚀चुनौती

टाइम सीरीज़ मॉडल की सटीकता की जांच करने के तरीकों में गहराई से जाएं। इस पाठ में हमने MAPE पर चर्चा की है, लेकिन क्या आप अन्य तरीकों का उपयोग कर सकते हैं? उनका शोध करें और उन्हें नोट करें। एक सहायक दस्तावेज़ [यहां](https://otexts.com/fpp2/accuracy.html) पाया जा सकता है।

## [पाठ के बाद की क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

यह पाठ ARIMA के साथ टाइम सीरीज़ पूर्वानुमान की केवल मूल बातें छूता है। [इस रिपॉजिटरी](https://microsoft.github.io/forecasting/) और इसके विभिन्न मॉडल प्रकारों में गहराई से जाकर अन्य तरीकों से टाइम सीरीज़ मॉडल बनाने के बारे में जानने के लिए समय निकालें।

## असाइनमेंट

[एक नया ARIMA मॉडल](assignment.md)

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  