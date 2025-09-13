<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T21:17:56+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "th"
}
-->
# การพยากรณ์อนุกรมเวลาโดยใช้ ARIMA

ในบทเรียนก่อนหน้านี้ คุณได้เรียนรู้เกี่ยวกับการพยากรณ์อนุกรมเวลาและโหลดชุดข้อมูลที่แสดงการเปลี่ยนแปลงของโหลดไฟฟ้าในช่วงเวลาหนึ่ง

[![Introduction to ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduction to ARIMA")

> 🎥 คลิกที่ภาพด้านบนเพื่อดูวิดีโอ: บทนำสั้น ๆ เกี่ยวกับโมเดล ARIMA ตัวอย่างในวิดีโอใช้ R แต่แนวคิดสามารถนำไปใช้ได้ทั่วไป

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)

## บทนำ

ในบทเรียนนี้ คุณจะได้เรียนรู้วิธีเฉพาะในการสร้างโมเดลด้วย [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) โมเดล ARIMA เหมาะอย่างยิ่งสำหรับการปรับข้อมูลที่แสดง [non-stationarity](https://wikipedia.org/wiki/Stationary_process)

## แนวคิดทั่วไป

เพื่อที่จะทำงานกับ ARIMA มีแนวคิดบางอย่างที่คุณจำเป็นต้องรู้:

- 🎓 **Stationarity**. ในบริบททางสถิติ Stationarity หมายถึงข้อมูลที่การแจกแจงไม่เปลี่ยนแปลงเมื่อเลื่อนเวลา ข้อมูลที่ไม่เป็น Stationary จะแสดงการเปลี่ยนแปลงเนื่องจากแนวโน้มที่ต้องแปลงเพื่อวิเคราะห์ ตัวอย่างเช่น Seasonality สามารถทำให้เกิดการเปลี่ยนแปลงในข้อมูลและสามารถกำจัดได้โดยกระบวนการ 'seasonal-differencing'

- 🎓 **[Differencing](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. การ Differencing ข้อมูลในบริบททางสถิติ หมายถึงกระบวนการแปลงข้อมูลที่ไม่เป็น Stationary ให้เป็น Stationary โดยการลบแนวโน้มที่ไม่คงที่ "Differencing ช่วยลบการเปลี่ยนแปลงในระดับของอนุกรมเวลา กำจัดแนวโน้มและ Seasonality และทำให้ค่าเฉลี่ยของอนุกรมเวลามีเสถียรภาพ" [บทความโดย Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA ในบริบทของอนุกรมเวลา

มาทำความเข้าใจส่วนต่าง ๆ ของ ARIMA เพื่อให้เข้าใจว่าโมเดลนี้ช่วยเราในการสร้างแบบจำลองอนุกรมเวลาและช่วยในการพยากรณ์ได้อย่างไร

- **AR - AutoRegressive**. โมเดล AutoRegressive ตามชื่อ หมายถึงการมองย้อนกลับไปในเวลาเพื่อวิเคราะห์ค่าก่อนหน้าในข้อมูลของคุณและทำการสันนิษฐานเกี่ยวกับค่าเหล่านั้น ค่าก่อนหน้าเหล่านี้เรียกว่า 'lags' ตัวอย่างเช่น ข้อมูลที่แสดงยอดขายดินสอรายเดือน ยอดขายรวมของแต่ละเดือนจะถือว่าเป็น 'ตัวแปรที่เปลี่ยนแปลง' ในชุดข้อมูล โมเดลนี้ถูกสร้างขึ้นโดย "ตัวแปรที่เปลี่ยนแปลงที่สนใจถูกวิเคราะห์กับค่าก่อนหน้าของตัวเอง" [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrated**. แตกต่างจากโมเดล 'ARMA' ที่คล้ายกัน 'I' ใน ARIMA หมายถึงลักษณะ *[integrated](https://wikipedia.org/wiki/Order_of_integration)* ข้อมูลจะถูก 'integrated' เมื่อมีการใช้ขั้นตอน Differencing เพื่อกำจัด non-stationarity

- **MA - Moving Average**. ลักษณะ [moving-average](https://wikipedia.org/wiki/Moving-average_model) ของโมเดลนี้หมายถึงตัวแปรผลลัพธ์ที่ถูกกำหนดโดยการสังเกตค่าปัจจุบันและค่าก่อนหน้าของ lags

สรุป: ARIMA ถูกใช้เพื่อสร้างโมเดลที่ปรับให้เข้ากับรูปแบบพิเศษของข้อมูลอนุกรมเวลาให้ใกล้เคียงที่สุด

## แบบฝึกหัด - สร้างโมเดล ARIMA

เปิดโฟลเดอร์ [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) ในบทเรียนนี้และค้นหาไฟล์ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb)

1. รัน notebook เพื่อโหลดไลบรารี Python `statsmodels`; คุณจะต้องใช้ไลบรารีนี้สำหรับโมเดล ARIMA

1. โหลดไลบรารีที่จำเป็น

1. ตอนนี้โหลดไลบรารีเพิ่มเติมที่มีประโยชน์สำหรับการสร้างกราฟข้อมูล:

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

1. โหลดข้อมูลจากไฟล์ `/data/energy.csv` ลงใน Pandas dataframe และดูข้อมูล:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. สร้างกราฟข้อมูลพลังงานทั้งหมดที่มีตั้งแต่เดือนมกราคม 2012 ถึงเดือนธันวาคม 2014 ไม่มีอะไรน่าประหลาดใจเพราะเราเห็นข้อมูลนี้ในบทเรียนที่แล้ว:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ตอนนี้มาสร้างโมเดลกัน!

### สร้างชุดข้อมูลสำหรับการฝึกและการทดสอบ

ตอนนี้ข้อมูลของคุณถูกโหลดแล้ว คุณสามารถแยกมันออกเป็นชุดข้อมูลสำหรับการฝึกและการทดสอบ คุณจะฝึกโมเดลของคุณด้วยชุดข้อมูลสำหรับการฝึก และหลังจากโมเดลฝึกเสร็จแล้ว คุณจะประเมินความแม่นยำของมันโดยใช้ชุดข้อมูลสำหรับการทดสอบ คุณต้องมั่นใจว่าชุดข้อมูลสำหรับการทดสอบครอบคลุมช่วงเวลาที่เกิดขึ้นหลังจากชุดข้อมูลสำหรับการฝึกเพื่อให้มั่นใจว่าโมเดลไม่ได้รับข้อมูลจากช่วงเวลาที่จะเกิดขึ้นในอนาคต

1. กำหนดช่วงเวลาสองเดือนตั้งแต่วันที่ 1 กันยายนถึง 31 ตุลาคม 2014 ให้เป็นชุดข้อมูลสำหรับการฝึก ชุดข้อมูลสำหรับการทดสอบจะรวมช่วงเวลาสองเดือนตั้งแต่วันที่ 1 พฤศจิกายนถึง 31 ธันวาคม 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    เนื่องจากข้อมูลนี้สะท้อนการบริโภคพลังงานรายวัน มีรูปแบบตามฤดูกาลที่ชัดเจน แต่การบริโภคมีความคล้ายคลึงกับการบริโภคในวันล่าสุดมากที่สุด

1. สร้างกราฟเปรียบเทียบ:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![training and testing data](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    ดังนั้น การใช้ช่วงเวลาที่ค่อนข้างเล็กสำหรับการฝึกข้อมูลควรเพียงพอ

    > หมายเหตุ: เนื่องจากฟังก์ชันที่เราใช้ในการปรับโมเดล ARIMA ใช้การตรวจสอบภายในตัวอย่างระหว่างการปรับ เราจะละเว้นข้อมูลการตรวจสอบ

### เตรียมข้อมูลสำหรับการฝึก

ตอนนี้คุณต้องเตรียมข้อมูลสำหรับการฝึกโดยการกรองและปรับขนาดข้อมูล กรองชุดข้อมูลของคุณเพื่อรวมเฉพาะช่วงเวลาที่ต้องการและคอลัมน์ที่จำเป็น และปรับขนาดเพื่อให้แน่ใจว่าข้อมูลถูกฉายลงในช่วง 0,1

1. กรองชุดข้อมูลต้นฉบับเพื่อรวมเฉพาะช่วงเวลาที่กล่าวถึงต่อชุดและรวมเฉพาะคอลัมน์ 'load' และวันที่ที่จำเป็น:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    คุณสามารถดูรูปร่างของข้อมูล:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. ปรับขนาดข้อมูลให้อยู่ในช่วง (0, 1)

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. สร้างกราฟเปรียบเทียบข้อมูลต้นฉบับกับข้อมูลที่ปรับขนาด:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > ข้อมูลต้นฉบับ

    ![scaled](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > ข้อมูลที่ปรับขนาด

1. ตอนนี้คุณได้ปรับเทียบข้อมูลที่ปรับขนาดแล้ว คุณสามารถปรับขนาดข้อมูลสำหรับการทดสอบ:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ใช้ ARIMA

ถึงเวลาที่จะใช้ ARIMA! ตอนนี้คุณจะใช้ไลบรารี `statsmodels` ที่คุณติดตั้งไว้ก่อนหน้านี้

ตอนนี้คุณต้องทำตามขั้นตอนหลายขั้นตอน

   1. กำหนดโมเดลโดยเรียก `SARIMAX()` และส่งพารามิเตอร์ของโมเดล: พารามิเตอร์ p, d, และ q และพารามิเตอร์ P, D, และ Q
   2. เตรียมโมเดลสำหรับข้อมูลการฝึกโดยเรียกฟังก์ชัน fit()
   3. ทำการพยากรณ์โดยเรียกฟังก์ชัน `forecast()` และระบุจำนวนขั้นตอน (หรือ `horizon`) ที่ต้องการพยากรณ์

> 🎓 พารามิเตอร์เหล่านี้มีไว้เพื่ออะไร? ในโมเดล ARIMA มี 3 พารามิเตอร์ที่ใช้ช่วยสร้างแบบจำลองลักษณะสำคัญของอนุกรมเวลา: seasonality, trend, และ noise พารามิเตอร์เหล่านี้คือ:

`p`: พารามิเตอร์ที่เกี่ยวข้องกับลักษณะ auto-regressive ของโมเดล ซึ่งรวมค่าที่ผ่านมา
`d`: พารามิเตอร์ที่เกี่ยวข้องกับส่วน integrated ของโมเดล ซึ่งมีผลต่อจำนวน *differencing* (🎓 จำ differencing ได้ไหม 👆?) ที่จะใช้กับอนุกรมเวลา
`q`: พารามิเตอร์ที่เกี่ยวข้องกับส่วน moving-average ของโมเดล

> หมายเหตุ: หากข้อมูลของคุณมีลักษณะตามฤดูกาล - ซึ่งข้อมูลนี้มี - เราใช้โมเดล ARIMA ตามฤดูกาล (SARIMA) ในกรณีนี้คุณต้องใช้ชุดพารามิเตอร์อีกชุดหนึ่ง: `P`, `D`, และ `Q` ซึ่งอธิบายความสัมพันธ์เดียวกันกับ `p`, `d`, และ `q` แต่สอดคล้องกับส่วนตามฤดูกาลของโมเดล

1. เริ่มต้นโดยตั้งค่าค่าของ horizon ที่คุณต้องการ ลองใช้ 3 ชั่วโมง:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    การเลือกค่าที่ดีที่สุดสำหรับพารามิเตอร์ของโมเดล ARIMA อาจเป็นเรื่องท้าทายเนื่องจากเป็นเรื่องที่ค่อนข้างอัตวิสัยและใช้เวลามาก คุณอาจพิจารณาใช้ฟังก์ชัน `auto_arima()` จากไลบรารี [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html)

1. สำหรับตอนนี้ลองเลือกค่าด้วยตนเองเพื่อค้นหาโมเดลที่ดี

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    ตารางผลลัพธ์จะถูกพิมพ์ออกมา

คุณได้สร้างโมเดลแรกของคุณแล้ว! ตอนนี้เราต้องหาวิธีประเมินมัน

### ประเมินโมเดลของคุณ

เพื่อประเมินโมเดลของคุณ คุณสามารถใช้การตรวจสอบที่เรียกว่า `walk forward` validation ในทางปฏิบัติ โมเดลอนุกรมเวลาจะถูกฝึกใหม่ทุกครั้งที่มีข้อมูลใหม่เข้ามา สิ่งนี้ช่วยให้โมเดลสามารถทำการพยากรณ์ที่ดีที่สุดในแต่ละขั้นตอนเวลา

เริ่มต้นที่จุดเริ่มต้นของอนุกรมเวลาโดยใช้เทคนิคนี้ ฝึกโมเดลด้วยชุดข้อมูลสำหรับการฝึก จากนั้นทำการพยากรณ์ในขั้นตอนเวลาถัดไป การพยากรณ์จะถูกประเมินเทียบกับค่าที่ทราบ ชุดข้อมูลสำหรับการฝึกจะถูกขยายเพื่อรวมค่าที่ทราบและกระบวนการจะถูกทำซ้ำ

> หมายเหตุ: คุณควรรักษาหน้าต่างชุดข้อมูลสำหรับการฝึกให้คงที่เพื่อการฝึกที่มีประสิทธิภาพมากขึ้น เพื่อให้ทุกครั้งที่คุณเพิ่มการสังเกตใหม่ลงในชุดข้อมูลสำหรับการฝึก คุณจะลบการสังเกตจากจุดเริ่มต้นของชุดข้อมูล

กระบวนการนี้ให้การประมาณที่แข็งแกร่งขึ้นเกี่ยวกับวิธีที่โมเดลจะทำงานในทางปฏิบัติ อย่างไรก็ตาม มันมาพร้อมกับต้นทุนการคำนวณในการสร้างโมเดลจำนวนมาก สิ่งนี้เป็นที่ยอมรับได้หากข้อมูลมีขนาดเล็กหรือโมเดลมีความเรียบง่าย แต่สามารถเป็นปัญหาในระดับใหญ่

การตรวจสอบแบบ walk-forward เป็นมาตรฐานทองคำของการประเมินโมเดลอนุกรมเวลาและแนะนำสำหรับโครงการของคุณเอง

1. ก่อนอื่น สร้างจุดข้อมูลการทดสอบสำหรับแต่ละขั้นตอน HORIZON

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

    ข้อมูลจะถูกเลื่อนในแนวนอนตามจุด horizon

1. ทำการพยากรณ์ในข้อมูลการทดสอบของคุณโดยใช้วิธี sliding window ในลูปที่มีขนาดเท่ากับความยาวของข้อมูลการทดสอบ:

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

    คุณสามารถดูการฝึกที่เกิดขึ้น:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. เปรียบเทียบการพยากรณ์กับโหลดจริง:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    ผลลัพธ์
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    สังเกตการพยากรณ์ข้อมูลรายชั่วโมง เทียบกับโหลดจริง ความแม่นยำเป็นอย่างไร?

### ตรวจสอบความแม่นยำของโมเดล

ตรวจสอบความแม่นยำของโมเดลของคุณโดยทดสอบค่า mean absolute percentage error (MAPE) จากการพยากรณ์ทั้งหมด
> **🧮 แสดงคณิตศาสตร์**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) ใช้เพื่อแสดงความแม่นยำของการพยากรณ์ในรูปแบบอัตราส่วนที่กำหนดโดยสูตรด้านบน ความแตกต่างระหว่างค่าจริงและค่าที่พยากรณ์  
ถูกหารด้วยค่าจริง  
"ค่าที่เป็นบวกในสูตรนี้จะถูกนำมารวมกันสำหรับทุกจุดที่พยากรณ์ในช่วงเวลา และหารด้วยจำนวนจุดที่พอดี n" [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. แสดงสมการในโค้ด:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. คำนวณ MAPE ของการพยากรณ์หนึ่งขั้น:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE ของการพยากรณ์หนึ่งขั้น:  0.5570581332313952 %

1. แสดงผล MAPE ของการพยากรณ์หลายขั้น:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    ตัวเลขที่ต่ำถือว่าดี: พิจารณาว่าการพยากรณ์ที่มี MAPE เท่ากับ 10 หมายถึงค่าที่พยากรณ์ผิดไป 10%

1. แต่เช่นเคย การวัดความแม่นยำแบบนี้จะเห็นได้ง่ายขึ้นเมื่อดูภาพ ดังนั้นมาลองพล็อตกราฟกัน:

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

    ![โมเดลซีรีส์เวลา](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 กราฟที่ดูดีมาก แสดงให้เห็นว่าโมเดลมีความแม่นยำสูง ยอดเยี่ยมมาก!

---

## 🚀ความท้าทาย

สำรวจวิธีการทดสอบความแม่นยำของโมเดลซีรีส์เวลา ในบทเรียนนี้เราได้พูดถึง MAPE แต่มีวิธีอื่นที่คุณสามารถใช้ได้หรือไม่? ลองค้นคว้าและอธิบายเพิ่มเติม เอกสารที่เป็นประโยชน์สามารถดูได้ [ที่นี่](https://otexts.com/fpp2/accuracy.html)

## [แบบทดสอบหลังบทเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตัวเอง

บทเรียนนี้ครอบคลุมเพียงพื้นฐานของการพยากรณ์ซีรีส์เวลาด้วย ARIMA ใช้เวลาเพิ่มเติมเพื่อเพิ่มพูนความรู้ของคุณโดยสำรวจ [repository นี้](https://microsoft.github.io/forecasting/) และโมเดลประเภทต่าง ๆ เพื่อเรียนรู้วิธีอื่น ๆ ในการสร้างโมเดลซีรีส์เวลา

## งานที่ได้รับมอบหมาย

[โมเดล ARIMA ใหม่](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามอย่างเต็มที่เพื่อให้การแปลมีความถูกต้อง โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาต้นทางควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามนุษย์มืออาชีพ เราจะไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้