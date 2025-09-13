<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T21:23:03+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "th"
}
-->
# การพยากรณ์ข้อมูลอนุกรมเวลาด้วย Support Vector Regressor

ในบทเรียนก่อนหน้านี้ คุณได้เรียนรู้วิธีใช้โมเดล ARIMA เพื่อทำการพยากรณ์ข้อมูลอนุกรมเวลา ในบทนี้ คุณจะได้เรียนรู้เกี่ยวกับโมเดล Support Vector Regressor ซึ่งเป็นโมเดลสำหรับการพยากรณ์ข้อมูลแบบต่อเนื่อง

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/) 

## บทนำ

ในบทเรียนนี้ คุณจะได้ค้นพบวิธีการสร้างโมเดลด้วย [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) สำหรับการพยากรณ์ หรือที่เรียกว่า **SVR: Support Vector Regressor** 

### SVR ในบริบทของข้อมูลอนุกรมเวลา [^1]

ก่อนที่จะเข้าใจความสำคัญของ SVR ในการพยากรณ์ข้อมูลอนุกรมเวลา ต่อไปนี้คือแนวคิดสำคัญที่คุณควรรู้:

- **Regression:** เทคนิคการเรียนรู้แบบมีผู้สอนที่ใช้ในการพยากรณ์ค่าต่อเนื่องจากชุดข้อมูลที่กำหนด แนวคิดคือการหาค่าที่เหมาะสมที่สุดในพื้นที่ฟีเจอร์ที่มีจุดข้อมูลมากที่สุด [คลิกที่นี่](https://en.wikipedia.org/wiki/Regression_analysis) เพื่อดูข้อมูลเพิ่มเติม
- **Support Vector Machine (SVM):** โมเดลการเรียนรู้แบบมีผู้สอนที่ใช้สำหรับการจัดประเภท การพยากรณ์ และการตรวจจับค่าผิดปกติ โมเดลนี้จะสร้างไฮเปอร์เพลนในพื้นที่ฟีเจอร์ ซึ่งในกรณีของการจัดประเภทจะทำหน้าที่เป็นเส้นแบ่ง และในกรณีของการพยากรณ์จะทำหน้าที่เป็นเส้นที่เหมาะสมที่สุด โดยทั่วไป SVM จะใช้ฟังก์ชัน Kernel เพื่อแปลงชุดข้อมูลไปยังพื้นที่ที่มีมิติสูงขึ้นเพื่อให้สามารถแยกได้ง่ายขึ้น [คลิกที่นี่](https://en.wikipedia.org/wiki/Support-vector_machine) เพื่อดูข้อมูลเพิ่มเติมเกี่ยวกับ SVM
- **Support Vector Regressor (SVR):** เป็นประเภทหนึ่งของ SVM ที่ใช้ในการหาค่าที่เหมาะสมที่สุด (ซึ่งในกรณีของ SVM คือไฮเปอร์เพลน) ที่มีจุดข้อมูลมากที่สุด

### ทำไมต้องใช้ SVR? [^1]

ในบทเรียนก่อนหน้านี้ คุณได้เรียนรู้เกี่ยวกับ ARIMA ซึ่งเป็นวิธีการทางสถิติที่ประสบความสำเร็จในการพยากรณ์ข้อมูลอนุกรมเวลาแบบเชิงเส้น อย่างไรก็ตาม ในหลายกรณี ข้อมูลอนุกรมเวลามีความ *ไม่เชิงเส้น* ซึ่งไม่สามารถจับคู่ได้ด้วยโมเดลเชิงเส้น ในกรณีเช่นนี้ ความสามารถของ SVM ในการพิจารณาความไม่เชิงเส้นในข้อมูลสำหรับงานพยากรณ์ทำให้ SVR ประสบความสำเร็จในการพยากรณ์ข้อมูลอนุกรมเวลา

## แบบฝึกหัด - สร้างโมเดล SVR

ขั้นตอนแรกสำหรับการเตรียมข้อมูลเหมือนกับบทเรียนก่อนหน้านี้เกี่ยวกับ [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) 

เปิดโฟลเดอร์ [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) ในบทเรียนนี้และค้นหาไฟล์ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) [^2]

1. รันโน้ตบุ๊กและนำเข้าไลบรารีที่จำเป็น: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. โหลดข้อมูลจากไฟล์ `/data/energy.csv` ลงใน Pandas dataframe และดูข้อมูล: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. สร้างกราฟข้อมูลพลังงานทั้งหมดที่มีตั้งแต่เดือนมกราคม 2012 ถึงธันวาคม 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![full data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   ตอนนี้เรามาสร้างโมเดล SVR กัน

### สร้างชุดข้อมูลสำหรับการฝึกและการทดสอบ

เมื่อข้อมูลของคุณถูกโหลดแล้ว คุณสามารถแยกข้อมูลออกเป็นชุดฝึกและชุดทดสอบ จากนั้นคุณจะปรับรูปร่างข้อมูลเพื่อสร้างชุดข้อมูลตามลำดับเวลา ซึ่งจำเป็นสำหรับ SVR คุณจะฝึกโมเดลของคุณบนชุดฝึก หลังจากที่โมเดลฝึกเสร็จแล้ว คุณจะประเมินความแม่นยำของมันบนชุดฝึก ชุดทดสอบ และชุดข้อมูลทั้งหมดเพื่อดูประสิทธิภาพโดยรวม คุณต้องมั่นใจว่าชุดทดสอบครอบคลุมช่วงเวลาที่เกิดขึ้นหลังจากชุดฝึกเพื่อให้แน่ใจว่าโมเดลไม่ได้รับข้อมูลจากช่วงเวลาในอนาคต [^2] (สถานการณ์นี้เรียกว่า *Overfitting*)

1. กำหนดช่วงเวลาสองเดือนตั้งแต่วันที่ 1 กันยายนถึง 31 ตุลาคม 2014 ให้เป็นชุดฝึก ชุดทดสอบจะครอบคลุมช่วงเวลาสองเดือนตั้งแต่วันที่ 1 พฤศจิกายนถึง 31 ธันวาคม 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. แสดงความแตกต่างด้วยกราฟ: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### เตรียมข้อมูลสำหรับการฝึก

ตอนนี้คุณต้องเตรียมข้อมูลสำหรับการฝึกโดยการกรองและปรับขนาดข้อมูล กรองชุดข้อมูลของคุณเพื่อรวมเฉพาะช่วงเวลาและคอลัมน์ที่คุณต้องการ และปรับขนาดเพื่อให้ข้อมูลอยู่ในช่วง 0,1

1. กรองชุดข้อมูลต้นฉบับเพื่อรวมเฉพาะช่วงเวลาที่กล่าวถึงข้างต้นต่อชุด และรวมเฉพาะคอลัมน์ 'load' และวันที่: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. ปรับขนาดข้อมูลชุดฝึกให้อยู่ในช่วง (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. ตอนนี้ปรับขนาดข้อมูลชุดทดสอบ: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### สร้างข้อมูลด้วยลำดับเวลา [^1]

สำหรับ SVR คุณต้องแปลงข้อมูลอินพุตให้อยู่ในรูปแบบ `[batch, timesteps]` ดังนั้น คุณจะปรับรูปร่าง `train_data` และ `test_data` ที่มีอยู่เพื่อให้มีมิติใหม่ที่อ้างถึงลำดับเวลา

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

สำหรับตัวอย่างนี้ เรากำหนด `timesteps = 5` ดังนั้น อินพุตของโมเดลคือข้อมูลสำหรับ 4 ลำดับเวลาแรก และเอาต์พุตจะเป็นข้อมูลสำหรับลำดับเวลาที่ 5

```python
timesteps=5
```

แปลงข้อมูลชุดฝึกเป็น 2D tensor โดยใช้ nested list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

แปลงข้อมูลชุดทดสอบเป็น 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

เลือกอินพุตและเอาต์พุตจากข้อมูลชุดฝึกและชุดทดสอบ:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### ใช้ SVR [^1]

ตอนนี้ถึงเวลาที่จะใช้ SVR หากต้องการอ่านเพิ่มเติมเกี่ยวกับการใช้งานนี้ คุณสามารถดูได้ที่ [เอกสารนี้](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) สำหรับการใช้งานของเรา เราจะทำตามขั้นตอนดังนี้:

1. กำหนดโมเดลโดยเรียก `SVR()` และส่งผ่านไฮเปอร์พารามิเตอร์ของโมเดล: kernel, gamma, c และ epsilon
2. เตรียมโมเดลสำหรับข้อมูลชุดฝึกโดยเรียกฟังก์ชัน `fit()`
3. ทำการพยากรณ์โดยเรียกฟังก์ชัน `predict()`

ตอนนี้เราจะสร้างโมเดล SVR โดยใช้ [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) และตั้งค่าไฮเปอร์พารามิเตอร์ gamma, C และ epsilon เป็น 0.5, 10 และ 0.05 ตามลำดับ

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### ฝึกโมเดลบนข้อมูลชุดฝึก [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### ทำการพยากรณ์ด้วยโมเดล [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

คุณได้สร้าง SVR ของคุณแล้ว! ตอนนี้เราต้องประเมินผลลัพธ์

### ประเมินโมเดลของคุณ [^1]

สำหรับการประเมินผล ขั้นแรกเราจะปรับขนาดข้อมูลกลับไปยังสเกลเดิม จากนั้นเพื่อเช็คประสิทธิภาพ เราจะสร้างกราฟเปรียบเทียบข้อมูลจริงและข้อมูลที่พยากรณ์ และพิมพ์ผลลัพธ์ MAPE

ปรับขนาดข้อมูลที่พยากรณ์และข้อมูลจริงกลับ:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### ตรวจสอบประสิทธิภาพโมเดลบนข้อมูลชุดฝึกและชุดทดสอบ [^1]

เราจะดึง timestamps จากชุดข้อมูลเพื่อแสดงในแกน x ของกราฟ โปรดทราบว่าเราใช้ ```timesteps-1``` ค่าแรกเป็นอินพุตสำหรับเอาต์พุตแรก ดังนั้น timestamps สำหรับเอาต์พุตจะเริ่มหลังจากนั้น

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

สร้างกราฟการพยากรณ์สำหรับข้อมูลชุดฝึก:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![training data prediction](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

พิมพ์ค่า MAPE สำหรับข้อมูลชุดฝึก

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

สร้างกราฟการพยากรณ์สำหรับข้อมูลชุดทดสอบ

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testing data prediction](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

พิมพ์ค่า MAPE สำหรับข้อมูลชุดทดสอบ

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 คุณได้ผลลัพธ์ที่ดีมากบนชุดข้อมูลทดสอบ!

### ตรวจสอบประสิทธิภาพโมเดลบนชุดข้อมูลทั้งหมด [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![full data prediction](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 กราฟที่สวยงามมาก แสดงให้เห็นว่าโมเดลมีความแม่นยำที่ดี ยอดเยี่ยมมาก!

---

## 🚀ความท้าทาย

- ลองปรับไฮเปอร์พารามิเตอร์ (gamma, C, epsilon) ขณะสร้างโมเดลและประเมินผลบนข้อมูลเพื่อดูว่าชุดไฮเปอร์พารามิเตอร์ใดให้ผลลัพธ์ที่ดีที่สุดบนข้อมูลชุดทดสอบ หากต้องการทราบข้อมูลเพิ่มเติมเกี่ยวกับไฮเปอร์พารามิเตอร์เหล่านี้ คุณสามารถดูเอกสาร [ที่นี่](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) 
- ลองใช้ฟังก์ชัน kernel แบบต่างๆ สำหรับโมเดลและวิเคราะห์ประสิทธิภาพของพวกมันบนชุดข้อมูล เอกสารที่เป็นประโยชน์สามารถดูได้ [ที่นี่](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)
- ลองใช้ค่าที่แตกต่างกันสำหรับ `timesteps` เพื่อให้โมเดลย้อนกลับไปดูข้อมูลเพื่อทำการพยากรณ์

## [แบบทดสอบหลังเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

บทเรียนนี้เป็นการแนะนำการใช้ SVR สำหรับการพยากรณ์ข้อมูลอนุกรมเวลา หากต้องการอ่านเพิ่มเติมเกี่ยวกับ SVR คุณสามารถดูได้ที่ [บล็อกนี้](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) เอกสารนี้ [scikit-learn documentation](https://scikit-learn.org/stable/modules/svm.html) ให้คำอธิบายที่ครอบคลุมมากขึ้นเกี่ยวกับ SVM โดยทั่วไป [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) และรายละเอียดการใช้งานอื่นๆ เช่น [ฟังก์ชัน kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) ที่สามารถใช้ได้และพารามิเตอร์ของพวกมัน

## งานที่ได้รับมอบหมาย

[โมเดล SVR ใหม่](assignment.md)

## เครดิต

[^1]: ข้อความ โค้ด และผลลัพธ์ในส่วนนี้ได้รับการสนับสนุนโดย [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)  
[^2]: ข้อความ โค้ด และผลลัพธ์ในส่วนนี้นำมาจาก [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาต้นทางควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้