<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T19:08:01+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "vi"
}
-->
# Dự đoán chuỗi thời gian với Support Vector Regressor

Trong bài học trước, bạn đã học cách sử dụng mô hình ARIMA để dự đoán chuỗi thời gian. Bây giờ, bạn sẽ tìm hiểu về mô hình Support Vector Regressor, một mô hình hồi quy được sử dụng để dự đoán dữ liệu liên tục.

## [Câu hỏi trước bài học](https://ff-quizzes.netlify.app/en/ml/) 

## Giới thiệu

Trong bài học này, bạn sẽ khám phá một cách cụ thể để xây dựng mô hình với [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) cho hồi quy, hay **SVR: Support Vector Regressor**.

### SVR trong bối cảnh chuỗi thời gian [^1]

Trước khi hiểu được tầm quan trọng của SVR trong dự đoán chuỗi thời gian, đây là một số khái niệm quan trọng mà bạn cần biết:

- **Hồi quy:** Kỹ thuật học có giám sát để dự đoán giá trị liên tục từ một tập hợp đầu vào. Ý tưởng là tìm một đường cong (hoặc đường thẳng) trong không gian đặc trưng có số lượng điểm dữ liệu tối đa. [Nhấn vào đây](https://en.wikipedia.org/wiki/Regression_analysis) để biết thêm thông tin.
- **Support Vector Machine (SVM):** Một loại mô hình học máy có giám sát được sử dụng cho phân loại, hồi quy và phát hiện điểm bất thường. Mô hình là một siêu phẳng trong không gian đặc trưng, trong trường hợp phân loại nó hoạt động như một ranh giới, và trong trường hợp hồi quy nó hoạt động như đường thẳng phù hợp nhất. Trong SVM, một hàm Kernel thường được sử dụng để chuyển đổi tập dữ liệu sang không gian có số chiều cao hơn, để chúng có thể dễ dàng phân tách. [Nhấn vào đây](https://en.wikipedia.org/wiki/Support-vector_machine) để biết thêm thông tin về SVM.
- **Support Vector Regressor (SVR):** Một loại SVM, để tìm đường thẳng phù hợp nhất (trong trường hợp của SVM là siêu phẳng) có số lượng điểm dữ liệu tối đa.

### Tại sao lại là SVR? [^1]

Trong bài học trước, bạn đã học về ARIMA, một phương pháp thống kê tuyến tính rất thành công để dự đoán dữ liệu chuỗi thời gian. Tuy nhiên, trong nhiều trường hợp, dữ liệu chuỗi thời gian có tính *phi tuyến*, điều mà các mô hình tuyến tính không thể ánh xạ được. Trong những trường hợp như vậy, khả năng của SVM trong việc xem xét tính phi tuyến của dữ liệu cho các nhiệm vụ hồi quy khiến SVR trở nên thành công trong dự đoán chuỗi thời gian.

## Bài tập - xây dựng mô hình SVR

Các bước đầu tiên để chuẩn bị dữ liệu giống như bài học trước về [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Mở thư mục [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) trong bài học này và tìm tệp [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Chạy notebook và nhập các thư viện cần thiết: [^2]

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

2. Tải dữ liệu từ tệp `/data/energy.csv` vào một dataframe của Pandas và xem qua: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Vẽ biểu đồ tất cả dữ liệu năng lượng có sẵn từ tháng 1 năm 2012 đến tháng 12 năm 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![full data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Bây giờ, hãy xây dựng mô hình SVR của chúng ta.

### Tạo tập dữ liệu huấn luyện và kiểm tra

Bây giờ dữ liệu của bạn đã được tải, bạn có thể tách nó thành tập huấn luyện và kiểm tra. Sau đó, bạn sẽ định hình lại dữ liệu để tạo một tập dữ liệu dựa trên bước thời gian, điều này sẽ cần thiết cho SVR. Bạn sẽ huấn luyện mô hình của mình trên tập huấn luyện. Sau khi mô hình hoàn thành việc huấn luyện, bạn sẽ đánh giá độ chính xác của nó trên tập huấn luyện, tập kiểm tra và sau đó là toàn bộ tập dữ liệu để xem hiệu suất tổng thể. Bạn cần đảm bảo rằng tập kiểm tra bao gồm một khoảng thời gian sau tập huấn luyện để đảm bảo rằng mô hình không thu thập thông tin từ các khoảng thời gian trong tương lai [^2] (một tình huống được gọi là *Overfitting*).

1. Phân bổ khoảng thời gian hai tháng từ ngày 1 tháng 9 đến ngày 31 tháng 10 năm 2014 cho tập huấn luyện. Tập kiểm tra sẽ bao gồm khoảng thời gian hai tháng từ ngày 1 tháng 11 đến ngày 31 tháng 12 năm 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Hiển thị sự khác biệt: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Chuẩn bị dữ liệu để huấn luyện

Bây giờ, bạn cần chuẩn bị dữ liệu để huấn luyện bằng cách lọc và chuẩn hóa dữ liệu của mình. Lọc tập dữ liệu để chỉ bao gồm các khoảng thời gian và cột cần thiết, và chuẩn hóa để đảm bảo dữ liệu được chiếu trong khoảng 0,1.

1. Lọc tập dữ liệu gốc để chỉ bao gồm các khoảng thời gian đã đề cập ở trên cho mỗi tập và chỉ bao gồm cột 'load' cần thiết cùng với ngày tháng: [^2]

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
   
2. Chuẩn hóa dữ liệu huấn luyện để nằm trong khoảng (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Bây giờ, bạn chuẩn hóa dữ liệu kiểm tra: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Tạo dữ liệu với bước thời gian [^1]

Đối với SVR, bạn chuyển đổi dữ liệu đầu vào thành dạng `[batch, timesteps]`. Vì vậy, bạn định hình lại `train_data` và `test_data` hiện tại sao cho có một chiều mới đề cập đến các bước thời gian.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Trong ví dụ này, chúng ta lấy `timesteps = 5`. Vì vậy, đầu vào cho mô hình là dữ liệu của 4 bước thời gian đầu tiên, và đầu ra sẽ là dữ liệu của bước thời gian thứ 5.

```python
timesteps=5
```

Chuyển đổi dữ liệu huấn luyện thành tensor 2D bằng cách sử dụng list comprehension lồng nhau:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Chuyển đổi dữ liệu kiểm tra thành tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Chọn đầu vào và đầu ra từ dữ liệu huấn luyện và kiểm tra:

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

### Triển khai SVR [^1]

Bây giờ, đã đến lúc triển khai SVR. Để đọc thêm về triển khai này, bạn có thể tham khảo [tài liệu này](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Đối với triển khai của chúng ta, chúng ta thực hiện các bước sau:

  1. Định nghĩa mô hình bằng cách gọi `SVR()` và truyền vào các siêu tham số của mô hình: kernel, gamma, c và epsilon
  2. Chuẩn bị mô hình cho dữ liệu huấn luyện bằng cách gọi hàm `fit()`
  3. Thực hiện dự đoán bằng cách gọi hàm `predict()`

Bây giờ chúng ta tạo một mô hình SVR. Ở đây chúng ta sử dụng [kernel RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), và đặt các siêu tham số gamma, C và epsilon lần lượt là 0.5, 10 và 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Huấn luyện mô hình trên dữ liệu huấn luyện [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Thực hiện dự đoán của mô hình [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Bạn đã xây dựng SVR của mình! Bây giờ chúng ta cần đánh giá nó.

### Đánh giá mô hình của bạn [^1]

Để đánh giá, trước tiên chúng ta sẽ chuẩn hóa lại dữ liệu về thang đo ban đầu. Sau đó, để kiểm tra hiệu suất, chúng ta sẽ vẽ biểu đồ chuỗi thời gian gốc và dự đoán, và cũng in kết quả MAPE.

Chuẩn hóa lại dữ liệu dự đoán và đầu ra gốc:

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

#### Kiểm tra hiệu suất mô hình trên dữ liệu huấn luyện và kiểm tra [^1]

Chúng ta trích xuất các dấu thời gian từ tập dữ liệu để hiển thị trên trục x của biểu đồ. Lưu ý rằng chúng ta đang sử dụng ```timesteps-1``` giá trị đầu tiên làm đầu vào cho đầu ra đầu tiên, vì vậy các dấu thời gian cho đầu ra sẽ bắt đầu sau đó.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Vẽ biểu đồ dự đoán cho dữ liệu huấn luyện:

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

In MAPE cho dữ liệu huấn luyện

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Vẽ biểu đồ dự đoán cho dữ liệu kiểm tra

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testing data prediction](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

In MAPE cho dữ liệu kiểm tra

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Bạn đã đạt được kết quả rất tốt trên tập dữ liệu kiểm tra!

### Kiểm tra hiệu suất mô hình trên toàn bộ tập dữ liệu [^1]

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

🏆 Biểu đồ rất đẹp, cho thấy một mô hình với độ chính xác tốt. Làm tốt lắm!

---

## 🚀Thử thách

- Thử điều chỉnh các siêu tham số (gamma, C, epsilon) khi tạo mô hình và đánh giá trên dữ liệu để xem bộ siêu tham số nào cho kết quả tốt nhất trên dữ liệu kiểm tra. Để biết thêm về các siêu tham số này, bạn có thể tham khảo tài liệu [tại đây](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Thử sử dụng các hàm kernel khác nhau cho mô hình và phân tích hiệu suất của chúng trên tập dữ liệu. Một tài liệu hữu ích có thể được tìm thấy [tại đây](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Thử sử dụng các giá trị khác nhau cho `timesteps` để mô hình nhìn lại và thực hiện dự đoán.

## [Câu hỏi sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Bài học này nhằm giới thiệu ứng dụng của SVR trong dự đoán chuỗi thời gian. Để đọc thêm về SVR, bạn có thể tham khảo [blog này](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Tài liệu [scikit-learn này](https://scikit-learn.org/stable/modules/svm.html) cung cấp một giải thích toàn diện hơn về SVM nói chung, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) và cũng các chi tiết triển khai khác như các [hàm kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) khác nhau có thể được sử dụng, và các tham số của chúng.

## Bài tập

[Một mô hình SVR mới](assignment.md)

## Tín dụng

[^1]: Văn bản, mã và kết quả trong phần này được đóng góp bởi [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Văn bản, mã và kết quả trong phần này được lấy từ [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.