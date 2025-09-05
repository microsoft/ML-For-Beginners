<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T18:59:51+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "vi"
}
-->
# Dự đoán chuỗi thời gian với ARIMA

Trong bài học trước, bạn đã tìm hiểu một chút về dự đoán chuỗi thời gian và tải một tập dữ liệu cho thấy sự biến động của tải điện qua một khoảng thời gian.

[![Giới thiệu về ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Giới thiệu về ARIMA")

> 🎥 Nhấp vào hình ảnh trên để xem video: Giới thiệu ngắn gọn về mô hình ARIMA. Ví dụ được thực hiện bằng R, nhưng các khái niệm là phổ quát.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Giới thiệu

Trong bài học này, bạn sẽ khám phá một cách cụ thể để xây dựng mô hình với [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Mô hình ARIMA đặc biệt phù hợp để xử lý dữ liệu cho thấy [tính không dừng](https://wikipedia.org/wiki/Stationary_process).

## Các khái niệm chung

Để làm việc với ARIMA, có một số khái niệm bạn cần biết:

- 🎓 **Tính dừng**. Trong ngữ cảnh thống kê, tính dừng đề cập đến dữ liệu có phân phối không thay đổi khi dịch chuyển theo thời gian. Dữ liệu không dừng, do đó, cho thấy sự biến động do xu hướng và cần được biến đổi để phân tích. Tính thời vụ, ví dụ, có thể gây ra sự biến động trong dữ liệu và có thể được loại bỏ bằng quá trình 'khác biệt hóa theo mùa'.

- 🎓 **[Khác biệt hóa](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Khác biệt hóa dữ liệu, trong ngữ cảnh thống kê, đề cập đến quá trình biến đổi dữ liệu không dừng để làm cho nó trở thành dữ liệu dừng bằng cách loại bỏ xu hướng không cố định. "Khác biệt hóa loại bỏ sự thay đổi trong mức độ của chuỗi thời gian, loại bỏ xu hướng và tính thời vụ, và do đó ổn định giá trị trung bình của chuỗi thời gian." [Bài báo của Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA trong ngữ cảnh chuỗi thời gian

Hãy phân tích các phần của ARIMA để hiểu rõ hơn cách nó giúp chúng ta mô hình hóa chuỗi thời gian và đưa ra dự đoán.

- **AR - cho AutoRegressive (tự hồi quy)**. Mô hình tự hồi quy, như tên gọi, nhìn 'lại' thời gian để phân tích các giá trị trước đó trong dữ liệu của bạn và đưa ra giả định về chúng. Các giá trị trước đó này được gọi là 'độ trễ'. Một ví dụ là dữ liệu cho thấy doanh số bán bút chì hàng tháng. Tổng doanh số mỗi tháng sẽ được coi là một 'biến tiến hóa' trong tập dữ liệu. Mô hình này được xây dựng khi "biến tiến hóa được hồi quy trên các giá trị trễ (tức là giá trị trước đó) của chính nó." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - cho Integrated (tích hợp)**. Khác với mô hình 'ARMA' tương tự, 'I' trong ARIMA đề cập đến khía cạnh *[tích hợp](https://wikipedia.org/wiki/Order_of_integration)* của nó. Dữ liệu được 'tích hợp' khi các bước khác biệt hóa được áp dụng để loại bỏ tính không dừng.

- **MA - cho Moving Average (trung bình động)**. Khía cạnh [trung bình động](https://wikipedia.org/wiki/Moving-average_model) của mô hình này đề cập đến biến đầu ra được xác định bằng cách quan sát các giá trị hiện tại và quá khứ của độ trễ.

Tóm lại: ARIMA được sử dụng để tạo một mô hình phù hợp nhất với dạng đặc biệt của dữ liệu chuỗi thời gian.

## Bài tập - xây dựng mô hình ARIMA

Mở thư mục [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) trong bài học này và tìm tệp [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Chạy notebook để tải thư viện Python `statsmodels`; bạn sẽ cần thư viện này cho mô hình ARIMA.

1. Tải các thư viện cần thiết.

1. Bây giờ, tải thêm một số thư viện hữu ích để vẽ dữ liệu:

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

1. Tải dữ liệu từ tệp `/data/energy.csv` vào một dataframe Pandas và xem qua:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Vẽ tất cả dữ liệu năng lượng có sẵn từ tháng 1 năm 2012 đến tháng 12 năm 2014. Không có gì bất ngờ vì chúng ta đã thấy dữ liệu này trong bài học trước:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Bây giờ, hãy xây dựng một mô hình!

### Tạo tập dữ liệu huấn luyện và kiểm tra

Bây giờ dữ liệu của bạn đã được tải, bạn có thể tách nó thành tập huấn luyện và tập kiểm tra. Bạn sẽ huấn luyện mô hình của mình trên tập huấn luyện. Như thường lệ, sau khi mô hình hoàn thành huấn luyện, bạn sẽ đánh giá độ chính xác của nó bằng tập kiểm tra. Bạn cần đảm bảo rằng tập kiểm tra bao phủ một khoảng thời gian sau tập huấn luyện để đảm bảo rằng mô hình không nhận được thông tin từ các khoảng thời gian trong tương lai.

1. Phân bổ khoảng thời gian hai tháng từ ngày 1 tháng 9 đến ngày 31 tháng 10 năm 2014 cho tập huấn luyện. Tập kiểm tra sẽ bao gồm khoảng thời gian hai tháng từ ngày 1 tháng 11 đến ngày 31 tháng 12 năm 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Vì dữ liệu này phản ánh mức tiêu thụ năng lượng hàng ngày, có một mô hình thời vụ mạnh mẽ, nhưng mức tiêu thụ gần giống nhất với mức tiêu thụ trong những ngày gần đây.

1. Hiển thị sự khác biệt:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![dữ liệu huấn luyện và kiểm tra](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Do đó, sử dụng một khoảng thời gian tương đối nhỏ để huấn luyện dữ liệu nên đủ.

    > Lưu ý: Vì hàm chúng ta sử dụng để khớp mô hình ARIMA sử dụng xác thực trong mẫu trong quá trình khớp, chúng ta sẽ bỏ qua dữ liệu xác thực.

### Chuẩn bị dữ liệu để huấn luyện

Bây giờ, bạn cần chuẩn bị dữ liệu để huấn luyện bằng cách lọc và chuẩn hóa dữ liệu của mình. Lọc tập dữ liệu của bạn để chỉ bao gồm các khoảng thời gian và cột cần thiết, và chuẩn hóa để đảm bảo dữ liệu được chiếu trong khoảng 0,1.

1. Lọc tập dữ liệu gốc để chỉ bao gồm các khoảng thời gian đã đề cập ở trên cho mỗi tập và chỉ bao gồm cột 'load' cần thiết cùng với ngày:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Bạn có thể xem hình dạng của dữ liệu:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Chuẩn hóa dữ liệu để nằm trong khoảng (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Hiển thị dữ liệu gốc so với dữ liệu đã chuẩn hóa:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![gốc](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Dữ liệu gốc

    ![chuẩn hóa](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Dữ liệu đã chuẩn hóa

1. Bây giờ bạn đã hiệu chỉnh dữ liệu đã chuẩn hóa, bạn có thể chuẩn hóa dữ liệu kiểm tra:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Triển khai ARIMA

Đã đến lúc triển khai ARIMA! Bây giờ bạn sẽ sử dụng thư viện `statsmodels` mà bạn đã cài đặt trước đó.

Bây giờ bạn cần thực hiện một số bước:

   1. Định nghĩa mô hình bằng cách gọi `SARIMAX()` và truyền các tham số mô hình: các tham số p, d, và q, và các tham số P, D, và Q.
   2. Chuẩn bị mô hình cho dữ liệu huấn luyện bằng cách gọi hàm fit().
   3. Dự đoán bằng cách gọi hàm `forecast()` và chỉ định số bước (đường chân trời) để dự đoán.

> 🎓 Các tham số này dùng để làm gì? Trong mô hình ARIMA, có 3 tham số được sử dụng để giúp mô hình hóa các khía cạnh chính của chuỗi thời gian: tính thời vụ, xu hướng, và nhiễu. Các tham số này là:

`p`: tham số liên quan đến khía cạnh tự hồi quy của mô hình, kết hợp các giá trị *quá khứ*.
`d`: tham số liên quan đến phần tích hợp của mô hình, ảnh hưởng đến số lượng *khác biệt hóa* (🎓 nhớ khác biệt hóa 👆?) được áp dụng cho chuỗi thời gian.
`q`: tham số liên quan đến phần trung bình động của mô hình.

> Lưu ý: Nếu dữ liệu của bạn có khía cạnh thời vụ - như dữ liệu này - , chúng ta sử dụng mô hình ARIMA thời vụ (SARIMA). Trong trường hợp đó, bạn cần sử dụng một bộ tham số khác: `P`, `D`, và `Q` mô tả các liên kết tương tự như `p`, `d`, và `q`, nhưng tương ứng với các thành phần thời vụ của mô hình.

1. Bắt đầu bằng cách đặt giá trị đường chân trời mong muốn của bạn. Hãy thử 3 giờ:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Việc chọn các giá trị tốt nhất cho các tham số của mô hình ARIMA có thể khó khăn vì nó khá chủ quan và tốn thời gian. Bạn có thể cân nhắc sử dụng hàm `auto_arima()` từ thư viện [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Hiện tại hãy thử một số lựa chọn thủ công để tìm một mô hình tốt.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Một bảng kết quả được in ra.

Bạn đã xây dựng mô hình đầu tiên của mình! Bây giờ chúng ta cần tìm cách đánh giá nó.

### Đánh giá mô hình của bạn

Để đánh giá mô hình của bạn, bạn có thể thực hiện cái gọi là xác thực `walk forward`. Trong thực tế, các mô hình chuỗi thời gian được huấn luyện lại mỗi khi có dữ liệu mới. Điều này cho phép mô hình đưa ra dự đoán tốt nhất tại mỗi bước thời gian.

Bắt đầu từ đầu chuỗi thời gian bằng kỹ thuật này, huấn luyện mô hình trên tập dữ liệu huấn luyện. Sau đó đưa ra dự đoán cho bước thời gian tiếp theo. Dự đoán được đánh giá dựa trên giá trị đã biết. Tập huấn luyện sau đó được mở rộng để bao gồm giá trị đã biết và quá trình được lặp lại.

> Lưu ý: Bạn nên giữ cửa sổ tập huấn luyện cố định để huấn luyện hiệu quả hơn, để mỗi lần bạn thêm một quan sát mới vào tập huấn luyện, bạn loại bỏ quan sát từ đầu tập.

Quá trình này cung cấp một ước tính mạnh mẽ hơn về cách mô hình sẽ hoạt động trong thực tế. Tuy nhiên, nó đi kèm với chi phí tính toán khi tạo ra nhiều mô hình. Điều này chấp nhận được nếu dữ liệu nhỏ hoặc mô hình đơn giản, nhưng có thể là vấn đề ở quy mô lớn.

Xác thực walk-forward là tiêu chuẩn vàng để đánh giá mô hình chuỗi thời gian và được khuyến nghị cho các dự án của bạn.

1. Đầu tiên, tạo một điểm dữ liệu kiểm tra cho mỗi bước HORIZON.

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

    Dữ liệu được dịch ngang theo điểm đường chân trời của nó.

1. Dự đoán trên dữ liệu kiểm tra của bạn bằng cách sử dụng cách tiếp cận cửa sổ trượt trong một vòng lặp có kích thước bằng độ dài dữ liệu kiểm tra:

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

    Bạn có thể xem quá trình huấn luyện đang diễn ra:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. So sánh các dự đoán với tải thực tế:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Kết quả
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Quan sát dự đoán dữ liệu hàng giờ, so với tải thực tế. Độ chính xác của nó như thế nào?

### Kiểm tra độ chính xác của mô hình

Kiểm tra độ chính xác của mô hình của bạn bằng cách kiểm tra lỗi phần trăm tuyệt đối trung bình (MAPE) của nó trên tất cả các dự đoán.
> **🧮 Hiển thị công thức toán học**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) được sử dụng để thể hiện độ chính xác của dự đoán dưới dạng tỷ lệ được định nghĩa bởi công thức trên. Sự khác biệt giữa giá trị thực tế và giá trị dự đoán được chia cho giá trị thực tế. "Giá trị tuyệt đối trong phép tính này được cộng lại cho mỗi điểm dự đoán theo thời gian và chia cho số lượng điểm được khớp n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Biểu diễn phương trình trong mã:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Tính MAPE của một bước:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE dự báo một bước:  0.5570581332313952 %

1. In MAPE dự báo nhiều bước:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Một con số thấp là tốt: hãy xem xét rằng một dự báo có MAPE là 10 thì sai lệch 10%.

1. Nhưng như thường lệ, cách dễ nhất để thấy loại đo lường độ chính xác này là trực quan hóa, vì vậy hãy vẽ biểu đồ:

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

    ![một mô hình chuỗi thời gian](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Một biểu đồ rất đẹp, cho thấy một mô hình với độ chính xác tốt. Làm tốt lắm!

---

## 🚀Thử thách

Khám phá các cách để kiểm tra độ chính xác của một mô hình chuỗi thời gian. Chúng ta đã đề cập đến MAPE trong bài học này, nhưng liệu có các phương pháp khác mà bạn có thể sử dụng? Nghiên cứu chúng và chú thích lại. Một tài liệu hữu ích có thể được tìm thấy [tại đây](https://otexts.com/fpp2/accuracy.html)

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Bài học này chỉ đề cập đến những kiến thức cơ bản về Dự báo Chuỗi Thời Gian với ARIMA. Hãy dành thời gian để mở rộng kiến thức của bạn bằng cách khám phá [kho lưu trữ này](https://microsoft.github.io/forecasting/) và các loại mô hình khác nhau để học cách xây dựng các mô hình Chuỗi Thời Gian khác.

## Bài tập

[Một mô hình ARIMA mới](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.