<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T19:04:15+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về dự đoán chuỗi thời gian

![Tóm tắt về chuỗi thời gian trong một bản vẽ phác thảo](../../../../sketchnotes/ml-timeseries.png)

> Bản vẽ phác thảo bởi [Tomomi Imura](https://www.twitter.com/girlie_mac)

Trong bài học này và bài học tiếp theo, bạn sẽ tìm hiểu một chút về dự đoán chuỗi thời gian, một phần thú vị và có giá trị trong kho kiến thức của nhà khoa học ML, nhưng lại ít được biết đến hơn so với các chủ đề khác. Dự đoán chuỗi thời gian giống như một "quả cầu pha lê": dựa trên hiệu suất trong quá khứ của một biến số như giá cả, bạn có thể dự đoán giá trị tiềm năng của nó trong tương lai.

[![Giới thiệu về dự đoán chuỗi thời gian](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Giới thiệu về dự đoán chuỗi thời gian")

> 🎥 Nhấp vào hình ảnh trên để xem video về dự đoán chuỗi thời gian

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

Đây là một lĩnh vực hữu ích và thú vị với giá trị thực tế đối với doanh nghiệp, nhờ vào ứng dụng trực tiếp của nó trong các vấn đề về giá cả, hàng tồn kho và chuỗi cung ứng. Mặc dù các kỹ thuật học sâu đã bắt đầu được sử dụng để có thêm những hiểu biết nhằm dự đoán hiệu suất tương lai tốt hơn, dự đoán chuỗi thời gian vẫn là một lĩnh vực được thông tin rất nhiều bởi các kỹ thuật ML cổ điển.

> Chương trình học hữu ích về chuỗi thời gian của Penn State có thể được tìm thấy [tại đây](https://online.stat.psu.edu/stat510/lesson/1)

## Giới thiệu

Giả sử bạn quản lý một loạt các đồng hồ đỗ xe thông minh cung cấp dữ liệu về tần suất sử dụng và thời gian sử dụng theo thời gian.

> Điều gì sẽ xảy ra nếu bạn có thể dự đoán, dựa trên hiệu suất trong quá khứ của đồng hồ, giá trị tương lai của nó theo quy luật cung và cầu?

Dự đoán chính xác thời điểm hành động để đạt được mục tiêu của bạn là một thách thức có thể được giải quyết bằng dự đoán chuỗi thời gian. Mặc dù việc tăng giá vào thời điểm đông đúc khi mọi người đang tìm chỗ đỗ xe có thể không làm họ hài lòng, nhưng đó sẽ là một cách chắc chắn để tạo ra doanh thu để làm sạch đường phố!

Hãy cùng khám phá một số loại thuật toán chuỗi thời gian và bắt đầu một notebook để làm sạch và chuẩn bị dữ liệu. Dữ liệu bạn sẽ phân tích được lấy từ cuộc thi dự đoán GEFCom2014. Nó bao gồm 3 năm dữ liệu tải điện và nhiệt độ hàng giờ từ năm 2012 đến năm 2014. Dựa trên các mẫu lịch sử của tải điện và nhiệt độ, bạn có thể dự đoán giá trị tải điện trong tương lai.

Trong ví dụ này, bạn sẽ học cách dự đoán một bước thời gian trước, chỉ sử dụng dữ liệu tải lịch sử. Tuy nhiên, trước khi bắt đầu, sẽ rất hữu ích để hiểu những gì đang diễn ra phía sau.

## Một số định nghĩa

Khi gặp thuật ngữ "chuỗi thời gian", bạn cần hiểu cách sử dụng của nó trong một số ngữ cảnh khác nhau.

🎓 **Chuỗi thời gian**

Trong toán học, "chuỗi thời gian là một loạt các điểm dữ liệu được lập chỉ mục (hoặc liệt kê hoặc vẽ đồ thị) theo thứ tự thời gian. Thông thường nhất, chuỗi thời gian là một chuỗi được lấy tại các điểm cách đều nhau theo thời gian." Một ví dụ về chuỗi thời gian là giá trị đóng cửa hàng ngày của [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Việc sử dụng đồ thị chuỗi thời gian và mô hình thống kê thường được gặp trong xử lý tín hiệu, dự báo thời tiết, dự đoán động đất và các lĩnh vực khác nơi các sự kiện xảy ra và các điểm dữ liệu có thể được vẽ theo thời gian.

🎓 **Phân tích chuỗi thời gian**

Phân tích chuỗi thời gian là việc phân tích dữ liệu chuỗi thời gian đã đề cập ở trên. Dữ liệu chuỗi thời gian có thể có các dạng khác nhau, bao gồm 'chuỗi thời gian bị gián đoạn', phát hiện các mẫu trong sự phát triển của chuỗi thời gian trước và sau một sự kiện gián đoạn. Loại phân tích cần thiết cho chuỗi thời gian phụ thuộc vào bản chất của dữ liệu. Dữ liệu chuỗi thời gian có thể là một chuỗi số hoặc ký tự.

Phân tích được thực hiện sử dụng nhiều phương pháp khác nhau, bao gồm miền tần số và miền thời gian, tuyến tính và phi tuyến tính, và nhiều hơn nữa. [Tìm hiểu thêm](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) về các cách phân tích loại dữ liệu này.

🎓 **Dự đoán chuỗi thời gian**

Dự đoán chuỗi thời gian là việc sử dụng một mô hình để dự đoán giá trị tương lai dựa trên các mẫu được hiển thị bởi dữ liệu đã thu thập trước đó khi nó xảy ra trong quá khứ. Mặc dù có thể sử dụng các mô hình hồi quy để khám phá dữ liệu chuỗi thời gian, với các chỉ số thời gian làm biến x trên đồ thị, dữ liệu như vậy tốt nhất nên được phân tích bằng các loại mô hình đặc biệt.

Dữ liệu chuỗi thời gian là một danh sách các quan sát có thứ tự, không giống như dữ liệu có thể được phân tích bằng hồi quy tuyến tính. Loại phổ biến nhất là ARIMA, một từ viết tắt của "Autoregressive Integrated Moving Average".

[Mô hình ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "liên kết giá trị hiện tại của một chuỗi với các giá trị trong quá khứ và các lỗi dự đoán trong quá khứ." Chúng phù hợp nhất để phân tích dữ liệu miền thời gian, nơi dữ liệu được sắp xếp theo thời gian.

> Có một số loại mô hình ARIMA, bạn có thể tìm hiểu [tại đây](https://people.duke.edu/~rnau/411arim.htm) và sẽ được đề cập trong bài học tiếp theo.

Trong bài học tiếp theo, bạn sẽ xây dựng một mô hình ARIMA sử dụng [Chuỗi thời gian đơn biến](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), tập trung vào một biến số thay đổi giá trị theo thời gian. Một ví dụ về loại dữ liệu này là [bộ dữ liệu này](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) ghi lại nồng độ CO2 hàng tháng tại Đài quan sát Mauna Loa:

|   CO2   | YearMonth | Year  | Month |
| :-----: | :-------: | :---: | :---: |
| 330.62  |  1975.04  | 1975  |   1   |
| 331.40  |  1975.13  | 1975  |   2   |
| 331.87  |  1975.21  | 1975  |   3   |
| 333.18  |  1975.29  | 1975  |   4   |
| 333.92  |  1975.38  | 1975  |   5   |
| 333.43  |  1975.46  | 1975  |   6   |
| 331.85  |  1975.54  | 1975  |   7   |
| 330.01  |  1975.63  | 1975  |   8   |
| 328.51  |  1975.71  | 1975  |   9   |
| 328.41  |  1975.79  | 1975  |  10   |
| 329.25  |  1975.88  | 1975  |  11   |
| 330.97  |  1975.96  | 1975  |  12   |

✅ Xác định biến số thay đổi theo thời gian trong bộ dữ liệu này

## Các đặc điểm của dữ liệu chuỗi thời gian cần xem xét

Khi xem dữ liệu chuỗi thời gian, bạn có thể nhận thấy rằng nó có [một số đặc điểm nhất định](https://online.stat.psu.edu/stat510/lesson/1/1.1) mà bạn cần xem xét và giảm thiểu để hiểu rõ hơn các mẫu của nó. Nếu bạn coi dữ liệu chuỗi thời gian như một tín hiệu tiềm năng mà bạn muốn phân tích, các đặc điểm này có thể được coi là "nhiễu". Bạn thường cần giảm "nhiễu" này bằng cách bù đắp một số đặc điểm này bằng các kỹ thuật thống kê.

Dưới đây là một số khái niệm bạn nên biết để làm việc với chuỗi thời gian:

🎓 **Xu hướng**

Xu hướng được định nghĩa là sự tăng hoặc giảm có thể đo lường theo thời gian. [Đọc thêm](https://machinelearningmastery.com/time-series-trends-in-python). Trong ngữ cảnh chuỗi thời gian, đó là cách sử dụng và, nếu cần thiết, loại bỏ xu hướng khỏi chuỗi thời gian của bạn.

🎓 **[Tính thời vụ](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Tính thời vụ được định nghĩa là các biến động định kỳ, chẳng hạn như sự tăng đột biến trong kỳ nghỉ lễ có thể ảnh hưởng đến doanh số bán hàng. [Xem thêm](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) về cách các loại đồ thị khác nhau hiển thị tính thời vụ trong dữ liệu.

🎓 **Giá trị ngoại lai**

Giá trị ngoại lai nằm cách xa sự biến đổi dữ liệu tiêu chuẩn.

🎓 **Chu kỳ dài hạn**

Không phụ thuộc vào tính thời vụ, dữ liệu có thể hiển thị một chu kỳ dài hạn như suy thoái kinh tế kéo dài hơn một năm.

🎓 **Biến đổi không đổi**

Theo thời gian, một số dữ liệu hiển thị các biến động không đổi, chẳng hạn như mức sử dụng năng lượng mỗi ngày và đêm.

🎓 **Thay đổi đột ngột**

Dữ liệu có thể hiển thị một sự thay đổi đột ngột cần phân tích thêm. Ví dụ, việc đóng cửa đột ngột các doanh nghiệp do COVID đã gây ra những thay đổi trong dữ liệu.

✅ Đây là một [đồ thị chuỗi thời gian mẫu](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) hiển thị chi tiêu tiền tệ trong trò chơi hàng ngày trong vài năm. Bạn có thể xác định bất kỳ đặc điểm nào được liệt kê ở trên trong dữ liệu này không?

![Chi tiêu tiền tệ trong trò chơi](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Bài tập - bắt đầu với dữ liệu sử dụng năng lượng

Hãy bắt đầu tạo một mô hình chuỗi thời gian để dự đoán mức sử dụng năng lượng trong tương lai dựa trên mức sử dụng trong quá khứ.

> Dữ liệu trong ví dụ này được lấy từ cuộc thi dự đoán GEFCom2014. Nó bao gồm 3 năm dữ liệu tải điện và nhiệt độ hàng giờ từ năm 2012 đến năm 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli và Rob J. Hyndman, "Dự báo năng lượng xác suất: Cuộc thi Dự báo Năng lượng Toàn cầu 2014 và hơn thế nữa", Tạp chí Dự báo Quốc tế, tập 32, số 3, trang 896-913, tháng 7-tháng 9, 2016.

1. Trong thư mục `working` của bài học này, mở tệp _notebook.ipynb_. Bắt đầu bằng cách thêm các thư viện sẽ giúp bạn tải và trực quan hóa dữ liệu

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Lưu ý, bạn đang sử dụng các tệp từ thư mục `common` đi kèm, thiết lập môi trường của bạn và xử lý việc tải dữ liệu.

2. Tiếp theo, kiểm tra dữ liệu dưới dạng dataframe bằng cách gọi `load_data()` và `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Bạn có thể thấy rằng có hai cột đại diện cho ngày và tải:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Bây giờ, vẽ đồ thị dữ liệu bằng cách gọi `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Đồ thị năng lượng](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Tiếp theo, vẽ đồ thị tuần đầu tiên của tháng 7 năm 2014, bằng cách cung cấp nó làm đầu vào cho `energy` theo mẫu `[từ ngày]: [đến ngày]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Tháng 7](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Một đồ thị tuyệt đẹp! Hãy xem các đồ thị này và xem liệu bạn có thể xác định bất kỳ đặc điểm nào được liệt kê ở trên không. Chúng ta có thể suy luận gì khi trực quan hóa dữ liệu?

Trong bài học tiếp theo, bạn sẽ tạo một mô hình ARIMA để tạo một số dự đoán.

---

## 🚀Thử thách

Lập danh sách tất cả các ngành và lĩnh vực nghiên cứu mà bạn có thể nghĩ rằng sẽ được hưởng lợi từ dự đoán chuỗi thời gian. Bạn có thể nghĩ ra một ứng dụng của các kỹ thuật này trong nghệ thuật? Trong Kinh tế lượng? Sinh thái học? Bán lẻ? Công nghiệp? Tài chính? Còn ở đâu nữa?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Mặc dù chúng ta sẽ không đề cập đến chúng ở đây, mạng nơ-ron đôi khi được sử dụng để nâng cao các phương pháp cổ điển của dự đoán chuỗi thời gian. Đọc thêm về chúng [trong bài viết này](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Bài tập

[Trực quan hóa thêm một số chuỗi thời gian](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.