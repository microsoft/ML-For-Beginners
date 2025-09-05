<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T18:55:18+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "vi"
}
-->
# Xây dựng mô hình hồi quy sử dụng Scikit-learn: chuẩn bị và trực quan hóa dữ liệu

![Infographic trực quan hóa dữ liệu](../../../../2-Regression/2-Data/images/data-visualization.png)

Infographic bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Câu hỏi trước bài học](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bài học này có sẵn bằng R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Giới thiệu

Bây giờ bạn đã có các công cụ cần thiết để bắt đầu xây dựng mô hình học máy với Scikit-learn, bạn đã sẵn sàng để bắt đầu đặt câu hỏi về dữ liệu của mình. Khi làm việc với dữ liệu và áp dụng các giải pháp ML, điều rất quan trọng là phải hiểu cách đặt câu hỏi đúng để khai thác tiềm năng của tập dữ liệu một cách hiệu quả.

Trong bài học này, bạn sẽ học:

- Cách chuẩn bị dữ liệu cho việc xây dựng mô hình.
- Cách sử dụng Matplotlib để trực quan hóa dữ liệu.

## Đặt câu hỏi đúng về dữ liệu của bạn

Câu hỏi bạn cần trả lời sẽ quyết định loại thuật toán ML mà bạn sẽ sử dụng. Và chất lượng của câu trả lời bạn nhận được sẽ phụ thuộc rất nhiều vào bản chất của dữ liệu.

Hãy xem [dữ liệu](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) được cung cấp cho bài học này. Bạn có thể mở tệp .csv này trong VS Code. Một cái nhìn nhanh sẽ cho thấy rằng có các ô trống và sự pha trộn giữa dữ liệu dạng chuỗi và số. Ngoài ra còn có một cột kỳ lạ gọi là 'Package' với dữ liệu là sự pha trộn giữa 'sacks', 'bins' và các giá trị khác. Thực tế, dữ liệu này khá lộn xộn.

[![ML cho người mới bắt đầu - Cách phân tích và làm sạch tập dữ liệu](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML cho người mới bắt đầu - Cách phân tích và làm sạch tập dữ liệu")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về cách chuẩn bị dữ liệu cho bài học này.

Thực tế, không thường xuyên bạn nhận được một tập dữ liệu hoàn toàn sẵn sàng để sử dụng để tạo mô hình ML ngay lập tức. Trong bài học này, bạn sẽ học cách chuẩn bị một tập dữ liệu thô bằng cách sử dụng các thư viện Python tiêu chuẩn. Bạn cũng sẽ học các kỹ thuật khác nhau để trực quan hóa dữ liệu.

## Nghiên cứu trường hợp: 'thị trường bí ngô'

Trong thư mục này, bạn sẽ tìm thấy một tệp .csv trong thư mục gốc `data` có tên [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) bao gồm 1757 dòng dữ liệu về thị trường bí ngô, được phân loại theo thành phố. Đây là dữ liệu thô được trích xuất từ [Báo cáo Tiêu chuẩn Thị trường Cây Trồng Đặc Biệt](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) do Bộ Nông nghiệp Hoa Kỳ phân phối.

### Chuẩn bị dữ liệu

Dữ liệu này thuộc phạm vi công cộng. Nó có thể được tải xuống dưới dạng nhiều tệp riêng biệt, theo từng thành phố, từ trang web USDA. Để tránh quá nhiều tệp riêng biệt, chúng tôi đã gộp tất cả dữ liệu thành phố vào một bảng tính, do đó chúng tôi đã _chuẩn bị_ dữ liệu một chút. Tiếp theo, hãy xem xét kỹ hơn dữ liệu.

### Dữ liệu bí ngô - kết luận ban đầu

Bạn nhận thấy gì về dữ liệu này? Bạn đã thấy rằng có sự pha trộn giữa chuỗi, số, ô trống và các giá trị kỳ lạ mà bạn cần hiểu.

Bạn có thể đặt câu hỏi nào về dữ liệu này, sử dụng kỹ thuật hồi quy? Ví dụ: "Dự đoán giá của một quả bí ngô được bán trong một tháng cụ thể". Nhìn lại dữ liệu, có một số thay đổi bạn cần thực hiện để tạo cấu trúc dữ liệu cần thiết cho nhiệm vụ này.

## Bài tập - phân tích dữ liệu bí ngô

Hãy sử dụng [Pandas](https://pandas.pydata.org/) (tên viết tắt của `Python Data Analysis`), một công cụ rất hữu ích để định hình dữ liệu, để phân tích và chuẩn bị dữ liệu bí ngô này.

### Đầu tiên, kiểm tra các ngày bị thiếu

Bạn sẽ cần thực hiện các bước để kiểm tra các ngày bị thiếu:

1. Chuyển đổi các ngày sang định dạng tháng (đây là ngày tháng kiểu Mỹ, nên định dạng là `MM/DD/YYYY`).
2. Trích xuất tháng vào một cột mới.

Mở tệp _notebook.ipynb_ trong Visual Studio Code và nhập bảng tính vào một dataframe Pandas mới.

1. Sử dụng hàm `head()` để xem năm hàng đầu tiên.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Bạn sẽ sử dụng hàm nào để xem năm hàng cuối cùng?

1. Kiểm tra xem có dữ liệu bị thiếu trong dataframe hiện tại không:

    ```python
    pumpkins.isnull().sum()
    ```

    Có dữ liệu bị thiếu, nhưng có thể nó sẽ không ảnh hưởng đến nhiệm vụ hiện tại.

1. Để làm cho dataframe của bạn dễ làm việc hơn, chỉ chọn các cột bạn cần, sử dụng hàm `loc` để trích xuất từ dataframe gốc một nhóm hàng (được truyền làm tham số đầu tiên) và cột (được truyền làm tham số thứ hai). Biểu thức `:` trong trường hợp dưới đây có nghĩa là "tất cả các hàng".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Thứ hai, xác định giá trung bình của bí ngô

Hãy nghĩ về cách xác định giá trung bình của một quả bí ngô trong một tháng cụ thể. Bạn sẽ chọn những cột nào cho nhiệm vụ này? Gợi ý: bạn sẽ cần 3 cột.

Giải pháp: lấy trung bình của các cột `Low Price` và `High Price` để điền vào cột Price mới, và chuyển đổi cột Date để chỉ hiển thị tháng. May mắn thay, theo kiểm tra ở trên, không có dữ liệu bị thiếu cho ngày tháng hoặc giá cả.

1. Để tính trung bình, thêm đoạn mã sau:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Bạn có thể in bất kỳ dữ liệu nào bạn muốn kiểm tra bằng cách sử dụng `print(month)`.

2. Bây giờ, sao chép dữ liệu đã chuyển đổi của bạn vào một dataframe Pandas mới:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    In dataframe của bạn sẽ hiển thị một tập dữ liệu sạch sẽ, gọn gàng mà bạn có thể sử dụng để xây dựng mô hình hồi quy mới.

### Nhưng khoan đã! Có điều gì đó kỳ lạ ở đây

Nếu bạn nhìn vào cột `Package`, bí ngô được bán theo nhiều cấu hình khác nhau. Một số được bán theo đơn vị '1 1/9 bushel', một số theo '1/2 bushel', một số theo quả, một số theo pound, và một số trong các hộp lớn với các kích thước khác nhau.

> Bí ngô dường như rất khó để cân đo một cách nhất quán

Đào sâu vào dữ liệu gốc, thật thú vị khi bất kỳ mục nào có `Unit of Sale` bằng 'EACH' hoặc 'PER BIN' cũng có kiểu `Package` theo inch, theo bin, hoặc 'each'. Bí ngô dường như rất khó để cân đo một cách nhất quán, vì vậy hãy lọc chúng bằng cách chỉ chọn bí ngô có chuỗi 'bushel' trong cột `Package`.

1. Thêm bộ lọc ở đầu tệp, dưới phần nhập .csv ban đầu:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Nếu bạn in dữ liệu bây giờ, bạn có thể thấy rằng bạn chỉ nhận được khoảng 415 dòng dữ liệu chứa bí ngô theo bushel.

### Nhưng khoan đã! Còn một việc nữa cần làm

Bạn có nhận thấy rằng lượng bushel thay đổi theo từng dòng không? Bạn cần chuẩn hóa giá để hiển thị giá theo bushel, vì vậy hãy thực hiện một số phép toán để chuẩn hóa.

1. Thêm các dòng sau sau khối tạo dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Theo [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), trọng lượng của một bushel phụ thuộc vào loại sản phẩm, vì đây là một phép đo thể tích. "Một bushel cà chua, ví dụ, được cho là nặng 56 pound... Lá và rau xanh chiếm nhiều không gian hơn với ít trọng lượng hơn, vì vậy một bushel rau bina chỉ nặng 20 pound." Điều này khá phức tạp! Hãy không chuyển đổi bushel sang pound, thay vào đó tính giá theo bushel. Tất cả nghiên cứu về bushel bí ngô này, tuy nhiên, cho thấy việc hiểu rõ bản chất của dữ liệu là rất quan trọng!

Bây giờ, bạn có thể phân tích giá theo đơn vị dựa trên đo lường bushel của chúng. Nếu bạn in dữ liệu một lần nữa, bạn có thể thấy cách nó được chuẩn hóa.

✅ Bạn có nhận thấy rằng bí ngô được bán theo nửa bushel rất đắt không? Bạn có thể tìm ra lý do tại sao không? Gợi ý: bí ngô nhỏ thường đắt hơn bí ngô lớn, có lẽ vì có nhiều quả hơn trong một bushel, do không gian trống bị chiếm bởi một quả bí ngô lớn rỗng.

## Chiến lược trực quan hóa

Một phần vai trò của nhà khoa học dữ liệu là thể hiện chất lượng và bản chất của dữ liệu mà họ đang làm việc. Để làm điều này, họ thường tạo ra các hình ảnh trực quan thú vị, hoặc biểu đồ, đồ thị, và sơ đồ, hiển thị các khía cạnh khác nhau của dữ liệu. Bằng cách này, họ có thể trực quan hóa các mối quan hệ và khoảng trống mà nếu không sẽ khó phát hiện.

[![ML cho người mới bắt đầu - Cách trực quan hóa dữ liệu với Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML cho người mới bắt đầu - Cách trực quan hóa dữ liệu với Matplotlib")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về cách trực quan hóa dữ liệu cho bài học này.

Các hình ảnh trực quan cũng có thể giúp xác định kỹ thuật học máy phù hợp nhất với dữ liệu. Một biểu đồ phân tán có vẻ theo một đường thẳng, ví dụ, cho thấy rằng dữ liệu là ứng viên tốt cho bài tập hồi quy tuyến tính.

Một thư viện trực quan hóa dữ liệu hoạt động tốt trong Jupyter notebooks là [Matplotlib](https://matplotlib.org/) (mà bạn cũng đã thấy trong bài học trước).

> Tìm hiểu thêm về trực quan hóa dữ liệu trong [các hướng dẫn này](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Bài tập - thử nghiệm với Matplotlib

Hãy thử tạo một số biểu đồ cơ bản để hiển thị dataframe mới mà bạn vừa tạo. Một biểu đồ đường cơ bản sẽ hiển thị điều gì?

1. Nhập Matplotlib ở đầu tệp, dưới phần nhập Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Chạy lại toàn bộ notebook để làm mới.
1. Ở cuối notebook, thêm một ô để vẽ dữ liệu dưới dạng hộp:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Biểu đồ phân tán hiển thị mối quan hệ giữa giá và tháng](../../../../2-Regression/2-Data/images/scatterplot.png)

    Đây có phải là biểu đồ hữu ích không? Có điều gì về nó làm bạn ngạc nhiên không?

    Nó không đặc biệt hữu ích vì tất cả những gì nó làm là hiển thị dữ liệu của bạn dưới dạng một loạt các điểm trong một tháng nhất định.

### Làm cho nó hữu ích

Để các biểu đồ hiển thị dữ liệu hữu ích, bạn thường cần nhóm dữ liệu theo cách nào đó. Hãy thử tạo một biểu đồ mà trục y hiển thị các tháng và dữ liệu thể hiện sự phân bố của dữ liệu.

1. Thêm một ô để tạo biểu đồ cột nhóm:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Biểu đồ cột hiển thị mối quan hệ giữa giá và tháng](../../../../2-Regression/2-Data/images/barchart.png)

    Đây là một hình ảnh trực quan dữ liệu hữu ích hơn! Dường như nó chỉ ra rằng giá cao nhất cho bí ngô xảy ra vào tháng 9 và tháng 10. Điều này có đúng với mong đợi của bạn không? Tại sao hoặc tại sao không?

---

## 🚀Thử thách

Khám phá các loại hình ảnh trực quan khác nhau mà Matplotlib cung cấp. Loại nào phù hợp nhất cho các bài toán hồi quy?

## [Câu hỏi sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Hãy xem xét các cách khác nhau để trực quan hóa dữ liệu. Lập danh sách các thư viện khác nhau có sẵn và ghi chú loại nào tốt nhất cho các loại nhiệm vụ cụ thể, ví dụ trực quan hóa 2D so với trực quan hóa 3D. Bạn phát hiện ra điều gì?

## Bài tập

[Khám phá trực quan hóa](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.