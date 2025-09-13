<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T19:58:59+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về phân loại

Trong bốn bài học này, bạn sẽ khám phá một trọng tâm cơ bản của học máy cổ điển - _phân loại_. Chúng ta sẽ cùng tìm hiểu cách sử dụng các thuật toán phân loại khác nhau với một tập dữ liệu về các món ăn tuyệt vời của châu Á và Ấn Độ. Hy vọng bạn đã sẵn sàng để thưởng thức!

![chỉ một chút thôi!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Hãy cùng tôn vinh các món ăn châu Á trong những bài học này! Hình ảnh bởi [Jen Looper](https://twitter.com/jenlooper)

Phân loại là một hình thức [học có giám sát](https://wikipedia.org/wiki/Supervised_learning) có nhiều điểm tương đồng với các kỹ thuật hồi quy. Nếu học máy là về việc dự đoán giá trị hoặc tên của các đối tượng bằng cách sử dụng tập dữ liệu, thì phân loại thường chia thành hai nhóm: _phân loại nhị phân_ và _phân loại đa lớp_.

[![Giới thiệu về phân loại](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Giới thiệu về phân loại")

> 🎥 Nhấp vào hình ảnh trên để xem video: John Guttag của MIT giới thiệu về phân loại

Hãy nhớ:

- **Hồi quy tuyến tính** giúp bạn dự đoán mối quan hệ giữa các biến và đưa ra dự đoán chính xác về vị trí mà một điểm dữ liệu mới sẽ nằm trong mối quan hệ với đường thẳng đó. Ví dụ, bạn có thể dự đoán _giá của một quả bí ngô vào tháng 9 so với tháng 12_.
- **Hồi quy logistic** giúp bạn khám phá "các danh mục nhị phân": ở mức giá này, _quả bí ngô này có màu cam hay không màu cam_?

Phân loại sử dụng các thuật toán khác nhau để xác định các cách khác nhau nhằm gán nhãn hoặc lớp cho một điểm dữ liệu. Hãy cùng làm việc với dữ liệu về các món ăn này để xem liệu, bằng cách quan sát một nhóm nguyên liệu, chúng ta có thể xác định nguồn gốc của món ăn đó hay không.

## [Câu hỏi trước bài học](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bài học này có sẵn bằng R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Giới thiệu

Phân loại là một trong những hoạt động cơ bản của nhà nghiên cứu học máy và nhà khoa học dữ liệu. Từ việc phân loại cơ bản một giá trị nhị phân ("email này có phải là spam hay không?"), đến phân loại hình ảnh phức tạp và phân đoạn bằng cách sử dụng thị giác máy tính, việc có thể phân loại dữ liệu thành các lớp và đặt câu hỏi về nó luôn hữu ích.

Nói theo cách khoa học hơn, phương pháp phân loại của bạn tạo ra một mô hình dự đoán cho phép bạn ánh xạ mối quan hệ giữa các biến đầu vào và biến đầu ra.

![phân loại nhị phân vs. đa lớp](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Các vấn đề nhị phân và đa lớp mà các thuật toán phân loại cần xử lý. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

Trước khi bắt đầu quá trình làm sạch dữ liệu, trực quan hóa nó và chuẩn bị cho các nhiệm vụ học máy của chúng ta, hãy tìm hiểu một chút về các cách khác nhau mà học máy có thể được sử dụng để phân loại dữ liệu.

Xuất phát từ [thống kê](https://wikipedia.org/wiki/Statistical_classification), phân loại sử dụng học máy cổ điển dựa vào các đặc điểm như `smoker`, `weight`, và `age` để xác định _khả năng phát triển bệnh X_. Là một kỹ thuật học có giám sát tương tự như các bài tập hồi quy bạn đã thực hiện trước đó, dữ liệu của bạn được gán nhãn và các thuật toán học máy sử dụng các nhãn đó để phân loại và dự đoán các lớp (hoặc 'đặc điểm') của một tập dữ liệu và gán chúng vào một nhóm hoặc kết quả.

✅ Hãy dành một chút thời gian để tưởng tượng một tập dữ liệu về các món ăn. Một mô hình phân loại đa lớp có thể trả lời những câu hỏi gì? Một mô hình phân loại nhị phân có thể trả lời những câu hỏi gì? Điều gì sẽ xảy ra nếu bạn muốn xác định liệu một món ăn cụ thể có khả năng sử dụng hạt cỏ cà ri hay không? Điều gì sẽ xảy ra nếu bạn muốn xem liệu, với một túi quà gồm hoa hồi, atisô, súp lơ và cải ngựa, bạn có thể tạo ra một món ăn Ấn Độ điển hình hay không?

[![Giỏ bí ẩn điên rồ](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Giỏ bí ẩn điên rồ")

> 🎥 Nhấp vào hình ảnh trên để xem video. Toàn bộ ý tưởng của chương trình 'Chopped' là 'giỏ bí ẩn', nơi các đầu bếp phải tạo ra một món ăn từ một lựa chọn ngẫu nhiên các nguyên liệu. Chắc chắn một mô hình học máy sẽ giúp ích!

## Xin chào 'bộ phân loại'

Câu hỏi chúng ta muốn đặt ra với tập dữ liệu món ăn này thực sự là một câu hỏi **đa lớp**, vì chúng ta có nhiều món ăn quốc gia tiềm năng để làm việc. Với một nhóm nguyên liệu, lớp nào trong số nhiều lớp này sẽ phù hợp với dữ liệu?

Scikit-learn cung cấp một số thuật toán khác nhau để phân loại dữ liệu, tùy thuộc vào loại vấn đề bạn muốn giải quyết. Trong hai bài học tiếp theo, bạn sẽ tìm hiểu về một số thuật toán này.

## Bài tập - làm sạch và cân bằng dữ liệu của bạn

Nhiệm vụ đầu tiên, trước khi bắt đầu dự án này, là làm sạch và **cân bằng** dữ liệu của bạn để có kết quả tốt hơn. Bắt đầu với tệp _notebook.ipynb_ trống trong thư mục gốc của thư mục này.

Điều đầu tiên cần cài đặt là [imblearn](https://imbalanced-learn.org/stable/). Đây là một gói Scikit-learn sẽ cho phép bạn cân bằng dữ liệu tốt hơn (bạn sẽ tìm hiểu thêm về nhiệm vụ này trong một phút).

1. Để cài đặt `imblearn`, chạy `pip install`, như sau:

    ```python
    pip install imblearn
    ```

1. Nhập các gói bạn cần để nhập dữ liệu và trực quan hóa nó, cũng như nhập `SMOTE` từ `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Bây giờ bạn đã sẵn sàng để nhập dữ liệu tiếp theo.

1. Nhiệm vụ tiếp theo sẽ là nhập dữ liệu:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Sử dụng `read_csv()` sẽ đọc nội dung của tệp csv _cusines.csv_ và đặt nó vào biến `df`.

1. Kiểm tra hình dạng của dữ liệu:

    ```python
    df.head()
    ```

   Năm hàng đầu tiên trông như thế này:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Lấy thông tin về dữ liệu này bằng cách gọi `info()`:

    ```python
    df.info()
    ```

    Kết quả của bạn giống như sau:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Bài tập - tìm hiểu về các món ăn

Bây giờ công việc bắt đầu trở nên thú vị hơn. Hãy khám phá sự phân bố dữ liệu theo từng món ăn.

1. Vẽ dữ liệu dưới dạng biểu đồ thanh ngang bằng cách gọi `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![phân bố dữ liệu món ăn](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Có một số lượng món ăn hữu hạn, nhưng sự phân bố dữ liệu không đồng đều. Bạn có thể sửa điều đó! Trước khi làm vậy, hãy khám phá thêm một chút.

1. Tìm hiểu có bao nhiêu dữ liệu có sẵn cho mỗi món ăn và in ra:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    Kết quả trông như sau:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Khám phá nguyên liệu

Bây giờ bạn có thể đi sâu hơn vào dữ liệu và tìm hiểu những nguyên liệu điển hình cho mỗi món ăn. Bạn nên loại bỏ dữ liệu lặp lại gây nhầm lẫn giữa các món ăn, vì vậy hãy tìm hiểu về vấn đề này.

1. Tạo một hàm `create_ingredient()` trong Python để tạo một dataframe nguyên liệu. Hàm này sẽ bắt đầu bằng cách loại bỏ một cột không hữu ích và sắp xếp các nguyên liệu theo số lượng:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Bây giờ bạn có thể sử dụng hàm đó để có ý tưởng về mười nguyên liệu phổ biến nhất theo từng món ăn.

1. Gọi `create_ingredient()` và vẽ biểu đồ bằng cách gọi `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Làm tương tự với dữ liệu món ăn Nhật Bản:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Bây giờ với các nguyên liệu món ăn Trung Quốc:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Vẽ biểu đồ các nguyên liệu món ăn Ấn Độ:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Cuối cùng, vẽ biểu đồ các nguyên liệu món ăn Hàn Quốc:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Bây giờ, loại bỏ các nguyên liệu phổ biến nhất gây nhầm lẫn giữa các món ăn khác nhau bằng cách gọi `drop()`:

   Ai cũng yêu thích cơm, tỏi và gừng!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Cân bằng tập dữ liệu

Bây giờ bạn đã làm sạch dữ liệu, hãy sử dụng [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Kỹ thuật Tăng Cường Mẫu Thiểu Số Tổng Hợp" - để cân bằng nó.

1. Gọi `fit_resample()`, chiến lược này tạo ra các mẫu mới bằng cách nội suy.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Bằng cách cân bằng dữ liệu của bạn, bạn sẽ có kết quả tốt hơn khi phân loại nó. Hãy nghĩ về một phân loại nhị phân. Nếu phần lớn dữ liệu của bạn thuộc một lớp, một mô hình học máy sẽ dự đoán lớp đó thường xuyên hơn, chỉ vì có nhiều dữ liệu hơn cho nó. Cân bằng dữ liệu giúp loại bỏ sự mất cân bằng này.

1. Bây giờ bạn có thể kiểm tra số lượng nhãn theo nguyên liệu:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Kết quả của bạn trông như sau:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Dữ liệu đã được làm sạch, cân bằng và rất hấp dẫn!

1. Bước cuối cùng là lưu dữ liệu đã cân bằng của bạn, bao gồm nhãn và đặc điểm, vào một dataframe mới có thể được xuất ra tệp:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Bạn có thể xem lại dữ liệu bằng cách sử dụng `transformed_df.head()` và `transformed_df.info()`. Lưu một bản sao của dữ liệu này để sử dụng trong các bài học sau:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Tệp CSV mới này hiện có thể được tìm thấy trong thư mục dữ liệu gốc.

---

## 🚀Thử thách

Chương trình học này chứa một số tập dữ liệu thú vị. Hãy tìm kiếm trong các thư mục `data` và xem liệu có tập dữ liệu nào phù hợp cho phân loại nhị phân hoặc đa lớp không? Bạn sẽ đặt câu hỏi gì với tập dữ liệu này?

## [Câu hỏi sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Khám phá API của SMOTE. Những trường hợp sử dụng nào là tốt nhất cho nó? Những vấn đề nào nó giải quyết?

## Bài tập 

[Khám phá các phương pháp phân loại](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.