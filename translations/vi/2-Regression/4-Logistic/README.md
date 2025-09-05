<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T18:46:57+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "vi"
}
-->
# Hồi quy Logistic để dự đoán danh mục

![Infographic về hồi quy Logistic và hồi quy tuyến tính](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Quiz trước bài học](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bài học này có sẵn bằng R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Giới thiệu

Trong bài học cuối cùng về Hồi quy, một trong những kỹ thuật _cổ điển_ cơ bản của ML, chúng ta sẽ tìm hiểu về Hồi quy Logistic. Bạn sẽ sử dụng kỹ thuật này để khám phá các mẫu nhằm dự đoán các danh mục nhị phân. Kẹo này có phải là sô-cô-la hay không? Bệnh này có lây hay không? Khách hàng này có chọn sản phẩm này hay không?

Trong bài học này, bạn sẽ học:

- Một thư viện mới để trực quan hóa dữ liệu
- Các kỹ thuật hồi quy logistic

✅ Nâng cao hiểu biết của bạn về cách làm việc với loại hồi quy này trong [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Điều kiện tiên quyết

Sau khi làm việc với dữ liệu về bí ngô, chúng ta đã đủ quen thuộc để nhận ra rằng có một danh mục nhị phân mà chúng ta có thể làm việc: `Color`.

Hãy xây dựng một mô hình hồi quy logistic để dự đoán rằng, dựa trên một số biến, _màu sắc của một quả bí ngô cụ thể có khả năng là gì_ (cam 🎃 hoặc trắng 👻).

> Tại sao chúng ta lại nói về phân loại nhị phân trong một bài học về hồi quy? Chỉ vì sự tiện lợi về ngôn ngữ, vì hồi quy logistic thực chất là [một phương pháp phân loại](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), mặc dù dựa trên tuyến tính. Tìm hiểu về các cách khác để phân loại dữ liệu trong nhóm bài học tiếp theo.

## Xác định câu hỏi

Đối với mục đích của chúng ta, chúng ta sẽ biểu thị điều này dưới dạng nhị phân: 'Trắng' hoặc 'Không Trắng'. Cũng có một danh mục 'có sọc' trong tập dữ liệu của chúng ta nhưng có rất ít trường hợp, vì vậy chúng ta sẽ không sử dụng nó. Nó sẽ biến mất khi chúng ta loại bỏ các giá trị null khỏi tập dữ liệu.

> 🎃 Thực tế thú vị, đôi khi chúng ta gọi bí ngô trắng là bí ngô 'ma'. Chúng không dễ khắc hình, vì vậy chúng không phổ biến như bí ngô cam nhưng trông rất thú vị! Vì vậy, chúng ta cũng có thể diễn đạt lại câu hỏi của mình là: 'Ma' hoặc 'Không Ma'. 👻

## Về hồi quy logistic

Hồi quy logistic khác với hồi quy tuyến tính, mà bạn đã học trước đó, ở một số điểm quan trọng.

[![ML cho người mới bắt đầu - Hiểu về hồi quy Logistic trong phân loại dữ liệu](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML cho người mới bắt đầu - Hiểu về hồi quy Logistic trong phân loại dữ liệu")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về hồi quy logistic.

### Phân loại nhị phân

Hồi quy logistic không cung cấp các tính năng giống như hồi quy tuyến tính. Phương pháp trước đưa ra dự đoán về một danh mục nhị phân ("trắng hoặc không trắng") trong khi phương pháp sau có khả năng dự đoán các giá trị liên tục, ví dụ như dựa trên nguồn gốc của bí ngô và thời gian thu hoạch, _giá của nó sẽ tăng bao nhiêu_.

![Mô hình phân loại bí ngô](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infographic bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Các phân loại khác

Có các loại hồi quy logistic khác, bao gồm đa danh mục và thứ tự:

- **Đa danh mục**, liên quan đến việc có nhiều hơn một danh mục - "Cam, Trắng, và Có Sọc".
- **Thứ tự**, liên quan đến các danh mục có thứ tự, hữu ích nếu chúng ta muốn sắp xếp kết quả một cách logic, như bí ngô của chúng ta được sắp xếp theo một số kích thước hữu hạn (mini, nhỏ, vừa, lớn, rất lớn, cực lớn).

![Hồi quy đa danh mục vs hồi quy thứ tự](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Các biến KHÔNG cần phải tương quan

Bạn còn nhớ hồi quy tuyến tính hoạt động tốt hơn với các biến tương quan không? Hồi quy logistic thì ngược lại - các biến không cần phải liên kết. Điều này phù hợp với dữ liệu này, vốn có các mối tương quan khá yếu.

### Bạn cần nhiều dữ liệu sạch

Hồi quy logistic sẽ cho kết quả chính xác hơn nếu bạn sử dụng nhiều dữ liệu; tập dữ liệu nhỏ của chúng ta không phải là tối ưu cho nhiệm vụ này, vì vậy hãy ghi nhớ điều đó.

[![ML cho người mới bắt đầu - Phân tích và chuẩn bị dữ liệu cho hồi quy Logistic](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML cho người mới bắt đầu - Phân tích và chuẩn bị dữ liệu cho hồi quy Logistic")

✅ Hãy suy nghĩ về các loại dữ liệu phù hợp với hồi quy logistic.

## Bài tập - làm sạch dữ liệu

Đầu tiên, làm sạch dữ liệu một chút, loại bỏ các giá trị null và chỉ chọn một số cột:

1. Thêm đoạn mã sau:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Bạn luôn có thể xem qua dataframe mới của mình:

    ```python
    pumpkins.info
    ```

### Trực quan hóa - biểu đồ danh mục

Đến giờ bạn đã tải lên [notebook khởi đầu](../../../../2-Regression/4-Logistic/notebook.ipynb) với dữ liệu bí ngô một lần nữa và làm sạch nó để giữ lại một tập dữ liệu chứa một vài biến, bao gồm `Color`. Hãy trực quan hóa dataframe trong notebook bằng một thư viện khác: [Seaborn](https://seaborn.pydata.org/index.html), được xây dựng trên Matplotlib mà chúng ta đã sử dụng trước đó.

Seaborn cung cấp một số cách thú vị để trực quan hóa dữ liệu của bạn. Ví dụ, bạn có thể so sánh phân phối dữ liệu cho mỗi `Variety` và `Color` trong một biểu đồ danh mục.

1. Tạo biểu đồ như vậy bằng cách sử dụng hàm `catplot`, sử dụng dữ liệu bí ngô `pumpkins`, và chỉ định ánh xạ màu cho mỗi danh mục bí ngô (cam hoặc trắng):

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![Lưới dữ liệu được trực quan hóa](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Bằng cách quan sát dữ liệu, bạn có thể thấy cách dữ liệu `Color` liên quan đến `Variety`.

    ✅ Dựa trên biểu đồ danh mục này, bạn có thể hình dung những khám phá thú vị nào?

### Tiền xử lý dữ liệu: mã hóa đặc trưng và nhãn
Tập dữ liệu bí ngô của chúng ta chứa các giá trị chuỗi cho tất cả các cột. Làm việc với dữ liệu danh mục rất trực quan đối với con người nhưng không phải đối với máy móc. Các thuật toán học máy hoạt động tốt với các con số. Đó là lý do tại sao mã hóa là một bước rất quan trọng trong giai đoạn tiền xử lý dữ liệu, vì nó cho phép chúng ta chuyển đổi dữ liệu danh mục thành dữ liệu số mà không mất thông tin. Mã hóa tốt dẫn đến việc xây dựng một mô hình tốt.

Đối với mã hóa đặc trưng, có hai loại mã hóa chính:

1. Mã hóa thứ tự: phù hợp với các biến thứ tự, là các biến danh mục mà dữ liệu của chúng tuân theo một thứ tự logic, như cột `Item Size` trong tập dữ liệu của chúng ta. Nó tạo ra một ánh xạ sao cho mỗi danh mục được biểu thị bằng một số, là thứ tự của danh mục trong cột.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Mã hóa danh mục: phù hợp với các biến danh mục, là các biến danh mục mà dữ liệu của chúng không tuân theo một thứ tự logic, như tất cả các đặc trưng khác ngoài `Item Size` trong tập dữ liệu của chúng ta. Đây là một mã hóa one-hot, nghĩa là mỗi danh mục được biểu thị bằng một cột nhị phân: biến được mã hóa bằng 1 nếu bí ngô thuộc về Variety đó và 0 nếu không.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Sau đó, `ColumnTransformer` được sử dụng để kết hợp nhiều bộ mã hóa thành một bước duy nhất và áp dụng chúng cho các cột thích hợp.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
Mặt khác, để mã hóa nhãn, chúng ta sử dụng lớp `LabelEncoder` của scikit-learn, là một lớp tiện ích để giúp chuẩn hóa nhãn sao cho chúng chỉ chứa các giá trị từ 0 đến n_classes-1 (ở đây là 0 và 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Khi chúng ta đã mã hóa các đặc trưng và nhãn, chúng ta có thể hợp nhất chúng thành một dataframe mới `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Những lợi ích của việc sử dụng mã hóa thứ tự cho cột `Item Size` là gì?

### Phân tích mối quan hệ giữa các biến

Bây giờ chúng ta đã tiền xử lý dữ liệu, chúng ta có thể phân tích mối quan hệ giữa các đặc trưng và nhãn để hiểu rõ hơn về khả năng dự đoán nhãn của mô hình dựa trên các đặc trưng.
Cách tốt nhất để thực hiện loại phân tích này là vẽ biểu đồ dữ liệu. Chúng ta sẽ sử dụng lại hàm `catplot` của Seaborn để trực quan hóa mối quan hệ giữa `Item Size`, `Variety` và `Color` trong một biểu đồ danh mục. Để vẽ biểu đồ dữ liệu tốt hơn, chúng ta sẽ sử dụng cột `Item Size` đã được mã hóa và cột `Variety` chưa được mã hóa.

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```
![Biểu đồ danh mục của dữ liệu được trực quan hóa](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Sử dụng biểu đồ swarm

Vì `Color` là một danh mục nhị phân (Trắng hoặc Không), nó cần 'một [cách tiếp cận chuyên biệt](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) để trực quan hóa'. Có những cách khác để trực quan hóa mối quan hệ của danh mục này với các biến khác.

Bạn có thể trực quan hóa các biến cạnh nhau bằng các biểu đồ của Seaborn.

1. Thử sử dụng biểu đồ 'swarm' để hiển thị phân phối các giá trị:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Một swarm của dữ liệu được trực quan hóa](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Lưu ý**: đoạn mã trên có thể tạo ra cảnh báo, vì Seaborn không thể biểu diễn số lượng điểm dữ liệu lớn như vậy trong biểu đồ swarm. Một giải pháp khả thi là giảm kích thước của điểm đánh dấu bằng cách sử dụng tham số 'size'. Tuy nhiên, hãy lưu ý rằng điều này ảnh hưởng đến khả năng đọc của biểu đồ.

> **🧮 Hiển thị Toán học**
>
> Hồi quy logistic dựa trên khái niệm 'xác suất tối đa' sử dụng [hàm sigmoid](https://wikipedia.org/wiki/Sigmoid_function). Một 'Hàm Sigmoid' trên biểu đồ trông giống như hình chữ 'S'. Nó lấy một giá trị và ánh xạ nó vào khoảng từ 0 đến 1. Đường cong của nó cũng được gọi là 'đường cong logistic'. Công thức của nó trông như thế này:
>
> ![hàm logistic](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> trong đó điểm giữa của sigmoid nằm ở điểm 0 của x, L là giá trị tối đa của đường cong, và k là độ dốc của đường cong. Nếu kết quả của hàm lớn hơn 0.5, nhãn được xét sẽ được gán vào lớp '1' của lựa chọn nhị phân. Nếu không, nó sẽ được phân loại là '0'.

## Xây dựng mô hình của bạn

Xây dựng một mô hình để tìm các phân loại nhị phân này khá đơn giản trong Scikit-learn.

[![ML cho người mới bắt đầu - Hồi quy Logistic để phân loại dữ liệu](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML cho người mới bắt đầu - Hồi quy Logistic để phân loại dữ liệu")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về xây dựng mô hình hồi quy tuyến tính.

1. Chọn các biến bạn muốn sử dụng trong mô hình phân loại của mình và chia tập huấn luyện và tập kiểm tra bằng cách gọi `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Bây giờ bạn có thể huấn luyện mô hình của mình bằng cách gọi `fit()` với dữ liệu huấn luyện và in kết quả của nó:

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    Xem qua bảng điểm của mô hình của bạn. Không tệ, xét rằng bạn chỉ có khoảng 1000 hàng dữ liệu:

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## Hiểu rõ hơn qua ma trận nhầm lẫn

Mặc dù bạn có thể nhận được báo cáo bảng điểm [thuật ngữ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) bằng cách in các mục trên, bạn có thể hiểu mô hình của mình dễ dàng hơn bằng cách sử dụng [ma trận nhầm lẫn](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) để giúp chúng ta hiểu cách mô hình đang hoạt động.

> 🎓 Một '[ma trận nhầm lẫn](https://wikipedia.org/wiki/Confusion_matrix)' (hoặc 'ma trận lỗi') là một bảng biểu thị các giá trị dương và âm thực sự so với sai của mô hình, từ đó đánh giá độ chính xác của dự đoán.

1. Để sử dụng ma trận nhầm lẫn, gọi `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Xem qua ma trận nhầm lẫn của mô hình của bạn:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Trong Scikit-learn, các hàng (trục 0) là nhãn thực tế và các cột (trục 1) là nhãn dự đoán.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Chuyện gì đang xảy ra ở đây? Giả sử mô hình của chúng ta được yêu cầu phân loại bí ngô giữa hai danh mục nhị phân, danh mục 'trắng' và danh mục 'không trắng'.

- Nếu mô hình của bạn dự đoán một quả bí ngô là không trắng và thực tế nó thuộc danh mục 'không trắng', chúng ta gọi đó là âm tính thực sự, được biểu thị bằng số ở góc trên bên trái.
- Nếu mô hình của bạn dự đoán một quả bí ngô là trắng và thực tế nó thuộc danh mục 'không trắng', chúng ta gọi đó là âm tính sai, được biểu thị bằng số ở góc dưới bên trái.
- Nếu mô hình của bạn dự đoán một quả bí ngô là không trắng và thực tế nó thuộc danh mục 'trắng', chúng ta gọi đó là dương tính sai, được biểu thị bằng số ở góc trên bên phải.
- Nếu mô hình của bạn dự đoán một quả bí ngô là trắng và thực tế nó thuộc danh mục 'trắng', chúng ta gọi đó là dương tính thực sự, được biểu thị bằng số ở góc dưới bên phải.

Như bạn có thể đoán, sẽ tốt hơn nếu có số lượng dương tính thực sự và âm tính thực sự lớn hơn, và số lượng dương tính sai và âm tính sai nhỏ hơn, điều này cho thấy mô hình hoạt động tốt hơn.
Làm thế nào ma trận nhầm lẫn liên quan đến độ chính xác và độ hồi tưởng? Hãy nhớ rằng báo cáo phân loại được in ở trên đã hiển thị độ chính xác (0.85) và độ hồi tưởng (0.67).

Độ chính xác = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Độ hồi tưởng = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Hỏi: Theo ma trận nhầm lẫn, mô hình hoạt động như thế nào? Trả lời: Không tệ; có một số lượng lớn các giá trị âm đúng nhưng cũng có một vài giá trị âm sai.

Hãy cùng xem lại các thuật ngữ mà chúng ta đã thấy trước đó với sự trợ giúp của việc ánh xạ TP/TN và FP/FN trong ma trận nhầm lẫn:

🎓 Độ chính xác: TP/(TP + FP) Phần trăm các trường hợp liên quan trong số các trường hợp được truy xuất (ví dụ: các nhãn được gán đúng)

🎓 Độ hồi tưởng: TP/(TP + FN) Phần trăm các trường hợp liên quan được truy xuất, bất kể có được gán đúng hay không

🎓 f1-score: (2 * độ chính xác * độ hồi tưởng)/(độ chính xác + độ hồi tưởng) Trung bình có trọng số của độ chính xác và độ hồi tưởng, với giá trị tốt nhất là 1 và tệ nhất là 0

🎓 Support: Số lần xuất hiện của mỗi nhãn được truy xuất

🎓 Độ chính xác: (TP + TN)/(TP + TN + FP + FN) Phần trăm các nhãn được dự đoán chính xác cho một mẫu.

🎓 Macro Avg: Tính toán trung bình không trọng số của các chỉ số cho mỗi nhãn, không tính đến sự mất cân bằng nhãn.

🎓 Weighted Avg: Tính toán trung bình có trọng số của các chỉ số cho mỗi nhãn, tính đến sự mất cân bằng nhãn bằng cách trọng số theo số lượng hỗ trợ (số trường hợp đúng cho mỗi nhãn).

✅ Bạn có thể nghĩ đến chỉ số nào cần theo dõi nếu bạn muốn mô hình của mình giảm số lượng giá trị âm sai?

## Trực quan hóa đường cong ROC của mô hình này

[![ML cho người mới bắt đầu - Phân tích hiệu suất hồi quy logistic với đường cong ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML cho người mới bắt đầu - Phân tích hiệu suất hồi quy logistic với đường cong ROC")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về đường cong ROC

Hãy thực hiện một hình ảnh hóa nữa để xem cái gọi là 'ROC' curve:

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Sử dụng Matplotlib, vẽ [Đặc tính Hoạt động Nhận diện](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) hoặc ROC của mô hình. Đường cong ROC thường được sử dụng để có cái nhìn về đầu ra của một bộ phân loại theo các giá trị đúng và sai. "Đường cong ROC thường có tỷ lệ đúng trên trục Y và tỷ lệ sai trên trục X." Do đó, độ dốc của đường cong và khoảng cách giữa đường trung điểm và đường cong rất quan trọng: bạn muốn một đường cong nhanh chóng đi lên và vượt qua đường. Trong trường hợp của chúng ta, có các giá trị sai ban đầu, sau đó đường cong đi lên và vượt qua đúng cách:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Cuối cùng, sử dụng API [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) của Scikit-learn để tính toán 'Diện tích Dưới Đường Cong' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Kết quả là `0.9749908725812341`. Vì AUC dao động từ 0 đến 1, bạn muốn một điểm số lớn, vì một mô hình dự đoán chính xác 100% sẽ có AUC là 1; trong trường hợp này, mô hình _khá tốt_.

Trong các bài học tương lai về phân loại, bạn sẽ học cách lặp lại để cải thiện điểm số của mô hình. Nhưng hiện tại, chúc mừng bạn! Bạn đã hoàn thành các bài học về hồi quy này!

---
## 🚀Thử thách

Có rất nhiều điều để khám phá về hồi quy logistic! Nhưng cách tốt nhất để học là thử nghiệm. Tìm một tập dữ liệu phù hợp với loại phân tích này và xây dựng một mô hình với nó. Bạn học được gì? mẹo: thử [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) để tìm các tập dữ liệu thú vị.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Đọc vài trang đầu của [bài viết này từ Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) về một số ứng dụng thực tế của hồi quy logistic. Hãy suy nghĩ về các nhiệm vụ phù hợp hơn với một loại hồi quy hoặc loại khác mà chúng ta đã học cho đến nay. Điều gì sẽ hoạt động tốt nhất?

## Bài tập

[Thử lại hồi quy này](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.