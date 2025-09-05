<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T19:50:19+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "vi"
}
-->
# Bộ phân loại ẩm thực 1

Trong bài học này, bạn sẽ sử dụng tập dữ liệu mà bạn đã lưu từ bài học trước, chứa đầy dữ liệu cân bằng và sạch về các nền ẩm thực.

Bạn sẽ sử dụng tập dữ liệu này với nhiều bộ phân loại khác nhau để _dự đoán một nền ẩm thực quốc gia dựa trên nhóm nguyên liệu_. Trong quá trình thực hiện, bạn sẽ tìm hiểu thêm về một số cách mà các thuật toán có thể được sử dụng cho các nhiệm vụ phân loại.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)
# Chuẩn bị

Giả sử bạn đã hoàn thành [Bài học 1](../1-Introduction/README.md), hãy đảm bảo rằng tệp _cleaned_cuisines.csv_ tồn tại trong thư mục gốc `/data` cho bốn bài học này.

## Bài tập - dự đoán một nền ẩm thực quốc gia

1. Làm việc trong thư mục _notebook.ipynb_ của bài học này, nhập tệp đó cùng với thư viện Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dữ liệu trông như thế này:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Bây giờ, nhập thêm một số thư viện:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Chia tọa độ X và y thành hai dataframe để huấn luyện. `cuisine` có thể là dataframe nhãn:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Nó sẽ trông như thế này:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Loại bỏ cột `Unnamed: 0` và cột `cuisine` bằng cách gọi `drop()`. Lưu phần còn lại của dữ liệu làm các đặc trưng để huấn luyện:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Các đặc trưng của bạn trông như thế này:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Bây giờ bạn đã sẵn sàng để huấn luyện mô hình của mình!

## Chọn bộ phân loại

Bây giờ dữ liệu của bạn đã sạch và sẵn sàng để huấn luyện, bạn cần quyết định thuật toán nào sẽ sử dụng cho công việc.

Scikit-learn nhóm phân loại dưới Học có giám sát, và trong danh mục này, bạn sẽ tìm thấy nhiều cách để phân loại. [Sự đa dạng](https://scikit-learn.org/stable/supervised_learning.html) có thể khá choáng ngợp lúc ban đầu. Các phương pháp sau đây đều bao gồm các kỹ thuật phân loại:

- Mô hình tuyến tính
- Máy vector hỗ trợ
- Gradient ngẫu nhiên
- Láng giềng gần nhất
- Quá trình Gaussian
- Cây quyết định
- Phương pháp tổng hợp (bộ phân loại bỏ phiếu)
- Thuật toán đa lớp và đa đầu ra (phân loại đa lớp và đa nhãn, phân loại đa lớp-đa đầu ra)

> Bạn cũng có thể sử dụng [mạng nơ-ron để phân loại dữ liệu](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), nhưng điều này nằm ngoài phạm vi của bài học này.

### Nên chọn bộ phân loại nào?

Vậy, bạn nên chọn bộ phân loại nào? Thường thì việc thử qua nhiều bộ phân loại và tìm kiếm kết quả tốt là một cách để kiểm tra. Scikit-learn cung cấp một [so sánh song song](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) trên một tập dữ liệu được tạo, so sánh KNeighbors, SVC hai cách, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB và QuadraticDiscrinationAnalysis, hiển thị kết quả dưới dạng hình ảnh:

![so sánh các bộ phân loại](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Các biểu đồ được tạo trên tài liệu của Scikit-learn

> AutoML giải quyết vấn đề này một cách gọn gàng bằng cách chạy các so sánh này trên đám mây, cho phép bạn chọn thuật toán tốt nhất cho dữ liệu của mình. Thử tại [đây](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Một cách tiếp cận tốt hơn

Một cách tốt hơn thay vì đoán mò là làm theo các ý tưởng trong [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) có thể tải xuống. Tại đây, chúng ta phát hiện rằng, đối với vấn đề phân loại đa lớp của chúng ta, chúng ta có một số lựa chọn:

![cheatsheet cho các vấn đề đa lớp](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Một phần của Bảng Cheat Thuật toán của Microsoft, chi tiết các tùy chọn phân loại đa lớp

✅ Tải xuống bảng cheat này, in ra và treo lên tường của bạn!

### Lý luận

Hãy xem liệu chúng ta có thể lý luận qua các cách tiếp cận khác nhau dựa trên các ràng buộc mà chúng ta có:

- **Mạng nơ-ron quá nặng**. Với tập dữ liệu sạch nhưng tối thiểu của chúng ta, và thực tế là chúng ta đang chạy huấn luyện cục bộ qua notebook, mạng nơ-ron quá nặng cho nhiệm vụ này.
- **Không sử dụng bộ phân loại hai lớp**. Chúng ta không sử dụng bộ phân loại hai lớp, vì vậy loại bỏ phương pháp one-vs-all.
- **Cây quyết định hoặc hồi quy logistic có thể hoạt động**. Một cây quyết định có thể hoạt động, hoặc hồi quy logistic cho dữ liệu đa lớp.
- **Cây quyết định tăng cường đa lớp giải quyết vấn đề khác**. Cây quyết định tăng cường đa lớp phù hợp nhất cho các nhiệm vụ phi tham số, ví dụ như các nhiệm vụ được thiết kế để xây dựng xếp hạng, vì vậy nó không hữu ích cho chúng ta.

### Sử dụng Scikit-learn 

Chúng ta sẽ sử dụng Scikit-learn để phân tích dữ liệu của mình. Tuy nhiên, có nhiều cách để sử dụng hồi quy logistic trong Scikit-learn. Hãy xem các [tham số cần truyền](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Về cơ bản, có hai tham số quan trọng - `multi_class` và `solver` - mà chúng ta cần chỉ định khi yêu cầu Scikit-learn thực hiện hồi quy logistic. Giá trị `multi_class` áp dụng một hành vi nhất định. Giá trị của solver là thuật toán nào sẽ được sử dụng. Không phải tất cả các solver đều có thể kết hợp với tất cả các giá trị `multi_class`.

Theo tài liệu, trong trường hợp đa lớp, thuật toán huấn luyện:

- **Sử dụng phương pháp one-vs-rest (OvR)**, nếu tùy chọn `multi_class` được đặt là `ovr`
- **Sử dụng tổn thất cross-entropy**, nếu tùy chọn `multi_class` được đặt là `multinomial`. (Hiện tại tùy chọn `multinomial` chỉ được hỗ trợ bởi các solver ‘lbfgs’, ‘sag’, ‘saga’ và ‘newton-cg’.)"

> 🎓 'Phương pháp' ở đây có thể là 'ovr' (one-vs-rest) hoặc 'multinomial'. Vì hồi quy logistic thực sự được thiết kế để hỗ trợ phân loại nhị phân, các phương pháp này cho phép nó xử lý tốt hơn các nhiệm vụ phân loại đa lớp. [nguồn](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' được định nghĩa là "thuật toán được sử dụng trong bài toán tối ưu hóa". [nguồn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn cung cấp bảng này để giải thích cách các solver xử lý các thách thức khác nhau do các cấu trúc dữ liệu khác nhau gây ra:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Bài tập - chia dữ liệu

Chúng ta có thể tập trung vào hồi quy logistic cho lần thử huấn luyện đầu tiên của mình vì bạn đã học về nó trong bài học trước.
Chia dữ liệu của bạn thành nhóm huấn luyện và kiểm tra bằng cách gọi `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Bài tập - áp dụng hồi quy logistic

Vì bạn đang sử dụng trường hợp đa lớp, bạn cần chọn _phương pháp_ nào để sử dụng và _solver_ nào để đặt. Sử dụng LogisticRegression với cài đặt đa lớp và solver **liblinear** để huấn luyện.

1. Tạo một hồi quy logistic với multi_class được đặt là `ovr` và solver được đặt là `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Thử một solver khác như `lbfgs`, thường được đặt làm mặc định.
> Lưu ý, sử dụng hàm Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) để làm phẳng dữ liệu của bạn khi cần thiết.
Độ chính xác đạt trên **80%**!

1. Bạn có thể xem mô hình này hoạt động bằng cách thử nghiệm một hàng dữ liệu (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Kết quả được in ra:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Thử một số hàng khác và kiểm tra kết quả

1. Đào sâu hơn, bạn có thể kiểm tra độ chính xác của dự đoán này:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Kết quả được in ra - ẩm thực Ấn Độ là dự đoán tốt nhất, với xác suất cao:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Bạn có thể giải thích tại sao mô hình khá chắc chắn đây là ẩm thực Ấn Độ không?

1. Tìm hiểu chi tiết hơn bằng cách in báo cáo phân loại, như bạn đã làm trong bài học về hồi quy:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Thử thách

Trong bài học này, bạn đã sử dụng dữ liệu đã được làm sạch để xây dựng một mô hình học máy có thể dự đoán ẩm thực quốc gia dựa trên một loạt các nguyên liệu. Dành thời gian để đọc qua các tùy chọn mà Scikit-learn cung cấp để phân loại dữ liệu. Đào sâu hơn vào khái niệm 'solver' để hiểu những gì diễn ra phía sau.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Tìm hiểu thêm về toán học đằng sau hồi quy logistic trong [bài học này](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Bài tập 

[Khám phá các solver](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.