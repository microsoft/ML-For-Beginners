<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T19:56:29+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "vi"
}
-->
# Bộ phân loại ẩm thực 2

Trong bài học phân loại thứ hai này, bạn sẽ khám phá thêm các cách để phân loại dữ liệu số. Bạn cũng sẽ tìm hiểu về hậu quả của việc chọn một bộ phân loại này thay vì bộ phân loại khác.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

### Điều kiện tiên quyết

Chúng tôi giả định rằng bạn đã hoàn thành các bài học trước và có một tập dữ liệu đã được làm sạch trong thư mục `data` của bạn, được gọi là _cleaned_cuisines.csv_ trong thư mục gốc của bài học gồm 4 phần này.

### Chuẩn bị

Chúng tôi đã tải tệp _notebook.ipynb_ của bạn với tập dữ liệu đã được làm sạch và đã chia nó thành các dataframe X và y, sẵn sàng cho quá trình xây dựng mô hình.

## Bản đồ phân loại

Trước đây, bạn đã tìm hiểu về các tùy chọn khác nhau khi phân loại dữ liệu bằng bảng cheat sheet của Microsoft. Scikit-learn cung cấp một bảng cheat sheet tương tự nhưng chi tiết hơn, giúp bạn thu hẹp các bộ ước lượng (một thuật ngữ khác cho bộ phân loại):

![Bản đồ ML từ Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Mẹo: [truy cập bản đồ này trực tuyến](https://scikit-learn.org/stable/tutorial/machine_learning_map/) và nhấp vào các đường dẫn để đọc tài liệu.

### Kế hoạch

Bản đồ này rất hữu ích khi bạn đã hiểu rõ về dữ liệu của mình, vì bạn có thể 'đi bộ' dọc theo các đường dẫn để đưa ra quyết định:

- Chúng ta có >50 mẫu
- Chúng ta muốn dự đoán một danh mục
- Chúng ta có dữ liệu được gắn nhãn
- Chúng ta có ít hơn 100K mẫu
- ✨ Chúng ta có thể chọn Linear SVC
- Nếu điều đó không hiệu quả, vì chúng ta có dữ liệu số
    - Chúng ta có thể thử ✨ KNeighbors Classifier 
      - Nếu điều đó không hiệu quả, thử ✨ SVC và ✨ Ensemble Classifiers

Đây là một lộ trình rất hữu ích để làm theo.

## Bài tập - chia dữ liệu

Theo lộ trình này, chúng ta nên bắt đầu bằng cách nhập một số thư viện cần thiết.

1. Nhập các thư viện cần thiết:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Chia dữ liệu huấn luyện và kiểm tra:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Bộ phân loại Linear SVC

Support-Vector clustering (SVC) là một nhánh của gia đình các kỹ thuật máy học Support-Vector machines (tìm hiểu thêm về chúng bên dưới). Trong phương pháp này, bạn có thể chọn một 'kernel' để quyết định cách phân cụm các nhãn. Tham số 'C' đề cập đến 'regularization', điều chỉnh ảnh hưởng của các tham số. Kernel có thể là một trong [nhiều loại](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); ở đây chúng ta đặt nó là 'linear' để đảm bảo rằng chúng ta sử dụng Linear SVC. Xác suất mặc định là 'false'; ở đây chúng ta đặt nó là 'true' để thu thập các ước tính xác suất. Chúng ta đặt random state là '0' để xáo trộn dữ liệu nhằm thu được xác suất.

### Bài tập - áp dụng Linear SVC

Bắt đầu bằng cách tạo một mảng các bộ phân loại. Bạn sẽ thêm dần vào mảng này khi chúng ta thử nghiệm.

1. Bắt đầu với Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Huấn luyện mô hình của bạn bằng Linear SVC và in ra báo cáo:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Kết quả khá tốt:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Bộ phân loại K-Neighbors

K-Neighbors là một phần của gia đình các phương pháp máy học "neighbors", có thể được sử dụng cho cả học có giám sát và không giám sát. Trong phương pháp này, một số điểm được xác định trước và dữ liệu được thu thập xung quanh các điểm này để dự đoán các nhãn tổng quát cho dữ liệu.

### Bài tập - áp dụng bộ phân loại K-Neighbors

Bộ phân loại trước đó khá tốt và hoạt động tốt với dữ liệu, nhưng có thể chúng ta có thể đạt được độ chính xác tốt hơn. Thử bộ phân loại K-Neighbors.

1. Thêm một dòng vào mảng bộ phân loại của bạn (thêm dấu phẩy sau mục Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Kết quả hơi kém hơn:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Tìm hiểu về [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Bộ phân loại Support Vector

Bộ phân loại Support-Vector là một phần của gia đình [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) các phương pháp máy học được sử dụng cho các nhiệm vụ phân loại và hồi quy. SVMs "ánh xạ các ví dụ huấn luyện thành các điểm trong không gian" để tối đa hóa khoảng cách giữa hai danh mục. Dữ liệu tiếp theo được ánh xạ vào không gian này để dự đoán danh mục của chúng.

### Bài tập - áp dụng bộ phân loại Support Vector

Hãy thử đạt độ chính xác tốt hơn một chút với bộ phân loại Support Vector.

1. Thêm dấu phẩy sau mục K-Neighbors, sau đó thêm dòng này:

    ```python
    'SVC': SVC(),
    ```

    Kết quả khá tốt!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Tìm hiểu về [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Bộ phân loại Ensemble

Hãy đi theo lộ trình đến cuối cùng, mặc dù thử nghiệm trước đó khá tốt. Hãy thử một số bộ phân loại 'Ensemble', cụ thể là Random Forest và AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Kết quả rất tốt, đặc biệt là với Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Tìm hiểu về [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Phương pháp máy học này "kết hợp các dự đoán của một số bộ ước lượng cơ bản" để cải thiện chất lượng mô hình. Trong ví dụ của chúng ta, chúng ta đã sử dụng Random Trees và AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), một phương pháp trung bình, xây dựng một 'rừng' các 'cây quyết định' được thêm ngẫu nhiên để tránh overfitting. Tham số n_estimators được đặt là số lượng cây.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) khớp một bộ phân loại với tập dữ liệu và sau đó khớp các bản sao của bộ phân loại đó với cùng tập dữ liệu. Nó tập trung vào trọng số của các mục được phân loại sai và điều chỉnh khớp cho bộ phân loại tiếp theo để sửa lỗi.

---

## 🚀Thử thách

Mỗi kỹ thuật này có một số lượng lớn các tham số mà bạn có thể điều chỉnh. Nghiên cứu các tham số mặc định của từng kỹ thuật và suy nghĩ về ý nghĩa của việc điều chỉnh các tham số này đối với chất lượng mô hình.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Có rất nhiều thuật ngữ chuyên ngành trong các bài học này, vì vậy hãy dành một chút thời gian để xem lại [danh sách này](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) các thuật ngữ hữu ích!

## Bài tập 

[Chơi với tham số](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.