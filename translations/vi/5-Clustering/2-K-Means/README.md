<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T19:17:36+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "vi"
}
-->
# Phân cụm K-Means

## [Câu hỏi trước bài học](https://ff-quizzes.netlify.app/en/ml/)

Trong bài học này, bạn sẽ học cách tạo các cụm bằng cách sử dụng Scikit-learn và bộ dữ liệu âm nhạc Nigeria mà bạn đã nhập trước đó. Chúng ta sẽ tìm hiểu những điều cơ bản về K-Means để phân cụm. Hãy nhớ rằng, như bạn đã học trong bài trước, có nhiều cách để làm việc với các cụm và phương pháp bạn sử dụng phụ thuộc vào dữ liệu của bạn. Chúng ta sẽ thử K-Means vì đây là kỹ thuật phân cụm phổ biến nhất. Bắt đầu nào!

Các thuật ngữ bạn sẽ học:

- Điểm số Silhouette
- Phương pháp Elbow
- Quán tính (Inertia)
- Phương sai (Variance)

## Giới thiệu

[Phân cụm K-Means](https://wikipedia.org/wiki/K-means_clustering) là một phương pháp xuất phát từ lĩnh vực xử lý tín hiệu. Nó được sử dụng để chia và phân nhóm dữ liệu thành 'k' cụm bằng cách sử dụng một loạt các quan sát. Mỗi quan sát hoạt động để nhóm một điểm dữ liệu gần nhất với 'mean' của nó, hoặc điểm trung tâm của một cụm.

Các cụm có thể được hình dung dưới dạng [biểu đồ Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), bao gồm một điểm (hoặc 'hạt giống') và vùng tương ứng của nó.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

Quy trình phân cụm K-Means [thực hiện theo ba bước](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Thuật toán chọn số lượng điểm trung tâm k bằng cách lấy mẫu từ tập dữ liệu. Sau đó, nó lặp lại:
    1. Gán mỗi mẫu cho điểm trung tâm gần nhất.
    2. Tạo các điểm trung tâm mới bằng cách lấy giá trị trung bình của tất cả các mẫu được gán cho các điểm trung tâm trước đó.
    3. Sau đó, tính toán sự khác biệt giữa các điểm trung tâm mới và cũ và lặp lại cho đến khi các điểm trung tâm ổn định.

Một nhược điểm của việc sử dụng K-Means là bạn cần xác định 'k', tức là số lượng điểm trung tâm. May mắn thay, phương pháp 'elbow' giúp ước tính giá trị khởi đầu tốt cho 'k'. Bạn sẽ thử nó ngay bây giờ.

## Điều kiện tiên quyết

Bạn sẽ làm việc trong tệp [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) của bài học này, bao gồm việc nhập dữ liệu và làm sạch sơ bộ mà bạn đã thực hiện trong bài học trước.

## Bài tập - chuẩn bị

Bắt đầu bằng cách xem lại dữ liệu bài hát.

1. Tạo biểu đồ hộp, gọi `boxplot()` cho mỗi cột:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    Dữ liệu này hơi nhiễu: bằng cách quan sát mỗi cột dưới dạng biểu đồ hộp, bạn có thể thấy các giá trị ngoại lai.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Bạn có thể đi qua tập dữ liệu và loại bỏ các giá trị ngoại lai này, nhưng điều đó sẽ làm cho dữ liệu khá ít.

1. Hiện tại, hãy chọn các cột bạn sẽ sử dụng cho bài tập phân cụm. Chọn các cột có phạm vi tương tự và mã hóa cột `artist_top_genre` dưới dạng dữ liệu số:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Bây giờ bạn cần chọn số lượng cụm để nhắm mục tiêu. Bạn biết có 3 thể loại bài hát mà chúng ta đã phân loại từ tập dữ liệu, vì vậy hãy thử 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Bạn sẽ thấy một mảng được in ra với các cụm dự đoán (0, 1 hoặc 2) cho mỗi hàng của dataframe.

1. Sử dụng mảng này để tính toán 'điểm số silhouette':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Điểm số Silhouette

Tìm điểm số silhouette gần 1. Điểm số này dao động từ -1 đến 1, và nếu điểm số là 1, cụm sẽ dày đặc và tách biệt tốt với các cụm khác. Giá trị gần 0 đại diện cho các cụm chồng chéo với các mẫu rất gần ranh giới quyết định của các cụm lân cận. [(Nguồn)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Điểm số của chúng ta là **.53**, tức là ở mức trung bình. Điều này cho thấy dữ liệu của chúng ta không thực sự phù hợp với loại phân cụm này, nhưng hãy tiếp tục.

### Bài tập - xây dựng mô hình

1. Nhập `KMeans` và bắt đầu quá trình phân cụm.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Có một vài phần cần giải thích.

    > 🎓 range: Đây là số lần lặp của quá trình phân cụm.

    > 🎓 random_state: "Xác định việc tạo số ngẫu nhiên để khởi tạo điểm trung tâm." [Nguồn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "tổng bình phương trong cụm" đo khoảng cách trung bình bình phương của tất cả các điểm trong một cụm đến điểm trung tâm của cụm. [Nguồn](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inertia: Thuật toán K-Means cố gắng chọn các điểm trung tâm để giảm thiểu 'inertia', "một thước đo mức độ gắn kết nội bộ của các cụm." [Nguồn](https://scikit-learn.org/stable/modules/clustering.html). Giá trị này được thêm vào biến wcss trong mỗi lần lặp.

    > 🎓 k-means++: Trong [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means), bạn có thể sử dụng tối ưu hóa 'k-means++', giúp "khởi tạo các điểm trung tâm để (thường) cách xa nhau, dẫn đến kết quả có thể tốt hơn so với khởi tạo ngẫu nhiên."

### Phương pháp Elbow

Trước đó, bạn đã suy đoán rằng, vì bạn đã nhắm mục tiêu 3 thể loại bài hát, bạn nên chọn 3 cụm. Nhưng có đúng như vậy không?

1. Sử dụng phương pháp 'elbow' để đảm bảo.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Sử dụng biến `wcss` mà bạn đã xây dựng ở bước trước để tạo biểu đồ hiển thị nơi 'gấp khúc' trong elbow, điều này cho thấy số lượng cụm tối ưu. Có thể nó **đúng là** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Bài tập - hiển thị các cụm

1. Thử lại quy trình, lần này đặt ba cụm và hiển thị các cụm dưới dạng biểu đồ phân tán:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. Kiểm tra độ chính xác của mô hình:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Độ chính xác của mô hình này không tốt lắm, và hình dạng của các cụm cho bạn một gợi ý tại sao.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Dữ liệu này quá mất cân bằng, quá ít tương quan và có quá nhiều phương sai giữa các giá trị cột để phân cụm tốt. Thực tế, các cụm được hình thành có thể bị ảnh hưởng hoặc lệch nhiều bởi ba thể loại mà chúng ta đã xác định ở trên. Đó là một quá trình học tập!

    Trong tài liệu của Scikit-learn, bạn có thể thấy rằng một mô hình như thế này, với các cụm không được phân định rõ ràng, có vấn đề về 'phương sai':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Đồ họa thông tin từ Scikit-learn

## Phương sai

Phương sai được định nghĩa là "trung bình của các bình phương sai khác từ giá trị trung bình" [(Nguồn)](https://www.mathsisfun.com/data/standard-deviation.html). Trong bối cảnh của vấn đề phân cụm này, nó đề cập đến dữ liệu mà các số trong tập dữ liệu của chúng ta có xu hướng lệch quá nhiều so với giá trị trung bình.

✅ Đây là thời điểm tuyệt vời để suy nghĩ về tất cả các cách bạn có thể khắc phục vấn đề này. Tinh chỉnh dữ liệu thêm một chút? Sử dụng các cột khác? Sử dụng thuật toán khác? Gợi ý: Thử [chuẩn hóa dữ liệu của bạn](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) để làm cho nó đồng nhất và thử nghiệm các cột khác.

> Thử '[máy tính phương sai](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' để hiểu thêm về khái niệm này.

---

## 🚀Thử thách

Dành thời gian với notebook này, tinh chỉnh các tham số. Bạn có thể cải thiện độ chính xác của mô hình bằng cách làm sạch dữ liệu thêm (ví dụ như loại bỏ các giá trị ngoại lai)? Bạn có thể sử dụng trọng số để tăng trọng số cho các mẫu dữ liệu nhất định. Bạn còn có thể làm gì để tạo ra các cụm tốt hơn?

Gợi ý: Thử chuẩn hóa dữ liệu của bạn. Có mã được bình luận trong notebook thêm chuẩn hóa tiêu chuẩn để làm cho các cột dữ liệu giống nhau hơn về phạm vi. Bạn sẽ thấy rằng mặc dù điểm số silhouette giảm xuống, nhưng 'gấp khúc' trong biểu đồ elbow trở nên mượt mà hơn. Điều này là do để dữ liệu không được chuẩn hóa cho phép dữ liệu có ít phương sai hơn mang trọng số lớn hơn. Đọc thêm về vấn đề này [tại đây](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Câu hỏi sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Xem một trình mô phỏng K-Means [như thế này](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Bạn có thể sử dụng công cụ này để hình dung các điểm dữ liệu mẫu và xác định các điểm trung tâm của chúng. Bạn có thể chỉnh sửa độ ngẫu nhiên của dữ liệu, số lượng cụm và số lượng điểm trung tâm. Điều này có giúp bạn hiểu cách dữ liệu có thể được nhóm lại không?

Ngoài ra, hãy xem [tài liệu về K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) từ Stanford.

## Bài tập

[Thử các phương pháp phân cụm khác](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, chúng tôi khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.