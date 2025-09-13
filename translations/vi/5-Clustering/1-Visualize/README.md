<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T19:12:58+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về phân cụm

Phân cụm là một loại [Học không giám sát](https://wikipedia.org/wiki/Unsupervised_learning) giả định rằng một tập dữ liệu không được gắn nhãn hoặc các đầu vào của nó không được liên kết với các đầu ra được định nghĩa trước. Nó sử dụng các thuật toán khác nhau để phân loại dữ liệu không gắn nhãn và cung cấp các nhóm dựa trên các mẫu mà nó nhận ra trong dữ liệu.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Nhấp vào hình ảnh trên để xem video. Trong khi bạn đang học máy với phân cụm, hãy thưởng thức một số bài hát Dance Hall của Nigeria - đây là một bài hát được đánh giá cao từ năm 2014 của PSquare.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

### Giới thiệu

[Phân cụm](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) rất hữu ích cho việc khám phá dữ liệu. Hãy xem liệu nó có thể giúp khám phá xu hướng và mẫu trong cách khán giả Nigeria tiêu thụ âm nhạc.

✅ Dành một phút để suy nghĩ về các ứng dụng của phân cụm. Trong đời sống thực, phân cụm xảy ra bất cứ khi nào bạn có một đống quần áo và cần phân loại quần áo của các thành viên trong gia đình 🧦👕👖🩲. Trong khoa học dữ liệu, phân cụm xảy ra khi cố gắng phân tích sở thích của người dùng hoặc xác định các đặc điểm của bất kỳ tập dữ liệu không gắn nhãn nào. Phân cụm, theo một cách nào đó, giúp làm sáng tỏ sự hỗn loạn, giống như ngăn kéo đựng tất.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Nhấp vào hình ảnh trên để xem video: John Guttag của MIT giới thiệu về phân cụm

Trong môi trường chuyên nghiệp, phân cụm có thể được sử dụng để xác định các phân khúc thị trường, chẳng hạn như xác định nhóm tuổi nào mua những mặt hàng nào. Một ứng dụng khác có thể là phát hiện bất thường, chẳng hạn để phát hiện gian lận từ một tập dữ liệu giao dịch thẻ tín dụng. Hoặc bạn có thể sử dụng phân cụm để xác định khối u trong một loạt các bản quét y tế.

✅ Dành một phút để suy nghĩ về cách bạn có thể đã gặp phân cụm 'trong thực tế', trong ngân hàng, thương mại điện tử hoặc môi trường kinh doanh.

> 🎓 Thú vị là, phân tích cụm bắt nguồn từ các lĩnh vực Nhân học và Tâm lý học vào những năm 1930. Bạn có thể tưởng tượng nó đã được sử dụng như thế nào không?

Ngoài ra, bạn có thể sử dụng nó để nhóm các kết quả tìm kiếm - chẳng hạn như liên kết mua sắm, hình ảnh hoặc đánh giá. Phân cụm rất hữu ích khi bạn có một tập dữ liệu lớn mà bạn muốn giảm bớt và thực hiện phân tích chi tiết hơn, vì vậy kỹ thuật này có thể được sử dụng để tìm hiểu về dữ liệu trước khi xây dựng các mô hình khác.

✅ Khi dữ liệu của bạn được tổ chức thành các cụm, bạn gán cho nó một Id cụm, và kỹ thuật này có thể hữu ích khi bảo vệ quyền riêng tư của tập dữ liệu; bạn có thể thay thế việc tham chiếu một điểm dữ liệu bằng Id cụm của nó, thay vì bằng dữ liệu nhận dạng tiết lộ hơn. Bạn có thể nghĩ ra những lý do khác tại sao bạn lại tham chiếu một Id cụm thay vì các yếu tố khác của cụm để xác định nó không?

Tìm hiểu sâu hơn về các kỹ thuật phân cụm trong [Learn module này](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Bắt đầu với phân cụm

[Scikit-learn cung cấp một loạt lớn](https://scikit-learn.org/stable/modules/clustering.html) các phương pháp để thực hiện phân cụm. Loại bạn chọn sẽ phụ thuộc vào trường hợp sử dụng của bạn. Theo tài liệu, mỗi phương pháp có các lợi ích khác nhau. Dưới đây là bảng đơn giản hóa các phương pháp được hỗ trợ bởi Scikit-learn và các trường hợp sử dụng phù hợp:

| Tên phương pháp              | Trường hợp sử dụng                                                   |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | mục đích chung, suy diễn                                             |
| Affinity propagation         | nhiều cụm không đều, suy diễn                                        |
| Mean-shift                   | nhiều cụm không đều, suy diễn                                        |
| Spectral clustering          | ít cụm đều, suy diễn ngược                                          |
| Ward hierarchical clustering | nhiều cụm bị ràng buộc, suy diễn ngược                              |
| Agglomerative clustering     | nhiều cụm bị ràng buộc, khoảng cách không Euclidean, suy diễn ngược |
| DBSCAN                       | hình học không phẳng, cụm không đều, suy diễn ngược                 |
| OPTICS                       | hình học không phẳng, cụm không đều với mật độ biến đổi, suy diễn ngược |
| Gaussian mixtures            | hình học phẳng, suy diễn                                            |
| BIRCH                        | tập dữ liệu lớn với các điểm ngoại lai, suy diễn                    |

> 🎓 Cách chúng ta tạo cụm có liên quan nhiều đến cách chúng ta tập hợp các điểm dữ liệu thành nhóm. Hãy cùng tìm hiểu một số thuật ngữ:
>
> 🎓 ['Suy diễn ngược' vs. 'suy diễn'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Suy diễn ngược được rút ra từ các trường hợp huấn luyện quan sát được ánh xạ tới các trường hợp kiểm tra cụ thể. Suy diễn được rút ra từ các trường hợp huấn luyện ánh xạ tới các quy tắc chung, sau đó mới được áp dụng cho các trường hợp kiểm tra.
> 
> Một ví dụ: Hãy tưởng tượng bạn có một tập dữ liệu chỉ được gắn nhãn một phần. Một số thứ là 'đĩa nhạc', một số là 'cd', và một số là trống. Nhiệm vụ của bạn là cung cấp nhãn cho các mục trống. Nếu bạn chọn cách tiếp cận suy diễn, bạn sẽ huấn luyện một mô hình tìm kiếm 'đĩa nhạc' và 'cd', và áp dụng các nhãn đó cho dữ liệu chưa được gắn nhãn. Cách tiếp cận này sẽ gặp khó khăn trong việc phân loại những thứ thực sự là 'băng cassette'. Một cách tiếp cận suy diễn ngược, mặt khác, xử lý dữ liệu chưa biết này hiệu quả hơn vì nó hoạt động để nhóm các mục tương tự lại với nhau và sau đó áp dụng nhãn cho một nhóm. Trong trường hợp này, các cụm có thể phản ánh 'những thứ âm nhạc hình tròn' và 'những thứ âm nhạc hình vuông'.
> 
> 🎓 ['Hình học không phẳng' vs. 'hình học phẳng'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Được lấy từ thuật ngữ toán học, hình học không phẳng vs. phẳng đề cập đến việc đo khoảng cách giữa các điểm bằng các phương pháp hình học 'phẳng' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) hoặc 'không phẳng' (không Euclidean).
>
>'Phẳng' trong ngữ cảnh này đề cập đến hình học Euclidean (một phần của nó được dạy như hình học 'mặt phẳng'), và không phẳng đề cập đến hình học không Euclidean. Hình học liên quan gì đến học máy? Vâng, vì hai lĩnh vực này đều dựa trên toán học, cần phải có một cách chung để đo khoảng cách giữa các điểm trong các cụm, và điều đó có thể được thực hiện theo cách 'phẳng' hoặc 'không phẳng', tùy thuộc vào bản chất của dữ liệu. [Khoảng cách Euclidean](https://wikipedia.org/wiki/Euclidean_distance) được đo bằng chiều dài của một đoạn thẳng giữa hai điểm. [Khoảng cách không Euclidean](https://wikipedia.org/wiki/Non-Euclidean_geometry) được đo dọc theo một đường cong. Nếu dữ liệu của bạn, khi được hình dung, dường như không tồn tại trên một mặt phẳng, bạn có thể cần sử dụng một thuật toán chuyên biệt để xử lý nó.
>
![Flat vs Nonflat Geometry Infographic](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infographic bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Khoảng cách'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Các cụm được định nghĩa bởi ma trận khoảng cách của chúng, ví dụ: khoảng cách giữa các điểm. Khoảng cách này có thể được đo bằng một vài cách. Các cụm Euclidean được định nghĩa bởi giá trị trung bình của các điểm, và chứa một 'trọng tâm' hoặc điểm trung tâm. Khoảng cách do đó được đo bằng khoảng cách tới trọng tâm đó. Khoảng cách không Euclidean đề cập đến 'clustroids', điểm gần nhất với các điểm khác. Clustroids lần lượt có thể được định nghĩa theo nhiều cách.
> 
> 🎓 ['Bị ràng buộc'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Phân cụm bị ràng buộc](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) giới thiệu 'học bán giám sát' vào phương pháp không giám sát này. Các mối quan hệ giữa các điểm được đánh dấu là 'không thể liên kết' hoặc 'phải liên kết' để một số quy tắc được áp dụng cho tập dữ liệu.
>
> Một ví dụ: Nếu một thuật toán được tự do trên một loạt dữ liệu không gắn nhãn hoặc bán gắn nhãn, các cụm mà nó tạo ra có thể có chất lượng kém. Trong ví dụ trên, các cụm có thể nhóm 'những thứ âm nhạc hình tròn' và 'những thứ âm nhạc hình vuông' và 'những thứ hình tam giác' và 'bánh quy'. Nếu được cung cấp một số ràng buộc, hoặc quy tắc để tuân theo ("mục phải được làm bằng nhựa", "mục cần có khả năng tạo ra âm nhạc") điều này có thể giúp 'ràng buộc' thuật toán để đưa ra các lựa chọn tốt hơn.
> 
> 🎓 'Mật độ'
> 
> Dữ liệu 'nhiễu' được coi là 'dày đặc'. Khoảng cách giữa các điểm trong mỗi cụm của nó có thể chứng minh, khi kiểm tra, là dày đặc hơn hoặc ít dày đặc hơn, hoặc 'đông đúc' và do đó dữ liệu này cần được phân tích bằng phương pháp phân cụm phù hợp. [Bài viết này](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) minh họa sự khác biệt giữa việc sử dụng phân cụm K-Means vs. các thuật toán HDBSCAN để khám phá một tập dữ liệu nhiễu với mật độ cụm không đều.

## Các thuật toán phân cụm

Có hơn 100 thuật toán phân cụm, và việc sử dụng chúng phụ thuộc vào bản chất của dữ liệu hiện có. Hãy thảo luận một số thuật toán chính:

- **Phân cụm phân cấp**. Nếu một đối tượng được phân loại dựa trên sự gần gũi của nó với một đối tượng gần đó, thay vì với một đối tượng xa hơn, các cụm được hình thành dựa trên khoảng cách của các thành viên của chúng tới và từ các đối tượng khác. Phân cụm kết hợp của Scikit-learn là phân cấp.

   ![Hierarchical clustering Infographic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infographic bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Phân cụm trọng tâm**. Thuật toán phổ biến này yêu cầu lựa chọn 'k', hoặc số lượng cụm cần hình thành, sau đó thuật toán xác định điểm trung tâm của một cụm và tập hợp dữ liệu xung quanh điểm đó. [Phân cụm K-means](https://wikipedia.org/wiki/K-means_clustering) là một phiên bản phổ biến của phân cụm trọng tâm. Trung tâm được xác định bởi giá trị trung bình gần nhất, do đó có tên gọi. Khoảng cách bình phương từ cụm được giảm thiểu.

   ![Centroid clustering Infographic](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infographic bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Phân cụm dựa trên phân phối**. Dựa trên mô hình thống kê, phân cụm dựa trên phân phối tập trung vào việc xác định xác suất rằng một điểm dữ liệu thuộc về một cụm, và gán nó tương ứng. Các phương pháp hỗn hợp Gaussian thuộc loại này.

- **Phân cụm dựa trên mật độ**. Các điểm dữ liệu được gán vào các cụm dựa trên mật độ của chúng, hoặc sự tập hợp xung quanh nhau. Các điểm dữ liệu xa nhóm được coi là điểm ngoại lai hoặc nhiễu. DBSCAN, Mean-shift và OPTICS thuộc loại phân cụm này.

- **Phân cụm dựa trên lưới**. Đối với các tập dữ liệu đa chiều, một lưới được tạo ra và dữ liệu được chia giữa các ô của lưới, từ đó tạo ra các cụm.

## Bài tập - phân cụm dữ liệu của bạn

Phân cụm như một kỹ thuật được hỗ trợ rất nhiều bởi việc trực quan hóa đúng cách, vì vậy hãy bắt đầu bằng cách trực quan hóa dữ liệu âm nhạc của chúng ta. Bài tập này sẽ giúp chúng ta quyết định phương pháp phân cụm nào nên được sử dụng hiệu quả nhất cho bản chất của dữ liệu này.

1. Mở tệp [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) trong thư mục này.

1. Nhập gói `Seaborn` để trực quan hóa dữ liệu tốt.

    ```python
    !pip install seaborn
    ```

1. Thêm dữ liệu bài hát từ [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Tải lên một dataframe với một số dữ liệu về các bài hát. Chuẩn bị khám phá dữ liệu này bằng cách nhập các thư viện và xuất dữ liệu:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Kiểm tra vài dòng đầu tiên của dữ liệu:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Lấy một số thông tin về dataframe bằng cách gọi `info()`:

    ```python
    df.info()
    ```

   Kết quả sẽ trông như sau:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Kiểm tra lại giá trị null bằng cách gọi `isnull()` và xác nhận tổng số bằng 0:

    ```python
    df.isnull().sum()
    ```

    Trông ổn:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Mô tả dữ liệu:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Nếu chúng ta đang làm việc với clustering, một phương pháp không giám sát không yêu cầu dữ liệu được gắn nhãn, tại sao lại hiển thị dữ liệu này với nhãn? Trong giai đoạn khám phá dữ liệu, chúng rất hữu ích, nhưng không cần thiết để các thuật toán clustering hoạt động. Bạn cũng có thể loại bỏ tiêu đề cột và tham chiếu dữ liệu bằng số cột.

Hãy xem các giá trị tổng quát của dữ liệu. Lưu ý rằng độ phổ biến có thể là '0', điều này cho thấy các bài hát không có xếp hạng. Hãy loại bỏ chúng ngay sau đây.

1. Sử dụng biểu đồ cột để tìm ra các thể loại phổ biến nhất:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Nếu bạn muốn xem thêm các giá trị hàng đầu, hãy thay đổi top `[:5]` thành một giá trị lớn hơn hoặc loại bỏ nó để xem tất cả.

Lưu ý, khi thể loại hàng đầu được mô tả là 'Missing', điều đó có nghĩa là Spotify không phân loại nó, vì vậy hãy loại bỏ nó.

1. Loại bỏ dữ liệu thiếu bằng cách lọc nó ra

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Bây giờ kiểm tra lại các thể loại:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Ba thể loại hàng đầu chiếm ưu thế trong tập dữ liệu này. Hãy tập trung vào `afro dancehall`, `afropop`, và `nigerian pop`, đồng thời lọc tập dữ liệu để loại bỏ bất kỳ giá trị độ phổ biến nào bằng 0 (nghĩa là nó không được phân loại với độ phổ biến trong tập dữ liệu và có thể được coi là nhiễu đối với mục đích của chúng ta):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Thực hiện một thử nghiệm nhanh để xem liệu dữ liệu có tương quan theo cách đặc biệt mạnh mẽ nào không:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Mối tương quan mạnh duy nhất là giữa `energy` và `loudness`, điều này không quá ngạc nhiên, vì âm nhạc lớn thường khá sôi động. Ngoài ra, các mối tương quan tương đối yếu. Sẽ rất thú vị để xem một thuật toán clustering có thể làm gì với dữ liệu này.

    > 🎓 Lưu ý rằng tương quan không ngụ ý nguyên nhân! Chúng ta có bằng chứng về tương quan nhưng không có bằng chứng về nguyên nhân. Một [trang web thú vị](https://tylervigen.com/spurious-correlations) có một số hình ảnh minh họa nhấn mạnh điểm này.

Liệu có sự hội tụ trong tập dữ liệu này xung quanh độ phổ biến và khả năng nhảy của một bài hát? Một FacetGrid cho thấy có các vòng tròn đồng tâm xếp hàng, bất kể thể loại. Có thể sở thích của người Nigeria hội tụ ở một mức độ nhảy nhất định cho thể loại này?

✅ Thử các điểm dữ liệu khác (energy, loudness, speechiness) và nhiều thể loại âm nhạc khác hoặc khác nhau. Bạn có thể khám phá được gì? Hãy xem bảng `df.describe()` để thấy sự phân bố tổng quát của các điểm dữ liệu.

### Bài tập - phân bố dữ liệu

Liệu ba thể loại này có khác biệt đáng kể trong cách nhìn nhận về khả năng nhảy của chúng, dựa trên độ phổ biến?

1. Kiểm tra phân bố dữ liệu của ba thể loại hàng đầu về độ phổ biến và khả năng nhảy dọc theo trục x và y nhất định.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Bạn có thể khám phá các vòng tròn đồng tâm xung quanh một điểm hội tụ tổng quát, cho thấy sự phân bố của các điểm.

    > 🎓 Lưu ý rằng ví dụ này sử dụng biểu đồ KDE (Kernel Density Estimate) để biểu diễn dữ liệu bằng một đường cong mật độ xác suất liên tục. Điều này cho phép chúng ta diễn giải dữ liệu khi làm việc với nhiều phân bố.

    Nhìn chung, ba thể loại này liên kết lỏng lẻo về độ phổ biến và khả năng nhảy. Xác định các cụm trong dữ liệu liên kết lỏng lẻo này sẽ là một thách thức:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Tạo biểu đồ scatter:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Biểu đồ scatter của cùng các trục cho thấy một mô hình hội tụ tương tự

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Nhìn chung, đối với clustering, bạn có thể sử dụng biểu đồ scatter để hiển thị các cụm dữ liệu, vì vậy việc thành thạo loại hình trực quan hóa này rất hữu ích. Trong bài học tiếp theo, chúng ta sẽ lấy dữ liệu đã lọc này và sử dụng k-means clustering để khám phá các nhóm trong dữ liệu này có xu hướng chồng lấn theo những cách thú vị.

---

## 🚀Thử thách

Để chuẩn bị cho bài học tiếp theo, hãy tạo một biểu đồ về các thuật toán clustering khác nhau mà bạn có thể khám phá và sử dụng trong môi trường sản xuất. Các vấn đề mà clustering đang cố gắng giải quyết là gì?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Trước khi áp dụng các thuật toán clustering, như chúng ta đã học, việc hiểu bản chất của tập dữ liệu là một ý tưởng tốt. Đọc thêm về chủ đề này [tại đây](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Bài viết hữu ích này](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) hướng dẫn bạn cách các thuật toán clustering khác nhau hoạt động, dựa trên các hình dạng dữ liệu khác nhau.

## Bài tập

[Nghiên cứu các hình thức trực quan hóa khác cho clustering](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.