<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T18:40:03+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "vi"
}
-->
# Xây dựng mô hình hồi quy sử dụng Scikit-learn: hồi quy theo bốn cách

![Đồ họa thông tin hồi quy tuyến tính và đa thức](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Đồ họa thông tin bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bài học này có sẵn bằng R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Giới thiệu 

Cho đến nay, bạn đã tìm hiểu hồi quy là gì với dữ liệu mẫu thu thập từ tập dữ liệu giá bí ngô mà chúng ta sẽ sử dụng xuyên suốt bài học này. Bạn cũng đã trực quan hóa nó bằng Matplotlib.

Bây giờ bạn đã sẵn sàng đi sâu hơn vào hồi quy cho ML. Trong khi trực quan hóa giúp bạn hiểu dữ liệu, sức mạnh thực sự của Machine Learning đến từ việc _huấn luyện mô hình_. Các mô hình được huấn luyện trên dữ liệu lịch sử để tự động nắm bắt các mối quan hệ dữ liệu, và chúng cho phép bạn dự đoán kết quả cho dữ liệu mới mà mô hình chưa từng thấy trước đó.

Trong bài học này, bạn sẽ tìm hiểu thêm về hai loại hồi quy: _hồi quy tuyến tính cơ bản_ và _hồi quy đa thức_, cùng với một số toán học cơ bản của các kỹ thuật này. Những mô hình này sẽ cho phép chúng ta dự đoán giá bí ngô dựa trên các dữ liệu đầu vào khác nhau.

[![ML cho người mới bắt đầu - Hiểu về hồi quy tuyến tính](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML cho người mới bắt đầu - Hiểu về hồi quy tuyến tính")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về hồi quy tuyến tính.

> Trong suốt chương trình học này, chúng tôi giả định kiến thức toán học tối thiểu và cố gắng làm cho nó dễ tiếp cận đối với học sinh đến từ các lĩnh vực khác, vì vậy hãy chú ý đến các ghi chú, 🧮 các điểm nhấn, sơ đồ và các công cụ học tập khác để hỗ trợ việc hiểu bài.

### Điều kiện tiên quyết

Đến giờ bạn đã quen thuộc với cấu trúc của dữ liệu bí ngô mà chúng ta đang xem xét. Bạn có thể tìm thấy nó được tải sẵn và làm sạch trước trong tệp _notebook.ipynb_ của bài học này. Trong tệp, giá bí ngô được hiển thị theo giạ trong một khung dữ liệu mới. Hãy đảm bảo rằng bạn có thể chạy các notebook này trong các kernel của Visual Studio Code.

### Chuẩn bị

Như một lời nhắc nhở, bạn đang tải dữ liệu này để đặt câu hỏi về nó.

- Khi nào là thời điểm tốt nhất để mua bí ngô?
- Giá của một thùng bí ngô nhỏ sẽ là bao nhiêu?
- Tôi nên mua chúng trong giạ nửa hay trong hộp 1 1/9 giạ?
Hãy tiếp tục khám phá dữ liệu này.

Trong bài học trước, bạn đã tạo một khung dữ liệu Pandas và điền vào nó một phần của tập dữ liệu gốc, chuẩn hóa giá theo giạ. Tuy nhiên, bằng cách làm như vậy, bạn chỉ có thể thu thập khoảng 400 điểm dữ liệu và chỉ cho các tháng mùa thu.

Hãy xem dữ liệu mà chúng tôi đã tải sẵn trong notebook đi kèm bài học này. Dữ liệu đã được tải sẵn và một biểu đồ phân tán ban đầu đã được vẽ để hiển thị dữ liệu theo tháng. Có lẽ chúng ta có thể tìm hiểu thêm về bản chất của dữ liệu bằng cách làm sạch nó nhiều hơn.

## Đường hồi quy tuyến tính

Như bạn đã học trong Bài học 1, mục tiêu của một bài tập hồi quy tuyến tính là có thể vẽ một đường để:

- **Hiển thị mối quan hệ giữa các biến**. Hiển thị mối quan hệ giữa các biến
- **Dự đoán**. Dự đoán chính xác nơi một điểm dữ liệu mới sẽ nằm trong mối quan hệ với đường đó.

Thông thường, **Hồi quy Bình phương Tối thiểu** được sử dụng để vẽ loại đường này. Thuật ngữ 'bình phương tối thiểu' có nghĩa là tất cả các điểm dữ liệu xung quanh đường hồi quy được bình phương và sau đó cộng lại. Lý tưởng nhất, tổng cuối cùng này càng nhỏ càng tốt, vì chúng ta muốn số lỗi thấp, hay `bình phương tối thiểu`.

Chúng ta làm như vậy vì muốn mô hình hóa một đường có khoảng cách tích lũy nhỏ nhất từ tất cả các điểm dữ liệu của chúng ta. Chúng ta cũng bình phương các giá trị trước khi cộng chúng vì chúng ta quan tâm đến độ lớn của chúng hơn là hướng của chúng.

> **🧮 Hiển thị toán học**
> 
> Đường này, được gọi là _đường phù hợp nhất_, có thể được biểu diễn bằng [một phương trình](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` là 'biến giải thích'. `Y` là 'biến phụ thuộc'. Độ dốc của đường là `b` và `a` là giao điểm với trục y, tức là giá trị của `Y` khi `X = 0`.
>
>![tính độ dốc](../../../../2-Regression/3-Linear/images/slope.png)
>
> Đầu tiên, tính độ dốc `b`. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)
>
> Nói cách khác, và liên quan đến câu hỏi ban đầu về dữ liệu bí ngô của chúng ta: "dự đoán giá của một giạ bí ngô theo tháng", `X` sẽ là giá và `Y` sẽ là tháng bán.
>
>![hoàn thành phương trình](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Tính giá trị của Y. Nếu bạn đang trả khoảng $4, chắc hẳn là tháng Tư! Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)
>
> Toán học tính toán đường này phải thể hiện độ dốc của đường, cũng phụ thuộc vào giao điểm, hoặc vị trí của `Y` khi `X = 0`.
>
> Bạn có thể quan sát phương pháp tính toán các giá trị này trên trang web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Cũng ghé thăm [máy tính Bình phương Tối thiểu này](https://www.mathsisfun.com/data/least-squares-calculator.html) để xem cách các giá trị số ảnh hưởng đến đường.

## Tương quan

Một thuật ngữ khác cần hiểu là **Hệ số Tương quan** giữa các biến X và Y cho trước. Sử dụng biểu đồ phân tán, bạn có thể nhanh chóng hình dung hệ số này. Một biểu đồ với các điểm dữ liệu phân tán theo một đường gọn gàng có tương quan cao, nhưng một biểu đồ với các điểm dữ liệu phân tán khắp nơi giữa X và Y có tương quan thấp.

Một mô hình hồi quy tuyến tính tốt sẽ là mô hình có Hệ số Tương quan cao (gần 1 hơn 0) sử dụng phương pháp Hồi quy Bình phương Tối thiểu với một đường hồi quy.

✅ Chạy notebook đi kèm bài học này và xem biểu đồ phân tán Giá theo Tháng. Dữ liệu liên kết Tháng với Giá bán bí ngô có vẻ có tương quan cao hay thấp, theo cách bạn diễn giải trực quan biểu đồ phân tán? Điều đó có thay đổi nếu bạn sử dụng thước đo chi tiết hơn thay vì `Tháng`, ví dụ như *ngày trong năm* (tức là số ngày kể từ đầu năm)?

Trong đoạn mã dưới đây, chúng ta sẽ giả định rằng chúng ta đã làm sạch dữ liệu và thu được một khung dữ liệu gọi là `new_pumpkins`, tương tự như sau:

ID | Tháng | NgàyTrongNăm | Loại | Thành phố | Gói | Giá thấp | Giá cao | Giá
---|-------|--------------|------|-----------|-----|----------|---------|-----
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Đoạn mã để làm sạch dữ liệu có sẵn trong [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Chúng tôi đã thực hiện các bước làm sạch tương tự như trong bài học trước và đã tính toán cột `NgàyTrongNăm` bằng cách sử dụng biểu thức sau:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Bây giờ bạn đã hiểu toán học đằng sau hồi quy tuyến tính, hãy tạo một mô hình Hồi quy để xem liệu chúng ta có thể dự đoán gói bí ngô nào sẽ có giá tốt nhất. Ai đó mua bí ngô cho một khu vườn bí ngô vào dịp lễ có thể muốn thông tin này để tối ưu hóa việc mua các gói bí ngô cho khu vườn.

## Tìm kiếm Tương quan

[![ML cho người mới bắt đầu - Tìm kiếm Tương quan: Chìa khóa cho Hồi quy Tuyến tính](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML cho người mới bắt đầu - Tìm kiếm Tương quan: Chìa khóa cho Hồi quy Tuyến tính")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về tương quan.

Từ bài học trước, bạn có thể đã thấy rằng giá trung bình cho các tháng khác nhau trông như thế này:

<img alt="Giá trung bình theo tháng" src="../2-Data/images/barchart.png" width="50%"/>

Điều này gợi ý rằng có thể có một số tương quan, và chúng ta có thể thử huấn luyện mô hình hồi quy tuyến tính để dự đoán mối quan hệ giữa `Tháng` và `Giá`, hoặc giữa `NgàyTrongNăm` và `Giá`. Đây là biểu đồ phân tán cho thấy mối quan hệ sau:

<img alt="Biểu đồ phân tán Giá vs. Ngày trong Năm" src="images/scatter-dayofyear.png" width="50%" /> 

Hãy xem liệu có tương quan nào không bằng cách sử dụng hàm `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Có vẻ như tương quan khá nhỏ, -0.15 theo `Tháng` và -0.17 theo `NgàyTrongNăm`, nhưng có thể có một mối quan hệ quan trọng khác. Có vẻ như có các cụm giá khác nhau tương ứng với các loại bí ngô khác nhau. Để xác nhận giả thuyết này, hãy vẽ từng loại bí ngô bằng một màu khác nhau. Bằng cách truyền tham số `ax` vào hàm vẽ biểu đồ phân tán, chúng ta có thể vẽ tất cả các điểm trên cùng một biểu đồ:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Biểu đồ phân tán Giá vs. Ngày trong Năm" src="images/scatter-dayofyear-color.png" width="50%" /> 

Cuộc điều tra của chúng ta gợi ý rằng loại bí ngô có ảnh hưởng lớn hơn đến giá tổng thể so với ngày bán thực tế. Chúng ta có thể thấy điều này với biểu đồ cột:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Biểu đồ cột giá vs loại bí ngô" src="images/price-by-variety.png" width="50%" /> 

Hãy tập trung vào một loại bí ngô, loại 'pie type', và xem ngày bán có ảnh hưởng gì đến giá:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Biểu đồ phân tán Giá vs. Ngày trong Năm" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Nếu bây giờ chúng ta tính toán tương quan giữa `Giá` và `NgàyTrongNăm` bằng cách sử dụng hàm `corr`, chúng ta sẽ nhận được giá trị khoảng `-0.27` - điều này có nghĩa là việc huấn luyện một mô hình dự đoán là hợp lý.

> Trước khi huấn luyện mô hình hồi quy tuyến tính, điều quan trọng là phải đảm bảo rằng dữ liệu của chúng ta đã được làm sạch. Hồi quy tuyến tính không hoạt động tốt với các giá trị bị thiếu, do đó, hợp lý để loại bỏ tất cả các ô trống:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Một cách tiếp cận khác là điền các giá trị trống bằng giá trị trung bình từ cột tương ứng.

## Hồi quy Tuyến tính Đơn giản

[![ML cho người mới bắt đầu - Hồi quy Tuyến tính và Đa thức sử dụng Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML cho người mới bắt đầu - Hồi quy Tuyến tính và Đa thức sử dụng Scikit-learn")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về hồi quy tuyến tính và đa thức.

Để huấn luyện mô hình Hồi quy Tuyến tính của chúng ta, chúng ta sẽ sử dụng thư viện **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Chúng ta bắt đầu bằng cách tách các giá trị đầu vào (đặc trưng) và đầu ra mong đợi (nhãn) thành các mảng numpy riêng biệt:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Lưu ý rằng chúng ta phải thực hiện `reshape` trên dữ liệu đầu vào để gói Hồi quy Tuyến tính hiểu đúng. Hồi quy Tuyến tính yêu cầu một mảng 2D làm đầu vào, trong đó mỗi hàng của mảng tương ứng với một vector của các đặc trưng đầu vào. Trong trường hợp của chúng ta, vì chỉ có một đầu vào - chúng ta cần một mảng có hình dạng N×1, trong đó N là kích thước tập dữ liệu.

Sau đó, chúng ta cần chia dữ liệu thành tập huấn luyện và tập kiểm tra, để có thể xác thực mô hình sau khi huấn luyện:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Cuối cùng, việc huấn luyện mô hình Hồi quy Tuyến tính thực tế chỉ mất hai dòng mã. Chúng ta định nghĩa đối tượng `LinearRegression`, và khớp nó với dữ liệu của chúng ta bằng phương thức `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Đối tượng `LinearRegression` sau khi được khớp chứa tất cả các hệ số của hồi quy, có thể truy cập bằng thuộc tính `.coef_`. Trong trường hợp của chúng ta, chỉ có một hệ số, giá trị này sẽ khoảng `-0.017`. Điều này có nghĩa là giá dường như giảm một chút theo thời gian, nhưng không quá nhiều, khoảng 2 xu mỗi ngày. Chúng ta cũng có thể truy cập điểm giao của hồi quy với trục Y bằng `lin_reg.intercept_` - giá trị này sẽ khoảng `21` trong trường hợp của chúng ta, chỉ ra giá vào đầu năm.

Để xem mô hình của chúng ta chính xác đến mức nào, chúng ta có thể dự đoán giá trên tập kiểm tra, và sau đó đo lường mức độ gần gũi giữa dự đoán và giá trị mong đợi. Điều này có thể được thực hiện bằng cách sử dụng chỉ số lỗi bình phương trung bình (MSE), là trung bình của tất cả các sai lệch bình phương giữa giá trị mong đợi và giá trị dự đoán.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Lỗi của chúng ta dường như nằm ở khoảng 2 điểm, tương đương ~17%. Không quá tốt. Một chỉ số khác để đánh giá chất lượng mô hình là **hệ số xác định**, có thể được tính như sau:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Nếu giá trị là 0, điều đó có nghĩa là mô hình không xem xét dữ liệu đầu vào và hoạt động như *dự đoán tuyến tính tệ nhất*, chỉ đơn giản là giá trị trung bình của kết quả. Giá trị 1 có nghĩa là chúng ta có thể dự đoán hoàn hảo tất cả các đầu ra mong đợi. Trong trường hợp của chúng ta, hệ số xác định khoảng 0.06, khá thấp.

Chúng ta cũng có thể vẽ dữ liệu kiểm tra cùng với đường hồi quy để thấy rõ hơn cách hồi quy hoạt động trong trường hợp này:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Hồi quy tuyến tính" src="images/linear-results.png" width="50%" />

## Hồi quy đa thức

Một loại hồi quy tuyến tính khác là hồi quy đa thức. Mặc dù đôi khi có mối quan hệ tuyến tính giữa các biến - ví dụ, bí ngô có thể tích lớn hơn thì giá cao hơn - nhưng đôi khi những mối quan hệ này không thể được biểu diễn bằng mặt phẳng hoặc đường thẳng.

✅ Đây là [một số ví dụ](https://online.stat.psu.edu/stat501/lesson/9/9.8) về dữ liệu có thể sử dụng hồi quy đa thức.

Hãy xem lại mối quan hệ giữa Ngày và Giá. Biểu đồ phân tán này có nhất thiết phải được phân tích bằng một đường thẳng không? Giá cả không thể dao động sao? Trong trường hợp này, bạn có thể thử hồi quy đa thức.

✅ Đa thức là các biểu thức toán học có thể bao gồm một hoặc nhiều biến và hệ số.

Hồi quy đa thức tạo ra một đường cong để phù hợp hơn với dữ liệu phi tuyến tính. Trong trường hợp của chúng ta, nếu chúng ta thêm biến `DayOfYear` bình phương vào dữ liệu đầu vào, chúng ta có thể phù hợp với dữ liệu bằng một đường cong parabol, có điểm cực tiểu tại một thời điểm nhất định trong năm.

Scikit-learn bao gồm một [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) hữu ích để kết hợp các bước xử lý dữ liệu khác nhau. Một **pipeline** là một chuỗi các **bộ ước lượng**. Trong trường hợp của chúng ta, chúng ta sẽ tạo một pipeline đầu tiên thêm các đặc trưng đa thức vào mô hình, sau đó huấn luyện hồi quy:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Sử dụng `PolynomialFeatures(2)` có nghĩa là chúng ta sẽ bao gồm tất cả các đa thức bậc hai từ dữ liệu đầu vào. Trong trường hợp của chúng ta, điều này chỉ có nghĩa là `DayOfYear`<sup>2</sup>, nhưng với hai biến đầu vào X và Y, điều này sẽ thêm X<sup>2</sup>, XY và Y<sup>2</sup>. Chúng ta cũng có thể sử dụng các đa thức bậc cao hơn nếu muốn.

Pipeline có thể được sử dụng theo cách tương tự như đối tượng `LinearRegression` ban đầu, tức là chúng ta có thể `fit` pipeline, sau đó sử dụng `predict` để nhận kết quả dự đoán. Đây là biểu đồ hiển thị dữ liệu kiểm tra và đường cong xấp xỉ:

<img alt="Hồi quy đa thức" src="images/poly-results.png" width="50%" />

Sử dụng hồi quy đa thức, chúng ta có thể đạt được MSE thấp hơn một chút và hệ số xác định cao hơn, nhưng không đáng kể. Chúng ta cần xem xét các đặc trưng khác!

> Bạn có thể thấy rằng giá bí ngô thấp nhất được quan sát vào khoảng Halloween. Làm thế nào bạn giải thích điều này?

🎃 Chúc mừng, bạn vừa tạo một mô hình giúp dự đoán giá bí ngô làm bánh. Bạn có thể lặp lại quy trình tương tự cho tất cả các loại bí ngô, nhưng điều đó sẽ rất tẻ nhạt. Hãy cùng học cách đưa loại bí ngô vào mô hình của chúng ta!

## Đặc trưng phân loại

Trong thế giới lý tưởng, chúng ta muốn có thể dự đoán giá cho các loại bí ngô khác nhau bằng cùng một mô hình. Tuy nhiên, cột `Variety` hơi khác so với các cột như `Month`, vì nó chứa các giá trị không phải số. Những cột như vậy được gọi là **phân loại**.

[![ML cho người mới bắt đầu - Dự đoán đặc trưng phân loại với hồi quy tuyến tính](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML cho người mới bắt đầu - Dự đoán đặc trưng phân loại với hồi quy tuyến tính")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về cách sử dụng đặc trưng phân loại.

Dưới đây là cách giá trung bình phụ thuộc vào loại bí ngô:

<img alt="Giá trung bình theo loại" src="images/price-by-variety.png" width="50%" />

Để đưa loại bí ngô vào mô hình, trước tiên chúng ta cần chuyển đổi nó sang dạng số, hoặc **mã hóa**. Có một số cách để thực hiện:

* **Mã hóa số đơn giản** sẽ tạo một bảng các loại khác nhau, sau đó thay thế tên loại bằng một chỉ số trong bảng đó. Đây không phải là ý tưởng tốt nhất cho hồi quy tuyến tính, vì hồi quy tuyến tính sử dụng giá trị số thực của chỉ số và thêm nó vào kết quả, nhân với một hệ số nào đó. Trong trường hợp của chúng ta, mối quan hệ giữa số chỉ số và giá rõ ràng là không tuyến tính, ngay cả khi chúng ta đảm bảo rằng các chỉ số được sắp xếp theo một cách cụ thể.
* **Mã hóa one-hot** sẽ thay thế cột `Variety` bằng 4 cột khác nhau, mỗi cột cho một loại. Mỗi cột sẽ chứa `1` nếu hàng tương ứng thuộc loại đó, và `0` nếu không. Điều này có nghĩa là sẽ có bốn hệ số trong hồi quy tuyến tính, mỗi hệ số cho một loại bí ngô, chịu trách nhiệm cho "giá khởi điểm" (hoặc "giá bổ sung") cho loại cụ thể đó.

Dưới đây là mã để mã hóa one-hot một loại:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Để huấn luyện hồi quy tuyến tính sử dụng loại mã hóa one-hot làm đầu vào, chúng ta chỉ cần khởi tạo dữ liệu `X` và `y` một cách chính xác:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Phần còn lại của mã giống như những gì chúng ta đã sử dụng ở trên để huấn luyện hồi quy tuyến tính. Nếu bạn thử, bạn sẽ thấy rằng sai số bình phương trung bình gần như giống nhau, nhưng chúng ta đạt được hệ số xác định cao hơn (~77%). Để có dự đoán chính xác hơn, chúng ta có thể xem xét thêm các đặc trưng phân loại khác, cũng như các đặc trưng số như `Month` hoặc `DayOfYear`. Để có một mảng lớn các đặc trưng, chúng ta có thể sử dụng `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Ở đây chúng ta cũng xem xét `City` và loại `Package`, điều này cho chúng ta MSE 2.84 (10%) và hệ số xác định 0.94!

## Tổng hợp tất cả

Để tạo mô hình tốt nhất, chúng ta có thể sử dụng dữ liệu kết hợp (mã hóa one-hot phân loại + số) từ ví dụ trên cùng với hồi quy đa thức. Dưới đây là mã hoàn chỉnh để bạn tiện tham khảo:

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Điều này sẽ cho chúng ta hệ số xác định tốt nhất gần 97% và MSE=2.23 (~8% lỗi dự đoán).

| Mô hình | MSE | Hệ số xác định |
|---------|-----|----------------|
| `DayOfYear` Tuyến tính | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Đa thức | 2.73 (17.0%) | 0.08 |
| `Variety` Tuyến tính | 5.24 (19.7%) | 0.77 |
| Tất cả đặc trưng Tuyến tính | 2.84 (10.5%) | 0.94 |
| Tất cả đặc trưng Đa thức | 2.23 (8.25%) | 0.97 |

🏆 Chúc mừng! Bạn đã tạo bốn mô hình hồi quy trong một bài học và cải thiện chất lượng mô hình lên 97%. Trong phần cuối về hồi quy, bạn sẽ học về hồi quy Logistic để xác định các danh mục.

---
## 🚀Thử thách

Thử nghiệm một số biến khác nhau trong notebook này để xem mối tương quan ảnh hưởng như thế nào đến độ chính xác của mô hình.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Trong bài học này, chúng ta đã học về hồi quy tuyến tính. Có những loại hồi quy quan trọng khác. Đọc về các kỹ thuật Stepwise, Ridge, Lasso và Elasticnet. Một khóa học tốt để học thêm là [khóa học Stanford Statistical Learning](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Bài tập

[Phát triển một mô hình](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.