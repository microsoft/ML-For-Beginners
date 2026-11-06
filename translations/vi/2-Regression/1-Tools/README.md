# Bắt đầu với Python và Scikit-learn cho các mô hình hồi quy

![Tóm tắt các hồi quy trong một sketchnote](../../../../translated_images/vi/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote bởi [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Bài kiểm tra trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bài học này có sẵn bằng R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Giới thiệu

Trong bốn bài học này, bạn sẽ khám phá cách xây dựng các mô hình hồi quy. Chúng ta sẽ sớm thảo luận về mục đích của chúng. Nhưng trước khi làm gì, hãy đảm bảo bạn đã có những công cụ phù hợp để bắt đầu quá trình!

Trong bài học này, bạn sẽ học cách:

- Cấu hình máy tính của bạn cho các nhiệm vụ học máy cục bộ.
- Làm việc với Jupyter Notebooks.
- Sử dụng Scikit-learn, bao gồm cả cài đặt.
- Khám phá hồi quy tuyến tính với bài tập thực hành.

## Cài đặt và cấu hình

[![ML cho người mới bắt đầu - Cài đặt công cụ sẵn sàng xây dựng mô hình Máy học](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML cho người mới bắt đầu - Cài đặt công cụ sẵn sàng xây dựng mô hình Máy học")

> 🎥 Nhấp vào ảnh trên để xem video ngắn hướng dẫn cấu hình máy tính của bạn cho ML.

1. **Cài đặt Python**. Đảm bảo rằng [Python](https://www.python.org/downloads/) đã được cài đặt trên máy tính của bạn. Bạn sẽ sử dụng Python cho nhiều tác vụ khoa học dữ liệu và học máy. Hầu hết các hệ thống máy tính đã có sẵn cài đặt Python. Ngoài ra còn có các [Gói mã Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) hữu ích giúp việc thiết lập dễ dàng hơn cho một số người dùng.

   Tuy nhiên, một số việc sử dụng Python đòi hỏi phiên bản phần mềm khác nhau. Vì lý do này, việc làm việc trong một [môi trường ảo](https://docs.python.org/3/library/venv.html) là hữu ích.

2. **Cài đặt Visual Studio Code**. Đảm bảo bạn đã cài đặt Visual Studio Code trên máy tính. Theo các hướng dẫn này để [cài đặt Visual Studio Code](https://code.visualstudio.com/) cho cài đặt cơ bản. Bạn sẽ sử dụng Python trong Visual Studio Code trong khóa học này, vì vậy bạn có thể muốn làm quen với cách [cấu hình Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) để phát triển Python.

   > Làm quen với Python bằng cách thực hành qua bộ sưu tập các [mô-đun học tập](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Cài đặt Python với Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Cài đặt Python với Visual Studio Code")
   >
   > 🎥 Nhấp vào ảnh trên để xem video: sử dụng Python trong VS Code.

3. **Cài đặt Scikit-learn**, bằng cách làm theo [các hướng dẫn này](https://scikit-learn.org/stable/install.html). Vì bạn cần đảm bảo dùng Python 3, nên sử dụng môi trường ảo. Lưu ý, nếu bạn cài thư viện này trên Mac M1, có các chỉ dẫn đặc biệt trên trang được liên kết phía trên.

1. **Cài đặt Jupyter Notebook**. Bạn sẽ cần phải [cài gói Jupyter](https://pypi.org/project/jupyter/).

## Môi trường viết mã ML của bạn

Bạn sẽ sử dụng **notebooks** để phát triển mã Python và tạo các mô hình học máy. Loại tệp này là công cụ phổ biến với các nhà khoa học dữ liệu, có thể nhận biết qua hậu tố hoặc phần mở rộng `.ipynb`.

Notebooks là môi trường tương tác cho phép lập trình viên vừa viết mã vừa thêm ghi chú và tài liệu xung quanh mã, rất hữu ích cho các dự án thử nghiệm hoặc nghiên cứu.

[![ML cho người mới bắt đầu - Thiết lập Jupyter Notebooks để bắt đầu xây dựng mô hình hồi quy](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML cho người mới bắt đầu - Thiết lập Jupyter Notebooks để bắt đầu xây dựng mô hình hồi quy")

> 🎥 Nhấp vào ảnh trên để xem video ngắn hướng dẫn thực hành bài tập này.

### Bài tập - làm việc với một notebook

Trong thư mục này, bạn sẽ tìm thấy tệp _notebook.ipynb_.

1. Mở _notebook.ipynb_ trong Visual Studio Code.

   Một server Jupyter sẽ khởi động với Python 3+ được bật. Bạn sẽ thấy các vùng của notebook có thể `run`, các đoạn mã. Bạn có thể chạy một đoạn mã bằng cách chọn biểu tượng giống nút phát.

1. Chọn biểu tượng `md` và thêm chút markdown, cùng với văn bản **# Chào mừng đến với notebook của bạn**.

   Tiếp theo, thêm một số mã Python.

1. Gõ **print('hello notebook')** trong đoạn mã.
1. Chọn mũi tên để chạy mã.

   Bạn sẽ thấy câu lệnh được in ra:

    ```output
    hello notebook
    ```

![VS Code với notebook đang mở](../../../../translated_images/vi/notebook.4a3ee31f396b8832.webp)

Bạn có thể xen kẽ mã với các bình luận để tự ghi chú cho notebook.

✅ Hãy nghĩ một lát xem môi trường làm việc của lập trình viên web khác biệt như thế nào so với của nhà khoa học dữ liệu.

## Khởi động với Scikit-learn

Bây giờ Python đã được thiết lập trong môi trường cục bộ của bạn, và bạn đã quen thuộc với Jupyter Notebooks, hãy làm quen với Scikit-learn (phát âm là `sci` như trong `science`). Scikit-learn cung cấp một [API rộng](https://scikit-learn.org/stable/modules/classes.html#api-ref) giúp bạn thực hiện các nhiệm vụ ML.

Theo [trang web](https://scikit-learn.org/stable/getting_started.html) của họ, "Scikit-learn là thư viện mã nguồn mở về học máy hỗ trợ học có giám sát và không giám sát. Nó cũng cung cấp nhiều công cụ để phù hợp mô hình, xử lý dữ liệu, chọn và đánh giá mô hình, cùng nhiều tiện ích khác."

Trong khóa học này, bạn sẽ sử dụng Scikit-learn và các công cụ khác để xây dựng mô hình học máy thực hiện các tác vụ được gọi là 'học máy truyền thống'. Chúng tôi cố ý tránh mạng nơ-ron và học sâu, vì những chủ đề đó sẽ được đề cập trong chương trình 'AI cho người mới bắt đầu' sắp tới.

Scikit-learn giúp việc xây dựng và đánh giá mô hình trở nên đơn giản. Nó tập trung vào dữ liệu số và có nhiều bộ dữ liệu sẵn để sử dụng như công cụ học tập. Nó cũng bao gồm các mô hình dựng sẵn để sinh viên thử. Hãy cùng khám phá quá trình tải dữ liệu đóng gói sẵn và sử dụng một bộ ước lượng tích hợp để tạo mô hình ML đầu tiên với Scikit-learn sử dụng dữ liệu cơ bản.

## Bài tập - notebook Scikit-learn đầu tiên

> Hướng dẫn này được lấy cảm hứng từ [ví dụ hồi quy tuyến tính](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) trên trang web của Scikit-learn.


[![ML cho người mới bắt đầu - Dự án hồi quy tuyến tính đầu tiên bằng Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML cho người mới bắt đầu - Dự án hồi quy tuyến tính đầu tiên bằng Python")

> 🎥 Nhấp vào ảnh trên để xem video ngắn hướng dẫn bài tập này.

Trong tệp _notebook.ipynb_ liên quan đến bài học này, xóa hết các ô bằng cách nhấn biểu tượng thùng rác.

Trong phần này, bạn sẽ làm việc với một bộ dữ liệu nhỏ về bệnh tiểu đường được tích hợp trong Scikit-learn để học tập. Hãy tưởng tượng bạn muốn thử nghiệm một phương pháp điều trị cho bệnh nhân tiểu đường. Mô hình học máy có thể giúp bạn xác định bệnh nhân nào sẽ phản ứng tốt hơn với phương pháp điều trị dựa trên sự kết hợp các biến số. Ngay cả một mô hình hồi quy rất cơ bản, khi được trực quan hóa, có thể cung cấp thông tin về các biến giúp bạn tổ chức các thử nghiệm lâm sàng lý thuyết.

✅ Có nhiều loại phương pháp hồi quy, và chọn phương pháp nào phụ thuộc vào câu trả lời bạn muốn tìm. Nếu muốn dự đoán chiều cao có thể của một người theo tuổi, bạn sẽ dùng hồi quy tuyến tính vì tìm giá trị **số**. Nếu muốn xác định xem một loại ẩm thực có phải là thuần chay hay không, bạn đang tìm **phân loại**, nên sẽ dùng hồi quy logistic. Bạn sẽ học thêm về hồi quy logistic sau. Hãy suy nghĩ về các câu hỏi bạn có thể đặt về dữ liệu, và phương pháp nào trong số này phù hợp hơn.

Hãy bắt đầu với nhiệm vụ này.

### Nhập thư viện

Cho nhiệm vụ này, chúng ta sẽ nhập một số thư viện:

- **matplotlib**. Đây là một [công cụ đồ họa](https://matplotlib.org/) hữu ích và chúng ta sẽ dùng nó để tạo biểu đồ đường.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) là thư viện hữu ích để xử lý dữ liệu số trong Python.
- **sklearn**. Đây là thư viện [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Nhập một số thư viện để hỗ trợ công việc của bạn.

1. Thêm các lệnh nhập bằng cách gõ đoạn mã sau:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ở trên, bạn nhập `matplotlib`, `numpy` và nhập `datasets`, `linear_model` cùng `model_selection` từ `sklearn`. `model_selection` dùng để chia dữ liệu thành các bộ huấn luyện và kiểm tra.

### Bộ dữ liệu tiểu đường

Bộ dữ liệu [tiểu đường](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) tích hợp có 442 mẫu dữ liệu về bệnh tiểu đường, với 10 biến đặc trưng, một số là:

- age: tuổi tính theo năm
- bmi: chỉ số khối cơ thể
- bp: huyết áp trung bình
- s1 tc: T-Cells (một loại bạch cầu)

✅ Bộ dữ liệu này bao gồm khái niệm 'giới tính' như một biến quan trọng trong nghiên cứu về tiểu đường. Nhiều bộ dữ liệu y tế bao gồm phân loại nhị phân này. Hãy suy nghĩ về cách phân loại như thế có thể loại trừ một số nhóm dân cư khỏi các phương pháp điều trị.

Bây giờ, hãy tải dữ liệu X và y.

> 🎓 Hãy nhớ rằng đây là học có giám sát, và chúng ta cần biến mục tiêu được đặt tên là 'y'.

Trong ô mã mới, tải bộ dữ liệu tiểu đường bằng cách gọi `load_diabetes()`. Tham số `return_X_y=True` báo hiệu rằng `X` sẽ là ma trận dữ liệu, và `y` là mục tiêu hồi quy.

1. Thêm một số lệnh in để hiển thị hình dạng ma trận dữ liệu và phần tử đầu tiên:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Kết quả trả về là một tuple. Bạn đang gán hai giá trị đầu của tuple cho `X` và `y` tương ứng. Tìm hiểu thêm [về tuple](https://wikipedia.org/wiki/Tuple).

    Bạn có thể thấy dữ liệu này có 442 phần tử, mỗi phần tử là mảng 10 phần tử:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Hãy suy nghĩ về mối quan hệ giữa dữ liệu và mục tiêu hồi quy. Hồi quy tuyến tính dự đoán quan hệ giữa đặc trưng X và biến mục tiêu y. Bạn có thể tìm [target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) cho bộ dữ liệu tiểu đường trong tài liệu không? Bộ dữ liệu này minh họa điều gì, dựa trên mục tiêu đó?

2. Tiếp theo, chọn một phần của bộ dữ liệu này để vẽ biểu đồ bằng cách chọn cột thứ 3 của dữ liệu. Bạn làm điều này bằng cách dùng toán tử `:` để chọn tất cả các hàng, sau đó chọn cột 3 qua chỉ số (2). Bạn cũng có thể đổi kích thước dữ liệu thành mảng 2 chiều - như yêu cầu để vẽ đồ thị - sử dụng `reshape(n_rows, n_columns)`. Nếu một trong các tham số là -1, kích thước tương ứng sẽ tự động được tính.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Bất cứ lúc nào, hãy in dữ liệu để kiểm tra hình dạng của nó.

3. Bây giờ bạn đã có dữ liệu để vẽ, hãy kiểm tra liệu máy có thể giúp xác định sự phân chia logic giữa các số trong bộ dữ liệu này không. Để làm điều đó, bạn cần chia cả dữ liệu (X) và mục tiêu (y) thành bộ kiểm tra và bộ huấn luyện. Scikit-learn có cách chia đơn giản; bạn có thể chia dữ liệu kiểm tra tại một điểm xác định.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Bây giờ bạn sẵn sàng huấn luyện mô hình! Tải mô hình hồi quy tuyến tính và huấn luyện nó với bộ huấn luyện X và y bằng `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` là hàm bạn sẽ thấy trong nhiều thư viện ML như TensorFlow

5. Sau đó, tạo dự đoán bằng dữ liệu kiểm tra, sử dụng hàm `predict()`. Hàm này dùng để vẽ đường thẳng phân chia nhóm dữ liệu.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Bây giờ là lúc hiển thị dữ liệu trên biểu đồ. Matplotlib là công cụ rất hữu ích cho việc này. Tạo biểu đồ điểm cho tất cả dữ liệu X và y của bộ kiểm tra, và dùng dự đoán để vẽ một đường thẳng ở vị trí phù hợp nhất giữa các nhóm dữ liệu mô hình.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![biểu đồ điểm thể hiện các điểm dữ liệu về tiểu đường](../../../../translated_images/vi/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Hãy suy nghĩ về những gì đang diễn ra ở đây. Một đường thẳng chạy qua nhiều điểm nhỏ của dữ liệu, nhưng nó thực sự làm gì? Bạn có thấy mình có thể dùng đường này để dự đoán vị trí một điểm dữ liệu mới chưa nhìn thấy dựa trên trục y của biểu đồ không? Hãy thử diễn đạt bằng lời công dụng thực tế của mô hình này.

Chúc mừng bạn đã xây dựng mô hình hồi quy tuyến tính đầu tiên, tạo dự đoán từ nó và hiển thị trên biểu đồ!

---
## 🚀Thử thách

Vẽ một biến khác từ bộ dữ liệu này. Gợi ý: chỉnh sửa dòng này: `X = X[:,2]`. Dựa vào mục tiêu của bộ dữ liệu này, bạn có thể khám phá gì về tiến trình bệnh tiểu đường như một căn bệnh?
## [Bài kiểm tra sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Trong hướng dẫn này, bạn làm việc với hồi quy tuyến tính đơn giản, thay vì hồi quy tuyến tính đơn biến hoặc đa biến. Đọc một chút về sự khác biệt giữa các phương pháp này, hoặc xem [video này](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Tìm hiểu thêm về khái niệm hồi quy và suy nghĩ về những loại câu hỏi nào có thể được trả lời bằng kỹ thuật này. Tham gia [hướng dẫn này](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) để tăng cường hiểu biết của bạn.

## Bài tập

[Một bộ dữ liệu khác](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng bản dịch tự động có thể chứa lỗi hoặc sai sót. Tài liệu gốc bằng ngôn ngữ gốc nên được coi là nguồn tin chính thức. Đối với thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->