<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T19:45:46+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "vi"
}
-->
# Xây dựng ứng dụng web sử dụng mô hình ML

Trong bài học này, bạn sẽ huấn luyện một mô hình ML trên một tập dữ liệu đặc biệt: _Các lần nhìn thấy UFO trong thế kỷ qua_, được lấy từ cơ sở dữ liệu của NUFORC.

Bạn sẽ học:

- Cách 'pickle' một mô hình đã được huấn luyện
- Cách sử dụng mô hình đó trong một ứng dụng Flask

Chúng ta sẽ tiếp tục sử dụng notebook để làm sạch dữ liệu và huấn luyện mô hình, nhưng bạn có thể tiến thêm một bước bằng cách khám phá cách sử dụng mô hình trong thực tế, cụ thể là trong một ứng dụng web.

Để làm điều này, bạn cần xây dựng một ứng dụng web sử dụng Flask.

## [Quiz trước bài học](https://ff-quizzes.netlify.app/en/ml/)

## Xây dựng ứng dụng

Có nhiều cách để xây dựng ứng dụng web sử dụng mô hình học máy. Kiến trúc web của bạn có thể ảnh hưởng đến cách mô hình được huấn luyện. Hãy tưởng tượng bạn đang làm việc trong một doanh nghiệp nơi nhóm khoa học dữ liệu đã huấn luyện một mô hình mà họ muốn bạn sử dụng trong ứng dụng.

### Những điều cần cân nhắc

Có nhiều câu hỏi bạn cần đặt ra:

- **Đó là ứng dụng web hay ứng dụng di động?** Nếu bạn đang xây dựng một ứng dụng di động hoặc cần sử dụng mô hình trong ngữ cảnh IoT, bạn có thể sử dụng [TensorFlow Lite](https://www.tensorflow.org/lite/) và tích hợp mô hình vào ứng dụng Android hoặc iOS.
- **Mô hình sẽ được lưu trữ ở đâu?** Trên đám mây hay cục bộ?
- **Hỗ trợ ngoại tuyến.** Ứng dụng có cần hoạt động ngoại tuyến không?
- **Công nghệ nào được sử dụng để huấn luyện mô hình?** Công nghệ được chọn có thể ảnh hưởng đến công cụ bạn cần sử dụng.
    - **Sử dụng TensorFlow.** Nếu bạn huấn luyện mô hình bằng TensorFlow, ví dụ, hệ sinh thái này cung cấp khả năng chuyển đổi mô hình TensorFlow để sử dụng trong ứng dụng web bằng cách sử dụng [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Sử dụng PyTorch.** Nếu bạn xây dựng mô hình bằng thư viện như [PyTorch](https://pytorch.org/), bạn có tùy chọn xuất mô hình ở định dạng [ONNX](https://onnx.ai/) (Open Neural Network Exchange) để sử dụng trong các ứng dụng web JavaScript có thể sử dụng [Onnx Runtime](https://www.onnxruntime.ai/). Tùy chọn này sẽ được khám phá trong bài học tương lai với mô hình được huấn luyện bằng Scikit-learn.
    - **Sử dụng Lobe.ai hoặc Azure Custom Vision.** Nếu bạn sử dụng hệ thống ML SaaS (Phần mềm như một dịch vụ) như [Lobe.ai](https://lobe.ai/) hoặc [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) để huấn luyện mô hình, loại phần mềm này cung cấp cách xuất mô hình cho nhiều nền tảng, bao gồm xây dựng API tùy chỉnh để truy vấn trên đám mây bởi ứng dụng trực tuyến của bạn.

Bạn cũng có cơ hội xây dựng toàn bộ ứng dụng web Flask có thể tự huấn luyện mô hình ngay trong trình duyệt web. Điều này cũng có thể được thực hiện bằng cách sử dụng TensorFlow.js trong ngữ cảnh JavaScript.

Đối với mục đích của chúng ta, vì chúng ta đã làm việc với notebook dựa trên Python, hãy khám phá các bước bạn cần thực hiện để xuất một mô hình đã huấn luyện từ notebook sang định dạng có thể đọc được bởi ứng dụng web xây dựng bằng Python.

## Công cụ

Để thực hiện nhiệm vụ này, bạn cần hai công cụ: Flask và Pickle, cả hai đều chạy trên Python.

✅ [Flask](https://palletsprojects.com/p/flask/) là gì? Được định nghĩa là 'micro-framework' bởi các nhà sáng tạo, Flask cung cấp các tính năng cơ bản của framework web sử dụng Python và một công cụ tạo mẫu để xây dựng trang web. Hãy xem [module học này](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) để thực hành xây dựng với Flask.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) là gì? Pickle 🥒 là một module Python dùng để tuần tự hóa và giải tuần tự hóa cấu trúc đối tượng Python. Khi bạn 'pickle' một mô hình, bạn tuần tự hóa hoặc làm phẳng cấu trúc của nó để sử dụng trên web. Hãy cẩn thận: pickle không an toàn về bản chất, vì vậy hãy cẩn thận nếu được yêu cầu 'un-pickle' một tệp. Một tệp pickled có hậu tố `.pkl`.

## Bài tập - làm sạch dữ liệu

Trong bài học này, bạn sẽ sử dụng dữ liệu từ 80,000 lần nhìn thấy UFO, được thu thập bởi [NUFORC](https://nuforc.org) (Trung tâm Báo cáo UFO Quốc gia). Dữ liệu này có một số mô tả thú vị về các lần nhìn thấy UFO, ví dụ:

- **Mô tả dài.** "Một người đàn ông xuất hiện từ một tia sáng chiếu xuống một cánh đồng cỏ vào ban đêm và chạy về phía bãi đậu xe của Texas Instruments".
- **Mô tả ngắn.** "những ánh sáng đuổi theo chúng tôi".

Bảng tính [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) bao gồm các cột về `city`, `state` và `country` nơi xảy ra lần nhìn thấy, hình dạng của vật thể (`shape`) và `latitude` và `longitude`.

Trong [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) trống được bao gồm trong bài học này:

1. import `pandas`, `matplotlib`, và `numpy` như bạn đã làm trong các bài học trước và import bảng tính ufos. Bạn có thể xem một mẫu tập dữ liệu:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Chuyển đổi dữ liệu ufos thành một dataframe nhỏ với tiêu đề mới. Kiểm tra các giá trị duy nhất trong trường `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Bây giờ, bạn có thể giảm lượng dữ liệu cần xử lý bằng cách loại bỏ các giá trị null và chỉ import các lần nhìn thấy trong khoảng từ 1-60 giây:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Import thư viện `LabelEncoder` của Scikit-learn để chuyển đổi các giá trị văn bản của quốc gia thành số:

    ✅ LabelEncoder mã hóa dữ liệu theo thứ tự bảng chữ cái

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Dữ liệu của bạn sẽ trông như thế này:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Bài tập - xây dựng mô hình

Bây giờ bạn có thể chuẩn bị huấn luyện mô hình bằng cách chia dữ liệu thành nhóm huấn luyện và kiểm tra.

1. Chọn ba đặc trưng bạn muốn huấn luyện làm vector X, và vector y sẽ là `Country`. Bạn muốn có thể nhập `Seconds`, `Latitude` và `Longitude` và nhận được mã quốc gia để trả về.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Huấn luyện mô hình của bạn bằng logistic regression:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

Độ chính xác không tệ **(khoảng 95%)**, không có gì ngạc nhiên, vì `Country` và `Latitude/Longitude` có mối tương quan.

Mô hình bạn tạo ra không quá đột phá vì bạn có thể suy luận một `Country` từ `Latitude` và `Longitude`, nhưng đây là một bài tập tốt để thử huấn luyện từ dữ liệu thô mà bạn đã làm sạch, xuất ra, và sau đó sử dụng mô hình này trong một ứng dụng web.

## Bài tập - 'pickle' mô hình của bạn

Bây giờ, đã đến lúc _pickle_ mô hình của bạn! Bạn có thể làm điều đó chỉ trong vài dòng mã. Sau khi _pickled_, tải mô hình đã pickled và kiểm tra nó với một mảng dữ liệu mẫu chứa các giá trị cho seconds, latitude và longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Mô hình trả về **'3'**, là mã quốc gia của Vương quốc Anh. Thật thú vị! 👽

## Bài tập - xây dựng ứng dụng Flask

Bây giờ bạn có thể xây dựng một ứng dụng Flask để gọi mô hình của bạn và trả về kết quả tương tự, nhưng theo cách trực quan hơn.

1. Bắt đầu bằng cách tạo một thư mục tên **web-app** bên cạnh tệp _notebook.ipynb_ nơi tệp _ufo-model.pkl_ của bạn nằm.

1. Trong thư mục đó, tạo thêm ba thư mục: **static**, với một thư mục **css** bên trong, và **templates**. Bạn sẽ có các tệp và thư mục sau:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Tham khảo thư mục giải pháp để xem ứng dụng hoàn chỉnh

1. Tệp đầu tiên cần tạo trong thư mục _web-app_ là tệp **requirements.txt**. Giống như _package.json_ trong ứng dụng JavaScript, tệp này liệt kê các phụ thuộc cần thiết cho ứng dụng. Trong **requirements.txt** thêm các dòng:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Bây giờ, chạy tệp này bằng cách điều hướng đến _web-app_:

    ```bash
    cd web-app
    ```

1. Trong terminal của bạn, gõ `pip install`, để cài đặt các thư viện được liệt kê trong _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Bây giờ, bạn đã sẵn sàng tạo thêm ba tệp để hoàn thành ứng dụng:

    1. Tạo **app.py** trong thư mục gốc.
    2. Tạo **index.html** trong thư mục _templates_.
    3. Tạo **styles.css** trong thư mục _static/css_.

1. Xây dựng tệp _styles.css_ với một vài kiểu:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. Tiếp theo, xây dựng tệp _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Hãy xem cách tạo mẫu trong tệp này. Lưu ý cú pháp 'mustache' xung quanh các biến sẽ được cung cấp bởi ứng dụng, như văn bản dự đoán: `{{}}`. Có một biểu mẫu gửi dự đoán đến route `/predict`.

    Cuối cùng, bạn đã sẵn sàng xây dựng tệp Python điều khiển việc sử dụng mô hình và hiển thị các dự đoán:

1. Trong `app.py` thêm:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Mẹo: khi bạn thêm [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) trong khi chạy ứng dụng web bằng Flask, bất kỳ thay đổi nào bạn thực hiện đối với ứng dụng của mình sẽ được phản ánh ngay lập tức mà không cần khởi động lại máy chủ. Lưu ý! Đừng bật chế độ này trong ứng dụng sản xuất.

Nếu bạn chạy `python app.py` hoặc `python3 app.py` - máy chủ web của bạn sẽ khởi động, cục bộ, và bạn có thể điền vào một biểu mẫu ngắn để nhận câu trả lời cho câu hỏi của bạn về nơi UFO đã được nhìn thấy!

Trước khi làm điều đó, hãy xem các phần của `app.py`:

1. Đầu tiên, các phụ thuộc được tải và ứng dụng bắt đầu.
1. Sau đó, mô hình được import.
1. Sau đó, index.html được render trên route chính.

Trên route `/predict`, một số điều xảy ra khi biểu mẫu được gửi:

1. Các biến biểu mẫu được thu thập và chuyển đổi thành mảng numpy. Sau đó, chúng được gửi đến mô hình và một dự đoán được trả về.
2. Các quốc gia mà chúng ta muốn hiển thị được render lại dưới dạng văn bản dễ đọc từ mã quốc gia dự đoán, và giá trị đó được gửi lại index.html để render trong mẫu.

Sử dụng mô hình theo cách này, với Flask và mô hình đã pickled, tương đối đơn giản. Điều khó nhất là hiểu dữ liệu cần gửi đến mô hình để nhận dự đoán có hình dạng như thế nào. Điều đó hoàn toàn phụ thuộc vào cách mô hình được huấn luyện. Mô hình này có ba điểm dữ liệu cần nhập để nhận dự đoán.

Trong môi trường chuyên nghiệp, bạn có thể thấy rằng giao tiếp tốt là cần thiết giữa những người huấn luyện mô hình và những người sử dụng nó trong ứng dụng web hoặc di động. Trong trường hợp của chúng ta, chỉ có một người, chính bạn!

---

## 🚀 Thử thách

Thay vì làm việc trong notebook và import mô hình vào ứng dụng Flask, bạn có thể huấn luyện mô hình ngay trong ứng dụng Flask! Hãy thử chuyển đổi mã Python của bạn trong notebook, có thể sau khi dữ liệu của bạn được làm sạch, để huấn luyện mô hình từ trong ứng dụng trên một route gọi là `train`. Những ưu và nhược điểm của việc theo đuổi phương pháp này là gì?

## [Quiz sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Có nhiều cách để xây dựng ứng dụng web sử dụng mô hình ML. Hãy lập danh sách các cách bạn có thể sử dụng JavaScript hoặc Python để xây dựng ứng dụng web tận dụng học máy. Xem xét kiến trúc: mô hình nên ở trong ứng dụng hay trên đám mây? Nếu là đám mây, bạn sẽ truy cập nó như thế nào? Vẽ ra một mô hình kiến trúc cho một giải pháp web ML ứng dụng.

## Bài tập

[Thử một mô hình khác](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.