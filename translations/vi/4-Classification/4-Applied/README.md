<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T19:54:06+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "vi"
}
-->
# Xây dựng ứng dụng web gợi ý món ăn

Trong bài học này, bạn sẽ xây dựng một mô hình phân loại bằng cách sử dụng một số kỹ thuật đã học trong các bài trước và với bộ dữ liệu món ăn ngon được sử dụng xuyên suốt loạt bài này. Ngoài ra, bạn sẽ xây dựng một ứng dụng web nhỏ để sử dụng mô hình đã lưu, tận dụng runtime web của Onnx.

Một trong những ứng dụng thực tiễn hữu ích nhất của học máy là xây dựng hệ thống gợi ý, và hôm nay bạn có thể bắt đầu bước đầu tiên trong hướng đi này!

[![Trình bày ứng dụng web này](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Nhấn vào hình ảnh trên để xem video: Jen Looper xây dựng một ứng dụng web sử dụng dữ liệu món ăn đã phân loại

## [Câu hỏi trước bài học](https://ff-quizzes.netlify.app/en/ml/)

Trong bài học này, bạn sẽ học:

- Cách xây dựng mô hình và lưu nó dưới dạng mô hình Onnx
- Cách sử dụng Netron để kiểm tra mô hình
- Cách sử dụng mô hình của bạn trong ứng dụng web để suy luận

## Xây dựng mô hình của bạn

Xây dựng hệ thống học máy ứng dụng là một phần quan trọng trong việc tận dụng các công nghệ này cho hệ thống kinh doanh của bạn. Bạn có thể sử dụng các mô hình trong ứng dụng web của mình (và do đó sử dụng chúng trong ngữ cảnh offline nếu cần) bằng cách sử dụng Onnx.

Trong một [bài học trước](../../3-Web-App/1-Web-App/README.md), bạn đã xây dựng một mô hình hồi quy về các lần nhìn thấy UFO, "pickled" nó, và sử dụng nó trong một ứng dụng Flask. Mặc dù kiến trúc này rất hữu ích để biết, nhưng nó là một ứng dụng Python full-stack, và yêu cầu của bạn có thể bao gồm việc sử dụng một ứng dụng JavaScript.

Trong bài học này, bạn có thể xây dựng một hệ thống cơ bản dựa trên JavaScript để suy luận. Tuy nhiên, trước tiên bạn cần huấn luyện một mô hình và chuyển đổi nó để sử dụng với Onnx.

## Bài tập - huấn luyện mô hình phân loại

Đầu tiên, huấn luyện một mô hình phân loại bằng cách sử dụng bộ dữ liệu món ăn đã được làm sạch mà chúng ta đã sử dụng.

1. Bắt đầu bằng cách nhập các thư viện hữu ích:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Bạn cần '[skl2onnx](https://onnx.ai/sklearn-onnx/)' để giúp chuyển đổi mô hình Scikit-learn của bạn sang định dạng Onnx.

1. Sau đó, làm việc với dữ liệu của bạn theo cách bạn đã làm trong các bài học trước, bằng cách đọc tệp CSV sử dụng `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Loại bỏ hai cột không cần thiết đầu tiên và lưu dữ liệu còn lại dưới dạng 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Lưu các nhãn dưới dạng 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Bắt đầu quy trình huấn luyện

Chúng ta sẽ sử dụng thư viện 'SVC' với độ chính xác tốt.

1. Nhập các thư viện phù hợp từ Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Tách tập huấn luyện và tập kiểm tra:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Xây dựng mô hình phân loại SVC như bạn đã làm trong bài học trước:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Bây giờ, kiểm tra mô hình của bạn bằng cách gọi `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. In báo cáo phân loại để kiểm tra chất lượng mô hình:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Như chúng ta đã thấy trước đó, độ chính xác là tốt:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Chuyển đổi mô hình của bạn sang Onnx

Đảm bảo thực hiện chuyển đổi với số Tensor phù hợp. Bộ dữ liệu này có 380 nguyên liệu được liệt kê, vì vậy bạn cần ghi chú số đó trong `FloatTensorType`:

1. Chuyển đổi sử dụng số tensor là 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Tạo tệp onx và lưu dưới dạng tệp **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Lưu ý, bạn có thể truyền vào [tùy chọn](https://onnx.ai/sklearn-onnx/parameterized.html) trong script chuyển đổi của mình. Trong trường hợp này, chúng ta đã truyền vào 'nocl' là True và 'zipmap' là False. Vì đây là mô hình phân loại, bạn có tùy chọn loại bỏ ZipMap, thứ tạo ra danh sách các từ điển (không cần thiết). `nocl` đề cập đến thông tin lớp được bao gồm trong mô hình. Giảm kích thước mô hình của bạn bằng cách đặt `nocl` là 'True'.

Chạy toàn bộ notebook bây giờ sẽ xây dựng một mô hình Onnx và lưu nó vào thư mục này.

## Xem mô hình của bạn

Các mô hình Onnx không hiển thị rõ ràng trong Visual Studio Code, nhưng có một phần mềm miễn phí rất tốt mà nhiều nhà nghiên cứu sử dụng để trực quan hóa mô hình nhằm đảm bảo rằng nó được xây dựng đúng cách. Tải xuống [Netron](https://github.com/lutzroeder/Netron) và mở tệp model.onnx của bạn. Bạn có thể thấy mô hình đơn giản của mình được trực quan hóa, với 380 đầu vào và bộ phân loại được liệt kê:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron là một công cụ hữu ích để xem các mô hình của bạn.

Bây giờ bạn đã sẵn sàng sử dụng mô hình thú vị này trong một ứng dụng web. Hãy xây dựng một ứng dụng sẽ hữu ích khi bạn nhìn vào tủ lạnh của mình và cố gắng tìm ra sự kết hợp của các nguyên liệu còn lại mà bạn có thể sử dụng để nấu một món ăn cụ thể, như được xác định bởi mô hình của bạn.

## Xây dựng ứng dụng web gợi ý

Bạn có thể sử dụng mô hình của mình trực tiếp trong một ứng dụng web. Kiến trúc này cũng cho phép bạn chạy nó cục bộ và thậm chí offline nếu cần. Bắt đầu bằng cách tạo tệp `index.html` trong cùng thư mục nơi bạn lưu tệp `model.onnx`.

1. Trong tệp này _index.html_, thêm đoạn mã sau:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Bây giờ, làm việc trong thẻ `body`, thêm một chút mã để hiển thị danh sách các hộp kiểm phản ánh một số nguyên liệu:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Lưu ý rằng mỗi hộp kiểm được gán một giá trị. Giá trị này phản ánh chỉ số nơi nguyên liệu được tìm thấy theo bộ dữ liệu. Ví dụ, táo trong danh sách theo thứ tự bảng chữ cái này chiếm cột thứ năm, vì vậy giá trị của nó là '4' vì chúng ta bắt đầu đếm từ 0. Bạn có thể tham khảo [bảng nguyên liệu](../../../../4-Classification/data/ingredient_indexes.csv) để tìm chỉ số của một nguyên liệu cụ thể.

    Tiếp tục làm việc trong tệp index.html, thêm một khối script nơi mô hình được gọi sau thẻ đóng `</div>` cuối cùng.

1. Đầu tiên, nhập [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime được sử dụng để cho phép chạy các mô hình Onnx của bạn trên nhiều nền tảng phần cứng, bao gồm các tối ưu hóa và một API để sử dụng.

1. Khi Runtime đã được thiết lập, bạn có thể gọi nó:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

Trong đoạn mã này, có một số điều đang diễn ra:

1. Bạn đã tạo một mảng gồm 380 giá trị có thể (1 hoặc 0) để được thiết lập và gửi đến mô hình để suy luận, tùy thuộc vào việc hộp kiểm nguyên liệu có được chọn hay không.
2. Bạn đã tạo một mảng các hộp kiểm và một cách để xác định liệu chúng có được chọn hay không trong hàm `init` được gọi khi ứng dụng bắt đầu. Khi một hộp kiểm được chọn, mảng `ingredients` được thay đổi để phản ánh nguyên liệu đã chọn.
3. Bạn đã tạo một hàm `testCheckboxes` để kiểm tra liệu có hộp kiểm nào được chọn hay không.
4. Bạn sử dụng hàm `startInference` khi nút được nhấn và, nếu có hộp kiểm nào được chọn, bạn bắt đầu suy luận.
5. Quy trình suy luận bao gồm:
   1. Thiết lập tải không đồng bộ của mô hình
   2. Tạo cấu trúc Tensor để gửi đến mô hình
   3. Tạo 'feeds' phản ánh đầu vào `float_input` mà bạn đã tạo khi huấn luyện mô hình của mình (bạn có thể sử dụng Netron để xác minh tên đó)
   4. Gửi các 'feeds' này đến mô hình và chờ phản hồi

## Kiểm tra ứng dụng của bạn

Mở một phiên terminal trong Visual Studio Code trong thư mục nơi tệp index.html của bạn nằm. Đảm bảo rằng bạn đã cài đặt [http-server](https://www.npmjs.com/package/http-server) toàn cầu, và gõ `http-server` tại dấu nhắc. Một localhost sẽ mở ra và bạn có thể xem ứng dụng web của mình. Kiểm tra món ăn nào được gợi ý dựa trên các nguyên liệu khác nhau:

![ứng dụng web nguyên liệu](../../../../4-Classification/4-Applied/images/web-app.png)

Chúc mừng, bạn đã tạo một ứng dụng web 'gợi ý' với một vài trường. Hãy dành thời gian để xây dựng hệ thống này!

## 🚀Thử thách

Ứng dụng web của bạn rất đơn giản, vì vậy hãy tiếp tục xây dựng nó bằng cách sử dụng các nguyên liệu và chỉ số của chúng từ dữ liệu [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Những sự kết hợp hương vị nào hoạt động để tạo ra một món ăn quốc gia cụ thể?

## [Câu hỏi sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Mặc dù bài học này chỉ đề cập đến tiện ích của việc tạo hệ thống gợi ý cho các nguyên liệu món ăn, lĩnh vực ứng dụng ML này rất phong phú với các ví dụ. Đọc thêm về cách các hệ thống này được xây dựng:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Bài tập 

[Hãy xây dựng một hệ thống gợi ý mới](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.