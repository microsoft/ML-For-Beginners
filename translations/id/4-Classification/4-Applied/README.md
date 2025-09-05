<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T19:54:34+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "id"
}
-->
# Membangun Aplikasi Web Rekomendasi Masakan

Dalam pelajaran ini, Anda akan membangun model klasifikasi menggunakan beberapa teknik yang telah Anda pelajari di pelajaran sebelumnya dan dengan dataset masakan lezat yang digunakan sepanjang seri ini. Selain itu, Anda akan membuat aplikasi web kecil untuk menggunakan model yang telah disimpan, memanfaatkan runtime web Onnx.

Salah satu penggunaan praktis yang paling berguna dari pembelajaran mesin adalah membangun sistem rekomendasi, dan Anda dapat mengambil langkah pertama ke arah itu hari ini!

[![Menyajikan aplikasi web ini](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Klik gambar di atas untuk video: Jen Looper membangun aplikasi web menggunakan data masakan yang telah diklasifikasi

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/)

Dalam pelajaran ini Anda akan belajar:

- Cara membangun model dan menyimpannya sebagai model Onnx
- Cara menggunakan Netron untuk memeriksa model
- Cara menggunakan model Anda dalam aplikasi web untuk inferensi

## Bangun model Anda

Membangun sistem pembelajaran mesin terapan adalah bagian penting dari memanfaatkan teknologi ini untuk sistem bisnis Anda. Anda dapat menggunakan model dalam aplikasi web Anda (dan dengan demikian menggunakannya dalam konteks offline jika diperlukan) dengan menggunakan Onnx.

Dalam [pelajaran sebelumnya](../../3-Web-App/1-Web-App/README.md), Anda membangun model Regresi tentang penampakan UFO, "pickled" model tersebut, dan menggunakannya dalam aplikasi Flask. Meskipun arsitektur ini sangat berguna untuk diketahui, ini adalah aplikasi Python full-stack, dan kebutuhan Anda mungkin mencakup penggunaan aplikasi JavaScript.

Dalam pelajaran ini, Anda dapat membangun sistem berbasis JavaScript sederhana untuk inferensi. Namun, pertama-tama Anda perlu melatih model dan mengonversinya untuk digunakan dengan Onnx.

## Latihan - latih model klasifikasi

Pertama, latih model klasifikasi menggunakan dataset masakan yang telah dibersihkan yang kita gunakan sebelumnya.

1. Mulailah dengan mengimpor pustaka yang berguna:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Anda memerlukan '[skl2onnx](https://onnx.ai/sklearn-onnx/)' untuk membantu mengonversi model Scikit-learn Anda ke format Onnx.

1. Kemudian, olah data Anda dengan cara yang sama seperti yang Anda lakukan di pelajaran sebelumnya, dengan membaca file CSV menggunakan `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Hapus dua kolom pertama yang tidak diperlukan dan simpan data yang tersisa sebagai 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Simpan label sebagai 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Mulai rutinitas pelatihan

Kami akan menggunakan pustaka 'SVC' yang memiliki akurasi yang baik.

1. Impor pustaka yang sesuai dari Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Pisahkan set pelatihan dan pengujian:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Bangun model Klasifikasi SVC seperti yang Anda lakukan di pelajaran sebelumnya:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Sekarang, uji model Anda dengan memanggil `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Cetak laporan klasifikasi untuk memeriksa kualitas model:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Seperti yang kita lihat sebelumnya, akurasinya baik:

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

### Konversi model Anda ke Onnx

Pastikan untuk melakukan konversi dengan jumlah Tensor yang tepat. Dataset ini memiliki 380 bahan yang terdaftar, jadi Anda perlu mencatat jumlah tersebut dalam `FloatTensorType`:

1. Konversi menggunakan jumlah tensor 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Buat file onx dan simpan sebagai **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Catatan, Anda dapat memasukkan [opsi](https://onnx.ai/sklearn-onnx/parameterized.html) dalam skrip konversi Anda. Dalam kasus ini, kami memasukkan 'nocl' sebagai True dan 'zipmap' sebagai False. Karena ini adalah model klasifikasi, Anda memiliki opsi untuk menghapus ZipMap yang menghasilkan daftar kamus (tidak diperlukan). `nocl` mengacu pada informasi kelas yang disertakan dalam model. Kurangi ukuran model Anda dengan mengatur `nocl` ke 'True'.

Menjalankan seluruh notebook sekarang akan membangun model Onnx dan menyimpannya ke folder ini.

## Lihat model Anda

Model Onnx tidak terlalu terlihat di Visual Studio Code, tetapi ada perangkat lunak gratis yang sangat baik yang banyak digunakan oleh peneliti untuk memvisualisasikan model guna memastikan bahwa model tersebut dibangun dengan benar. Unduh [Netron](https://github.com/lutzroeder/Netron) dan buka file model.onnx Anda. Anda dapat melihat model sederhana Anda yang divisualisasikan, dengan 380 input dan classifier yang terdaftar:

![Visual Netron](../../../../4-Classification/4-Applied/images/netron.png)

Netron adalah alat yang berguna untuk melihat model Anda.

Sekarang Anda siap menggunakan model keren ini dalam aplikasi web. Mari kita bangun aplikasi yang akan berguna ketika Anda melihat ke dalam lemari es Anda dan mencoba mencari tahu kombinasi bahan sisa mana yang dapat Anda gunakan untuk memasak masakan tertentu, seperti yang ditentukan oleh model Anda.

## Bangun aplikasi web rekomendasi

Anda dapat menggunakan model Anda langsung dalam aplikasi web. Arsitektur ini juga memungkinkan Anda menjalankannya secara lokal dan bahkan offline jika diperlukan. Mulailah dengan membuat file `index.html` di folder yang sama tempat Anda menyimpan file `model.onnx`.

1. Dalam file ini _index.html_, tambahkan markup berikut:

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

1. Sekarang, bekerja di dalam tag `body`, tambahkan sedikit markup untuk menampilkan daftar kotak centang yang mencerminkan beberapa bahan:

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

    Perhatikan bahwa setiap kotak centang diberi nilai. Ini mencerminkan indeks tempat bahan ditemukan sesuai dengan dataset. Apel, misalnya, dalam daftar alfabet ini, menempati kolom kelima, jadi nilainya adalah '4' karena kita mulai menghitung dari 0. Anda dapat berkonsultasi dengan [lembar kerja bahan](../../../../4-Classification/data/ingredient_indexes.csv) untuk menemukan indeks bahan tertentu.

    Melanjutkan pekerjaan Anda di file index.html, tambahkan blok skrip tempat model dipanggil setelah penutupan akhir `</div>`.

1. Pertama, impor [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime digunakan untuk memungkinkan menjalankan model Onnx Anda di berbagai platform perangkat keras, termasuk optimasi dan API untuk digunakan.

1. Setelah Runtime tersedia, Anda dapat memanggilnya:

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

Dalam kode ini, ada beberapa hal yang terjadi:

1. Anda membuat array dari 380 nilai yang mungkin (1 atau 0) untuk diatur dan dikirim ke model untuk inferensi, tergantung pada apakah kotak centang bahan dicentang.
2. Anda membuat array kotak centang dan cara untuk menentukan apakah mereka dicentang dalam fungsi `init` yang dipanggil saat aplikasi dimulai. Ketika kotak centang dicentang, array `ingredients` diubah untuk mencerminkan bahan yang dipilih.
3. Anda membuat fungsi `testCheckboxes` yang memeriksa apakah ada kotak centang yang dicentang.
4. Anda menggunakan fungsi `startInference` saat tombol ditekan dan, jika ada kotak centang yang dicentang, Anda memulai inferensi.
5. Rutinitas inferensi mencakup:
   1. Menyiapkan pemuatan asinkron model
   2. Membuat struktur Tensor untuk dikirim ke model
   3. Membuat 'feeds' yang mencerminkan input `float_input` yang Anda buat saat melatih model Anda (Anda dapat menggunakan Netron untuk memverifikasi nama tersebut)
   4. Mengirimkan 'feeds' ini ke model dan menunggu respons

## Uji aplikasi Anda

Buka sesi terminal di Visual Studio Code di folder tempat file index.html Anda berada. Pastikan Anda memiliki [http-server](https://www.npmjs.com/package/http-server) yang diinstal secara global, dan ketik `http-server` di prompt. Sebuah localhost akan terbuka dan Anda dapat melihat aplikasi web Anda. Periksa masakan apa yang direkomendasikan berdasarkan berbagai bahan:

![Aplikasi web bahan](../../../../4-Classification/4-Applied/images/web-app.png)

Selamat, Anda telah membuat aplikasi web 'rekomendasi' dengan beberapa bidang. Luangkan waktu untuk mengembangkan sistem ini!

## 🚀Tantangan

Aplikasi web Anda sangat minimal, jadi teruslah mengembangkannya menggunakan bahan dan indeksnya dari data [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Kombinasi rasa apa yang cocok untuk menciptakan hidangan nasional tertentu?

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Meskipun pelajaran ini hanya menyentuh kegunaan membuat sistem rekomendasi untuk bahan makanan, area aplikasi pembelajaran mesin ini sangat kaya dengan contoh. Bacalah lebih lanjut tentang bagaimana sistem ini dibangun:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tugas 

[Bangun rekomendasi baru](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.