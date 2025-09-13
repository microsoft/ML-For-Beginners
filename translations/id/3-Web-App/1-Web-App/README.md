<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T19:46:20+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "id"
}
-->
# Membangun Aplikasi Web untuk Menggunakan Model ML

Dalam pelajaran ini, Anda akan melatih model ML pada kumpulan data yang luar biasa: _Penampakan UFO selama abad terakhir_, yang bersumber dari database NUFORC.

Anda akan belajar:

- Cara 'pickle' model yang telah dilatih
- Cara menggunakan model tersebut dalam aplikasi Flask

Kita akan melanjutkan penggunaan notebook untuk membersihkan data dan melatih model, tetapi Anda dapat melangkah lebih jauh dengan mengeksplorasi penggunaan model 'di dunia nyata', yaitu dalam aplikasi web.

Untuk melakukan ini, Anda perlu membangun aplikasi web menggunakan Flask.

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Membangun aplikasi

Ada beberapa cara untuk membangun aplikasi web yang dapat menggunakan model machine learning. Arsitektur web Anda mungkin memengaruhi cara model Anda dilatih. Bayangkan Anda bekerja di sebuah perusahaan di mana tim data science telah melatih model yang ingin Anda gunakan dalam aplikasi.

### Pertimbangan

Ada banyak pertanyaan yang perlu Anda tanyakan:

- **Apakah ini aplikasi web atau aplikasi seluler?** Jika Anda membangun aplikasi seluler atau perlu menggunakan model dalam konteks IoT, Anda dapat menggunakan [TensorFlow Lite](https://www.tensorflow.org/lite/) dan menggunakan model tersebut dalam aplikasi Android atau iOS.
- **Di mana model akan ditempatkan?** Di cloud atau secara lokal?
- **Dukungan offline.** Apakah aplikasi harus berfungsi secara offline?
- **Teknologi apa yang digunakan untuk melatih model?** Teknologi yang dipilih dapat memengaruhi alat yang perlu Anda gunakan.
    - **Menggunakan TensorFlow.** Jika Anda melatih model menggunakan TensorFlow, misalnya, ekosistem tersebut menyediakan kemampuan untuk mengonversi model TensorFlow untuk digunakan dalam aplikasi web dengan menggunakan [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Menggunakan PyTorch.** Jika Anda membangun model menggunakan pustaka seperti [PyTorch](https://pytorch.org/), Anda memiliki opsi untuk mengekspornya dalam format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) untuk digunakan dalam aplikasi web JavaScript yang dapat menggunakan [Onnx Runtime](https://www.onnxruntime.ai/). Opsi ini akan dieksplorasi dalam pelajaran mendatang untuk model yang dilatih dengan Scikit-learn.
    - **Menggunakan Lobe.ai atau Azure Custom Vision.** Jika Anda menggunakan sistem ML SaaS (Software as a Service) seperti [Lobe.ai](https://lobe.ai/) atau [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) untuk melatih model, jenis perangkat lunak ini menyediakan cara untuk mengekspor model untuk berbagai platform, termasuk membangun API khusus yang dapat diakses di cloud oleh aplikasi online Anda.

Anda juga memiliki kesempatan untuk membangun seluruh aplikasi web Flask yang dapat melatih model itu sendiri di browser web. Ini juga dapat dilakukan menggunakan TensorFlow.js dalam konteks JavaScript.

Untuk tujuan kita, karena kita telah bekerja dengan notebook berbasis Python, mari kita eksplorasi langkah-langkah yang perlu Anda ambil untuk mengekspor model yang telah dilatih dari notebook tersebut ke format yang dapat dibaca oleh aplikasi web yang dibangun dengan Python.

## Alat

Untuk tugas ini, Anda memerlukan dua alat: Flask dan Pickle, keduanya berjalan di Python.

✅ Apa itu [Flask](https://palletsprojects.com/p/flask/)? Didefinisikan sebagai 'micro-framework' oleh pembuatnya, Flask menyediakan fitur dasar kerangka kerja web menggunakan Python dan mesin templating untuk membangun halaman web. Lihat [modul pembelajaran ini](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) untuk berlatih membangun dengan Flask.

✅ Apa itu [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 adalah modul Python yang melakukan serialisasi dan de-serialisasi struktur objek Python. Ketika Anda 'pickle' model, Anda melakukan serialisasi atau meratakan strukturnya untuk digunakan di web. Hati-hati: pickle tidak secara intrinsik aman, jadi berhati-hatilah jika diminta untuk 'un-pickle' file. File yang telah di-pickle memiliki akhiran `.pkl`.

## Latihan - membersihkan data Anda

Dalam pelajaran ini Anda akan menggunakan data dari 80.000 penampakan UFO, yang dikumpulkan oleh [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Data ini memiliki beberapa deskripsi menarik tentang penampakan UFO, misalnya:

- **Deskripsi panjang.** "Seorang pria muncul dari sinar cahaya yang bersinar di lapangan rumput pada malam hari dan dia berlari menuju tempat parkir Texas Instruments".
- **Deskripsi pendek.** "lampu-lampu itu mengejar kami".

Spreadsheet [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) mencakup kolom tentang `city`, `state`, dan `country` tempat penampakan terjadi, `shape` objek, serta `latitude` dan `longitude`.

Dalam [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) kosong yang disertakan dalam pelajaran ini:

1. import `pandas`, `matplotlib`, dan `numpy` seperti yang Anda lakukan dalam pelajaran sebelumnya dan import spreadsheet ufos. Anda dapat melihat contoh kumpulan data:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konversikan data ufos ke dataframe kecil dengan judul baru. Periksa nilai unik di bidang `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Sekarang, Anda dapat mengurangi jumlah data yang perlu kita tangani dengan menghapus nilai null dan hanya mengimpor penampakan antara 1-60 detik:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Import pustaka `LabelEncoder` dari Scikit-learn untuk mengonversi nilai teks untuk negara menjadi angka:

    ✅ LabelEncoder mengkodekan data secara alfabetis

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Data Anda seharusnya terlihat seperti ini:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Latihan - membangun model Anda

Sekarang Anda dapat bersiap untuk melatih model dengan membagi data menjadi kelompok pelatihan dan pengujian.

1. Pilih tiga fitur yang ingin Anda latih sebagai vektor X Anda, dan vektor y akan menjadi `Country`. Anda ingin dapat memasukkan `Seconds`, `Latitude`, dan `Longitude` dan mendapatkan id negara untuk dikembalikan.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Latih model Anda menggunakan regresi logistik:

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

Akurasi cukup baik **(sekitar 95%)**, tidak mengherankan, karena `Country` dan `Latitude/Longitude` berkorelasi.

Model yang Anda buat tidak terlalu revolusioner karena Anda seharusnya dapat menyimpulkan `Country` dari `Latitude` dan `Longitude`, tetapi ini adalah latihan yang baik untuk mencoba melatih dari data mentah yang telah Anda bersihkan, ekspor, dan kemudian menggunakan model ini dalam aplikasi web.

## Latihan - 'pickle' model Anda

Sekarang, saatnya untuk _pickle_ model Anda! Anda dapat melakukannya dalam beberapa baris kode. Setelah di-_pickle_, muat model yang telah di-pickle dan uji terhadap array data sampel yang berisi nilai untuk detik, latitude, dan longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model mengembalikan **'3'**, yang merupakan kode negara untuk Inggris. Luar biasa! 👽

## Latihan - membangun aplikasi Flask

Sekarang Anda dapat membangun aplikasi Flask untuk memanggil model Anda dan mengembalikan hasil serupa, tetapi dengan cara yang lebih menarik secara visual.

1. Mulailah dengan membuat folder bernama **web-app** di sebelah file _notebook.ipynb_ tempat file _ufo-model.pkl_ Anda berada.

1. Di dalam folder tersebut buat tiga folder lagi: **static**, dengan folder **css** di dalamnya, dan **templates**. Anda sekarang harus memiliki file dan direktori berikut:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Lihat folder solusi untuk melihat aplikasi yang sudah selesai

1. File pertama yang dibuat di folder _web-app_ adalah file **requirements.txt**. Seperti _package.json_ dalam aplikasi JavaScript, file ini mencantumkan dependensi yang diperlukan oleh aplikasi. Dalam **requirements.txt** tambahkan baris:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Sekarang, jalankan file ini dengan menavigasi ke _web-app_:

    ```bash
    cd web-app
    ```

1. Di terminal Anda ketik `pip install`, untuk menginstal pustaka yang tercantum dalam _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Sekarang, Anda siap membuat tiga file lagi untuk menyelesaikan aplikasi:

    1. Buat **app.py** di root.
    2. Buat **index.html** di direktori _templates_.
    3. Buat **styles.css** di direktori _static/css_.

1. Bangun file _styles.css_ dengan beberapa gaya:

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

1. Selanjutnya, bangun file _index.html_:

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

    Perhatikan templating dalam file ini. Perhatikan sintaks 'mustache' di sekitar variabel yang akan disediakan oleh aplikasi, seperti teks prediksi: `{{}}`. Ada juga formulir yang mengirimkan prediksi ke rute `/predict`.

    Akhirnya, Anda siap membangun file python yang menggerakkan konsumsi model dan tampilan prediksi:

1. Dalam `app.py` tambahkan:

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

    > 💡 Tip: saat Anda menambahkan [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) saat menjalankan aplikasi web menggunakan Flask, setiap perubahan yang Anda buat pada aplikasi Anda akan langsung tercermin tanpa perlu memulai ulang server. Hati-hati! Jangan aktifkan mode ini dalam aplikasi produksi.

Jika Anda menjalankan `python app.py` atau `python3 app.py` - server web Anda akan mulai berjalan secara lokal, dan Anda dapat mengisi formulir singkat untuk mendapatkan jawaban atas pertanyaan Anda tentang di mana UFO telah terlihat!

Sebelum melakukannya, lihat bagian-bagian dari `app.py`:

1. Pertama, dependensi dimuat dan aplikasi dimulai.
1. Kemudian, model diimpor.
1. Kemudian, index.html dirender di rute utama.

Di rute `/predict`, beberapa hal terjadi saat formulir dikirimkan:

1. Variabel formulir dikumpulkan dan dikonversi ke array numpy. Mereka kemudian dikirim ke model dan prediksi dikembalikan.
2. Negara-negara yang ingin ditampilkan dirender ulang sebagai teks yang dapat dibaca dari kode negara yang diprediksi, dan nilai tersebut dikirim kembali ke index.html untuk dirender dalam template.

Menggunakan model dengan cara ini, dengan Flask dan model yang di-pickle, cukup sederhana. Hal tersulit adalah memahami bentuk data yang harus dikirim ke model untuk mendapatkan prediksi. Itu semua tergantung pada bagaimana model dilatih. Model ini memiliki tiga titik data yang harus dimasukkan untuk mendapatkan prediksi.

Dalam pengaturan profesional, Anda dapat melihat betapa pentingnya komunikasi yang baik antara orang-orang yang melatih model dan mereka yang menggunakannya dalam aplikasi web atau seluler. Dalam kasus kita, hanya ada satu orang, yaitu Anda!

---

## 🚀 Tantangan

Alih-alih bekerja di notebook dan mengimpor model ke aplikasi Flask, Anda dapat melatih model langsung di dalam aplikasi Flask! Cobalah mengonversi kode Python Anda di notebook, mungkin setelah data Anda dibersihkan, untuk melatih model dari dalam aplikasi pada rute yang disebut `train`. Apa kelebihan dan kekurangan dari metode ini?

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Ada banyak cara untuk membangun aplikasi web yang menggunakan model ML. Buat daftar cara Anda dapat menggunakan JavaScript atau Python untuk membangun aplikasi web yang memanfaatkan machine learning. Pertimbangkan arsitektur: apakah model harus tetap berada di aplikasi atau hidup di cloud? Jika yang terakhir, bagaimana Anda mengaksesnya? Gambarlah model arsitektur untuk solusi web ML yang diterapkan.

## Tugas

[Coba model yang berbeda](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.