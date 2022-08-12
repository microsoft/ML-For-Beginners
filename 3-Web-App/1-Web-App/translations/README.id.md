# Bangun Aplikasi Web supaya dapat menggunakan Model ML

Dalam pelajaran ini, kamu akan melatih model ML pada kumpulan data 'diluar nalar': _penampakan UFO selama satu abad terakhir_, bersumber dari database NUFORC.

Kamu akan belajar:

- Cara mem-_pickle_ model yang sudah di latih (kamu akan tahu istilah _pickle_ nanti)
- Cara menggunakan model tersebut dengan Flask

Kita akan masih terus lanjut menggunakan _notebook_ untuk membersihkan data dan melatih model-nya, tetapi kamu akan mengambil proses satu langkah lebih jauh dengan menerapkan model 'di alam liar', alias di aplikasi web! (keren kan?).

Untuk melakukan hal tersebut, kamu perlu Flask untuk membangun aplikasi web.

## [Kuis Pra-Kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Membangun Aplikasi

Ada beberapa cara agar model _Machine Learning_ dapat digunakan pada aplikasi web. Arsitektur web kamu dapat memengaruhi cara model dilatih. Bayangkan kamu sedang bekerja pada sebuah bisnis, di mana sekumpulan ilmuan data telah melatih model dan mereka ingin kamu mengintegrasikannya pada sebuah aplikasi.

### Hal yang menjadi bahan pertimbangan

Pasti kamu bertanya-tanya:

- **Ini aplikasi web atau mobile ya?** Kalau kamu sedang membuat aplikasi _mobile_ atau perlu menggunakan model dalam konteks IoT, pakai [TensorFlow Lite](https://www.tensorflow.org/lite/) dan gunakan model di aplikasi Android atau iOS.
- **Nanti model-nya akan di taruh dimana?** Kamu bisa taruh di cloud atau [lokal](https://skillcrush.com/blog/whats-a-local-development-environment/).
- **Apakah dapat beroperasi secara offline?** Haruskah beroperasi secara offline? lihat [artikel berikut](https://www.qwak.com/post/online-vs-offline-machine-learning-whats-the-difference) untuk mengetahui lebih lanjut
- **Teknologi seperti apa yang diterapkan untuk melatih model?** Perlu kamu ketahui, teknologi yang dipilih dapat memengaruhi _tools_ yang akan gunakan.
    - **Pakai TensorFlow.** Seluruh ekosistem didalamnya (terkait TensorFlow) menawarkan kemampuan untuk dapat meng-konversi model yang dibuat menggunakan TensorFlow dengan menggunakan [TensorFlow.js](https://www.tensorflow.org/js/) agar dapat digunakan pada aplikasi web.
    - **Pakai PyTorch.** Jika kamu pakai [PyTorch](https://pytorch.org/) untuk membangun model, kamu perlu melakukan export ke format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) agar dapat dipanggil menggunakan bahasa javascript melalui [Onnx Runtime](https://www.onnxruntime.ai/) untuk keperluan pengembangan aplikasi web. Opsi ini akan kita pelajari lebih jauh di pelajaran mendatang.
    - **Pakai Lobe.ai atau Azure Custom Vision.** [Lobe.ai](https://lobe.ai/) atau [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-15963-cxa) adalah pengembangan Machine Learning yang berbasis Perangkat Lunak sebagai Layanan (ML SaaS) untuk keperluan melatih model. Tipe perangkat lunak ini menawarkan berbagai cara untuk ekspor model ke berbagai platform, termasuk membangun API yang dapat dipesan dan digunakan pada aplikasi online yang kamu miliki.

Kamu juga berkesempatan untuk membangun seluruh aplikasi web Flask yang dapat melatih model itu sendiri di browser web. Ini juga dapat dilakukan menggunakan TensorFlow.js dalam konteks JavaScript.

Sebagai upaya agar sejalan dengan yang kita pelajari selama ini,(menggunakan _notebook_ berbasis Python), mari jelajahi langkah-langkah yang perlu kamu ambil untuk mengekspor model terlatih dari notebook ke format yang dapat dibaca oleh aplikasi web yang mana format dibuat oleh Python langsung.

## Alat (Tools)

Pada tugas ini, kamu membutuhkan dua _tools_ (alat): **Flask** dan **Pickle**, keduanya berjalan pada bahasa Python

✅ Apa itu [Flask](https://palletsprojects.com/p/flask/)? Didefinisikan sebagai 'framework berbasis mikro' oleh pembuatnya, Flask menawarkan berbagai fitur dasar kerangka web menggunakan mesin _template_ untuk membangun halaman website. Lihat pada [modul ini](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-15963-cxa) untuk berlatih bagaimana cara membangun halaman web menggunakan Flask.

✅ Apa itu [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 (Hah... acar?) adalah moodul python yang dapat melakukan serialisasi dan de-serialisasi sebuah struktur objek Python (dikenal dengan istilah 'mem-pickle'). Saat kamu mem-'pickle' sebuah model, berarti kamu melakukan serialisasi atau meratakan struktur-nya agar dapat digunakan di web.
**Hati-hati**: Pickle tidak aman secara intrinsik, jadi berhati-hatilah jika diminta untuk melakukan 'un-pickle' sebuah file. File yang berkaitan dengan Pickle memiliki format nama file ber-akhiran `.pkl`

## Latihan - Bersihkan Data (Cleaning Data)

Pada latihan ini, kamu akan menggunakan sekumpulan data UFO yang pernah terlihat sebanyak 80,000 data yang diambil oleh [NUFORC](https://nuforc.org) (Semacam asosiasi pusat pelaporan UFO Nasional). Didalam data ini, ada deskripsi menarik terkait dengan penampakan UFO, contohnya :

- **Contoh deskripsi panjang.** "A man emerges from a beam of light that shines on a grassy field at night and he runs towards the Texas Instruments parking lot".
- **Contoh deskripsi pendek.** "the lights chased us".

Spreadsheet [ufos.csv](../data/ufos.csv) berisi kolom  `city`, `state` dan `country` dimana penampakan terjadi, serta bentukan objek yang dinotasikan sebagai `shape` serta `latitude` dan `longitude`-nya.

Buatlah sebuah [notebook](notebook.ipynb) kosong, dan ketik kode yang tercantum pada instruksi berikut:

1. impor `pandas`, `matplotlib`, dan `numpy` seperti yang sudah pernah kalian praktikan sebelumnya lalu impor juga dataset UFO nya. Lalu kamu bisa melihat contoh isian dataset-nya dengan memanggil `head()` :

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv') #sesuaikan dengan lokasi penaruhan ufos.csv
    ufos.head()
    ```

2. Konversi data ufo tersebut ke bentuk dataframe dengan kolom yang kita akan buat ulang. Lalu pada kolom `Country` kita akan cek untuk nilai yang unik (berbeda-beda nya).

    ```python
    ufos = pd.DataFrame(
        {
            'Seconds': ufos['duration (seconds)'],
            'Country': ufos['country'],
            'Latitude': ufos['latitude'],
            'Longitude': ufos['longitude']
        })
    
    ufos.Country.unique()
    ```

3. Sekarang, kurangi jumlah datanya, kamu akan perlu data penampakan yang terjadi antara 1 - 60 detik saja dan membuang selain yang dibutuhkan:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

4. Impor `LabelEncoder` dari Scikit-learn untuk mengkonversi seluruh teks untuk menotasikan negara ke dalam bentuk angka:

    ✅ LabelEncoder melakukan proses 'encode' data secara alfabet dan berurutan

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Maka data akan tampil sebagai berikut:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Latihan - Bangun Modelmu!

Sekarang kamu siap untuk melatih model!, diawali dengan membagi datanya terlebih dahulu menjadi dua grup, yaitu data latih (training data) dan data uji (test data).

1. Pilih tiga fitur yang ingin kamu latih sebagai vektor X, dan vektor y akan menjadi `Country`. Kita ingin nantinya dapat memasukkan `Seconds`, `Latitude` dan `Longitude` dan akan mengembalikan sebuah id negara yang mana nanti akan menjadi prediksi.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

2. Membangun _Logistic Regression_ dan melatihnya:

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

Akurasinya tidak terlalu buruk **(sekitar 95%)**,
The accuracy isn't bad **(around 95%)**, dan tidak heran juga bahwa `Country` dan `Latitude/Longitude` ternyata saling berkorelasi.

Model yang kamu buat tidak terlalu revolusioner karena kamu seharusnya dapat menyimpulkan `Country` dari `Latitude` dan `Longitude`-nya, tetapi ini adalah upaya latihan yang baik guna mencoba melatih dari data mentah yang sudah kamu bersihkan, ekspor, dan kemudian menggunakan model ini di aplikasi web kamu.

## Latihan - 'pickle' model kamu

Sekarang, saatnya untuk mem-pickle model kamu! kerennya kamu hanya dapat melakukannya dalam beberapa baris kode saja. Setelah di pickle, muat model yang dipickle tersebut dan uji terhadap data sampel yang kita sebutkan sendiri berisi nilai detik, lintang dan bujur didalam array,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[20,44,-12]]))
```

Model akan mem-prediksi **'3'**, yang mana jika kamu lihat kembali ke [sini pada tahap ke-2](#latihan---bersihkan-data-cleaning-data)(array posisi ke 3 dihitung dari 1) , adalah kode negara untuk Inggris. Wow, liar banget ya alien-nya ya! 👽

## Latihan - Membangun Aplikasi Flask

Sekarang kamu sudah bisa mulai menggunakan flask agar model kamu dapat dipanggil dan sama berfungsi dengan yang sudah kamu praktikan, tetapi melalui visualisasi yang menyenangkan.

1. Buat sebuah folder dan beri nama **web-app**, letakan folder tersebut dimana kamu menaruh file _notebook.ipynb_ dan _ufo-model.pkl_ terbentuk.

2. Didalam folder tersebut, buat tiga folder lagi: **static** didalamnya ada folder **css** saja, lalu **templates**. sekarang kamu memiliki struktur folder sebagai berikut:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Jadikan bentuk struktur folder diatas sebagai gambaran bentuk aplikasi jadinya sekaligus solusi bagi yang kebingungan.

3. Didalam folder _web-app_, buat file bernama **requirements.txt** . Kalau di aplikasi javascript Mirip _package.json_, file ini berisi dependensi yang diperlukan oleh aplikasi dengan menyebutkan nama dependensinya saja. Didalam **requirements.txt** tambahkan isian berikut:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

4. Buka terminal-nya, kamu akan pindah ke lokasi dimana requirements.txt diletakan yaitu folder web-app, jalankan perintah berikut:

    ```bash
    cd web-app
    ```

5. lalu ketik `pip install`, secara otomatis akan menginstall dependensi yang tercantum pada _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

6. Sekarang kamu siap untuk membuat tiga file lainnya agar aplikasi selesai dibangun:

    1. Buat **app.py** pada direktori akar.
    2. Buat **index.html** didalam direktori _templates_.
    3. Buat **styles.css** didalam direktori _static/css_.

7. Kamu isi _styles.css_ sebagai berikut:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4p x;
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

8. Selanjutnya, kamu isi _index.html_ sebagai berikut:

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
    
            <p>Menurut jumlah detik, latitude dan longitude, negara mana yang mungkin akan melapor penampakan UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Prediksi negara yang akan dihadiri UFO!</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Lihatlah template dalam pada ini. Perhatikan sintaks 'kumis' yang mengapit variabel, seperti teks prediksi: `{{}}`, sintaks tersebut disediakan oleh template engine dari Flask. Bahkan formulir prediksi diatas akan melakukan rute menuju `/predict`.

    Terakhir, kamu siap membuat file python yang akan mengkonsumsi model dan tampilan prediksi:

9. Pada file `app.py`, tambahkan baris berikut:

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
            "index.html", prediction_text="Kemungkinan negara: {}".format(countries[output]."-lah yang akan melihat penampakan UFO!")
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Tip: ketika kamu menambahkan [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) saat menjalankan aplikasi web menggunakan Flask, setiap perubahan yang kamu buat pada aplikasi akan segera terlihat tanpa perlu me-restart server. Perhatikan baik-baik! Jangan sampai kamu mengaaktifkan mode ini saat aplikasi berada pada _production stage_.

Jika Anda menjalankan perintah `python app.py` atau `python3 app.py` pada terminal - server web akan dijalankan secara lokal, dan kamu mengakses aplikasi dan mengisi formulir singkat untuk mendapatkan jawaban atas pertanyaan membara kamu tentang di mana UFO akan terlihat!

Sebelum melakukan hal tersebut, kita perlu lihat apa saja maksud yang terkandung didalam file `app.py`:

1. Pertama, dependensi dimuat dan aplikasi dimulai.
2. lalu, model di impor.
3. lalu, index.html di render pada _route_ awal (home).

Ketika formulir terkirim melalui rute `/predict`, berikut adalah hal yang terjadi didalamnya:

1. Variabel bentuk dikumpulkan dan dikonversi ke array numpy, kemudian dikirim ke model dan prediksi dihasilkan.
2. Negara yang ingin kita tampilkan dirender ulang sebagai teks yang dapat dibaca dari kode negara yang diprediksi, dan nilai tersebut dikirim kembali ke index.html untuk dirender dalam template.

Membangun dan menggunakan model dengan cara yang sudah kita lalui yaitu menggunakan Flask dan model yang di pickle ternyata relatif mudah. Yang paling sulit adalah memahami seperti apa bentuk data yang harus dikirim ke model untuk mendapatkan prediksi. Itu semua tergantung pada bagaimana model itu dilatih. Yang sudah kita praktikan lakukan yaitu kita memiliki tiga titik data yang akan dimasukkan untuk mendapatkan prediksi.

Dalam pengaturan profesional, kamu dapat melihat bagaimana komunikasi yang baik diperlukan antara orang-orang yang melatih model dan mereka yang menggunakannya di web atau aplikasi seluler. Dalam kasus kami, hanya satu orang, yaitu kamu!

---

## 🚀 Tantangan

Alih-alih bekerja di notebook dan mengimpor model ke aplikasi Flask, kamu bisa melatih model langsung di dalam aplikasi Flask! Coba ubah kode Python kamu di notebook, mungkin setelah data kamu dibersihkan, untuk melatih model dari dalam aplikasi pada rute yang disebut `train`. Apa pro dan kontra atas penerapan metode ini?

## [Kuis Pasca-belajar](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Review & Belajar Mandiri

Ada banyak cara untuk membangun aplikasi web untuk menggunakan model ML. Buat daftar ala kamu bagaimana agar dapat menggunakan JavaScript atau Python untuk membangun aplikasi web terintegrasi machine learning. Pertimbangkan arsitektur: haruskah model tetap berada di aplikasi atau hidup di cloud? Jika yang terakhir, bagaimana kamu mengaksesnya? Gambarkan model arsitektur untuk solusi web ML yang diterapkan.

## Tugas

[Coba model lainnya](assignment.md)
