# Bina Aplikasi Web untuk Menggunakan Model ML

Dalam pelajaran ini, anda akan melatih model ML pada set data yang sangat menarik: _Penampakan UFO selama abad yang lalu_, yang bersumber dari basis data NUFORC.

Anda akan belajar:

- Cara 'pickle' model yang telah dilatih
- Cara menggunakan model tersebut dalam aplikasi Flask

Kita akan melanjutkan penggunaan notebook untuk membersihkan data dan melatih model kita, tetapi anda dapat melangkah lebih jauh dengan mengeksplorasi penggunaan model 'di lapangan': dalam aplikasi web.

Untuk melakukan ini, anda perlu membangun aplikasi web menggunakan Flask.

## [Kuis Pra-Kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Membangun Aplikasi

Ada beberapa cara untuk membangun aplikasi web yang mengonsumsi model pembelajaran mesin. Arsitektur web anda mungkin mempengaruhi cara model anda dilatih. Bayangkan anda bekerja di sebuah perusahaan di mana kelompok ilmu data telah melatih model yang mereka ingin anda gunakan dalam aplikasi.

### Pertimbangan

Ada banyak pertanyaan yang perlu anda tanyakan:

- **Apakah itu aplikasi web atau aplikasi seluler?** Jika anda membangun aplikasi seluler atau perlu menggunakan model dalam konteks IoT, anda dapat menggunakan [TensorFlow Lite](https://www.tensorflow.org/lite/) dan menggunakan model dalam aplikasi Android atau iOS.
- **Di mana model akan berada?** Di cloud atau lokal?
- **Dukungan offline.** Apakah aplikasi harus berfungsi secara offline?
- **Teknologi apa yang digunakan untuk melatih model?** Teknologi yang dipilih mungkin mempengaruhi alat yang perlu anda gunakan.
    - **Menggunakan TensorFlow.** Jika anda melatih model menggunakan TensorFlow, misalnya, ekosistem tersebut menyediakan kemampuan untuk mengkonversi model TensorFlow untuk digunakan dalam aplikasi web dengan menggunakan [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Menggunakan PyTorch.** Jika anda membangun model menggunakan perpustakaan seperti [PyTorch](https://pytorch.org/), anda memiliki opsi untuk mengekspornya dalam format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) untuk digunakan dalam aplikasi web JavaScript yang dapat menggunakan [Onnx Runtime](https://www.onnxruntime.ai/). Opsi ini akan dieksplorasi dalam pelajaran mendatang untuk model yang dilatih dengan Scikit-learn.
    - **Menggunakan Lobe.ai atau Azure Custom Vision.** Jika anda menggunakan sistem ML SaaS (Software as a Service) seperti [Lobe.ai](https://lobe.ai/) atau [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) untuk melatih model, jenis perangkat lunak ini menyediakan cara untuk mengekspor model untuk banyak platform, termasuk membangun API khusus untuk di-query di cloud oleh aplikasi online anda.

Anda juga memiliki kesempatan untuk membangun seluruh aplikasi web Flask yang dapat melatih model itu sendiri dalam browser web. Ini juga dapat dilakukan menggunakan TensorFlow.js dalam konteks JavaScript.

Untuk tujuan kita, karena kita telah bekerja dengan notebook berbasis Python, mari kita eksplorasi langkah-langkah yang perlu diambil untuk mengekspor model yang telah dilatih dari notebook tersebut ke format yang dapat dibaca oleh aplikasi web yang dibangun dengan Python.

## Alat

Untuk tugas ini, anda memerlukan dua alat: Flask dan Pickle, keduanya berjalan di Python.

✅ Apa itu [Flask](https://palletsprojects.com/p/flask/)? Didefinisikan sebagai 'micro-framework' oleh penciptanya, Flask menyediakan fitur dasar kerangka kerja web menggunakan Python dan mesin templat untuk membangun halaman web. Lihat [modul Belajar ini](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) untuk berlatih membangun dengan Flask.

✅ Apa itu [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 adalah modul Python yang men-serialisasi dan de-serialisasi struktur objek Python. Ketika anda 'pickle' model, anda men-serialisasi atau meratakan strukturnya untuk digunakan di web. Hati-hati: pickle tidak secara intrinsik aman, jadi berhati-hatilah jika diminta untuk 'un-pickle' file. File yang di-pickle memiliki akhiran `.pkl`.

## Latihan - membersihkan data anda

Dalam pelajaran ini anda akan menggunakan data dari 80.000 penampakan UFO, dikumpulkan oleh [NUFORC](https://nuforc.org) (Pusat Pelaporan UFO Nasional). Data ini memiliki beberapa deskripsi menarik tentang penampakan UFO, misalnya:

- **Deskripsi contoh panjang.** "Seorang pria muncul dari sinar cahaya yang menyinari lapangan berumput di malam hari dan dia berlari menuju tempat parkir Texas Instruments".
- **Deskripsi contoh pendek.** "lampu-lampu mengejar kami".

Spreadsheet [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) mencakup kolom tentang `city`, `state`, dan `country` di mana penampakan terjadi, objek `shape` dan `latitude` serta `longitude`.

Dalam [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) kosong yang disertakan dalam pelajaran ini:

1. impor `pandas`, `matplotlib`, dan `numpy` seperti yang anda lakukan dalam pelajaran sebelumnya dan impor spreadsheet ufos. Anda dapat melihat sampel set data:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konversikan data ufos menjadi dataframe kecil dengan judul baru. Periksa nilai unik di bidang `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Sekarang, anda dapat mengurangi jumlah data yang perlu kita tangani dengan menghapus nilai null dan hanya mengimpor penampakan antara 1-60 detik:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Impor perpustakaan `LabelEncoder` dari Scikit-learn untuk mengonversi nilai teks untuk negara menjadi angka:

    ✅ LabelEncoder mengkodekan data secara alfabetis

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Data anda harus terlihat seperti ini:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Latihan - membangun model anda

Sekarang anda dapat bersiap untuk melatih model dengan membagi data menjadi kelompok pelatihan dan pengujian.

1. Pilih tiga fitur yang ingin anda latih sebagai vektor X anda, dan vektor y akan menjadi `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` dan mendapatkan id negara untuk dikembalikan.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Latih model anda menggunakan regresi logistik:

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

Akurasi tidak buruk **(sekitar 95%)**, tidak mengherankan, karena `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, tetapi ini adalah latihan yang baik untuk mencoba melatih dari data mentah yang anda bersihkan, diekspor, dan kemudian menggunakan model ini dalam aplikasi web.

## Latihan - 'pickle' model anda

Sekarang, saatnya untuk _pickle_ model anda! Anda dapat melakukannya dalam beberapa baris kode. Setelah itu di-_pickle_, muat model yang di-pickle dan uji terhadap array data sampel yang berisi nilai untuk detik, lintang, dan bujur,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model mengembalikan **'3'**, yang merupakan kode negara untuk Inggris. Luar biasa! 👽

## Latihan - membangun aplikasi Flask

Sekarang anda dapat membangun aplikasi Flask untuk memanggil model anda dan mengembalikan hasil serupa, tetapi dengan cara yang lebih menarik secara visual.

1. Mulailah dengan membuat folder bernama **web-app** di sebelah file _notebook.ipynb_ tempat file _ufo-model.pkl_ anda berada.

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

1. File pertama yang dibuat dalam folder _web-app_ adalah file **requirements.txt**. Seperti _package.json_ dalam aplikasi JavaScript, file ini mencantumkan ketergantungan yang diperlukan oleh aplikasi. Dalam **requirements.txt** tambahkan baris:

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

1. Di terminal anda ketik `pip install`, untuk menginstal perpustakaan yang tercantum dalam _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Sekarang, anda siap membuat tiga file lagi untuk menyelesaikan aplikasi:

    1. Buat **app.py** di root.
    2. Buat **index.html** di direktori _templates_.
    3. Buat **styles.css** di direktori _static/css_.

1. Buat file _styles.css_ dengan beberapa gaya:

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

1. Selanjutnya, buat file _index.html_:

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

    Lihatlah templating dalam file ini. Perhatikan sintaks 'mustache' di sekitar variabel yang akan disediakan oleh aplikasi, seperti teks prediksi: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` tambahkan:

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

    > 💡 Tip: ketika anda menambahkan [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. Apa pro dan kontra dari mengejar metode ini?

## [Kuis Pasca-Kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Tinjauan & Studi Mandiri

Ada banyak cara untuk membangun aplikasi web yang mengonsumsi model ML. Buatlah daftar cara-cara yang dapat anda gunakan JavaScript atau Python untuk membangun aplikasi web yang memanfaatkan pembelajaran mesin. Pertimbangkan arsitektur: apakah model harus tetap dalam aplikasi atau hidup di cloud? Jika yang terakhir, bagaimana anda mengaksesnya? Gambarlah model arsitektur untuk solusi web ML yang diterapkan.

## Tugas

[Cobalah model yang berbeda](assignment.md)

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat penting, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.