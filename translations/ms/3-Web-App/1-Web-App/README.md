<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T18:06:44+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ms"
}
-->
# Membina Aplikasi Web untuk Menggunakan Model ML

Dalam pelajaran ini, anda akan melatih model ML menggunakan set data yang luar biasa: _Penampakan UFO sepanjang abad yang lalu_, yang diperoleh daripada pangkalan data NUFORC.

Anda akan belajar:

- Cara 'pickle' model yang telah dilatih
- Cara menggunakan model tersebut dalam aplikasi Flask

Kita akan terus menggunakan notebook untuk membersihkan data dan melatih model, tetapi anda boleh melangkah lebih jauh dengan meneroka penggunaan model 'di dunia nyata', iaitu dalam aplikasi web.

Untuk melakukannya, anda perlu membina aplikasi web menggunakan Flask.

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Membina aplikasi

Terdapat beberapa cara untuk membina aplikasi web yang menggunakan model pembelajaran mesin. Seni bina web anda mungkin mempengaruhi cara model anda dilatih. Bayangkan anda bekerja dalam perniagaan di mana kumpulan sains data telah melatih model yang mereka mahu anda gunakan dalam aplikasi.

### Pertimbangan

Terdapat banyak soalan yang perlu anda tanya:

- **Adakah ia aplikasi web atau aplikasi mudah alih?** Jika anda membina aplikasi mudah alih atau perlu menggunakan model dalam konteks IoT, anda boleh menggunakan [TensorFlow Lite](https://www.tensorflow.org/lite/) dan menggunakan model tersebut dalam aplikasi Android atau iOS.
- **Di mana model akan berada?** Di awan atau secara tempatan?
- **Sokongan luar talian.** Adakah aplikasi perlu berfungsi secara luar talian?
- **Teknologi apa yang digunakan untuk melatih model?** Teknologi yang dipilih mungkin mempengaruhi alat yang perlu anda gunakan.
    - **Menggunakan TensorFlow.** Jika anda melatih model menggunakan TensorFlow, contohnya, ekosistem tersebut menyediakan keupayaan untuk menukar model TensorFlow untuk digunakan dalam aplikasi web dengan menggunakan [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Menggunakan PyTorch.** Jika anda membina model menggunakan pustaka seperti [PyTorch](https://pytorch.org/), anda mempunyai pilihan untuk mengeksportnya dalam format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) untuk digunakan dalam aplikasi web JavaScript yang boleh menggunakan [Onnx Runtime](https://www.onnxruntime.ai/). Pilihan ini akan diteroka dalam pelajaran akan datang untuk model yang dilatih menggunakan Scikit-learn.
    - **Menggunakan Lobe.ai atau Azure Custom Vision.** Jika anda menggunakan sistem ML SaaS (Perisian sebagai Perkhidmatan) seperti [Lobe.ai](https://lobe.ai/) atau [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) untuk melatih model, jenis perisian ini menyediakan cara untuk mengeksport model untuk pelbagai platform, termasuk membina API tersuai untuk ditanya di awan oleh aplikasi dalam talian anda.

Anda juga berpeluang untuk membina keseluruhan aplikasi web Flask yang boleh melatih model itu sendiri dalam pelayar web. Ini juga boleh dilakukan menggunakan TensorFlow.js dalam konteks JavaScript.

Untuk tujuan kita, memandangkan kita telah bekerja dengan notebook berasaskan Python, mari kita teroka langkah-langkah yang perlu diambil untuk mengeksport model yang telah dilatih daripada notebook tersebut ke format yang boleh dibaca oleh aplikasi web yang dibina menggunakan Python.

## Alat

Untuk tugas ini, anda memerlukan dua alat: Flask dan Pickle, kedua-duanya berjalan pada Python.

✅ Apa itu [Flask](https://palletsprojects.com/p/flask/)? Didefinisikan sebagai 'micro-framework' oleh penciptanya, Flask menyediakan ciri asas rangka kerja web menggunakan Python dan enjin templat untuk membina halaman web. Lihat [modul pembelajaran ini](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) untuk berlatih membina dengan Flask.

✅ Apa itu [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 ialah modul Python yang menyusun dan menyahsusun struktur objek Python. Apabila anda 'pickle' model, anda menyusun atau meratakan strukturnya untuk digunakan di web. Berhati-hati: pickle tidak secara intrinsik selamat, jadi berhati-hati jika diminta untuk 'un-pickle' fail. Fail yang telah dipickle mempunyai akhiran `.pkl`.

## Latihan - membersihkan data anda

Dalam pelajaran ini, anda akan menggunakan data daripada 80,000 penampakan UFO, yang dikumpulkan oleh [NUFORC](https://nuforc.org) (Pusat Pelaporan UFO Kebangsaan). Data ini mempunyai beberapa deskripsi menarik tentang penampakan UFO, contohnya:

- **Deskripsi contoh panjang.** "Seorang lelaki muncul dari pancaran cahaya yang bersinar di padang rumput pada waktu malam dan dia berlari ke arah tempat letak kereta Texas Instruments".
- **Deskripsi contoh pendek.** "lampu mengejar kami".

Spreadsheet [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) termasuk lajur tentang `city`, `state` dan `country` di mana penampakan berlaku, `shape` objek dan `latitude` serta `longitude`nya.

Dalam [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) kosong yang disertakan dalam pelajaran ini:

1. import `pandas`, `matplotlib`, dan `numpy` seperti yang anda lakukan dalam pelajaran sebelumnya dan import spreadsheet ufos. Anda boleh melihat sampel set data:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Tukarkan data ufos kepada dataframe kecil dengan tajuk baru. Periksa nilai unik dalam medan `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Kini, anda boleh mengurangkan jumlah data yang perlu kita uruskan dengan membuang nilai null dan hanya mengimport penampakan antara 1-60 saat:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Import pustaka `LabelEncoder` dari Scikit-learn untuk menukar nilai teks bagi negara kepada nombor:

    ✅ LabelEncoder menyandikan data mengikut abjad

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Data anda sepatutnya kelihatan seperti ini:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Latihan - membina model anda

Kini anda boleh bersedia untuk melatih model dengan membahagikan data kepada kumpulan latihan dan ujian.

1. Pilih tiga ciri yang ingin anda latih sebagai vektor X anda, dan vektor y akan menjadi `Country`. Anda ingin dapat memasukkan `Seconds`, `Latitude` dan `Longitude` dan mendapatkan id negara untuk dikembalikan.

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

Ketepatannya tidak buruk **(sekitar 95%)**, tidak mengejutkan, kerana `Country` dan `Latitude/Longitude` berkorelasi.

Model yang anda cipta tidak begitu revolusioner kerana anda sepatutnya dapat menyimpulkan `Country` daripada `Latitude` dan `Longitude`nya, tetapi ia adalah latihan yang baik untuk mencuba melatih daripada data mentah yang anda bersihkan, eksport, dan kemudian menggunakan model ini dalam aplikasi web.

## Latihan - 'pickle' model anda

Kini, tiba masanya untuk _pickle_ model anda! Anda boleh melakukannya dalam beberapa baris kod. Setelah ia _dipickle_, muatkan model yang dipickle dan uji terhadap array data sampel yang mengandungi nilai untuk seconds, latitude dan longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model mengembalikan **'3'**, yang merupakan kod negara untuk UK. Hebat! 👽

## Latihan - membina aplikasi Flask

Kini anda boleh membina aplikasi Flask untuk memanggil model anda dan mengembalikan hasil yang serupa, tetapi dengan cara yang lebih menarik secara visual.

1. Mulakan dengan mencipta folder bernama **web-app** di sebelah fail _notebook.ipynb_ di mana fail _ufo-model.pkl_ anda berada.

1. Dalam folder itu, cipta tiga lagi folder: **static**, dengan folder **css** di dalamnya, dan **templates**. Anda sepatutnya kini mempunyai fail dan direktori berikut:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Rujuk folder penyelesaian untuk melihat aplikasi yang telah siap

1. Fail pertama yang perlu dicipta dalam folder _web-app_ ialah fail **requirements.txt**. Seperti _package.json_ dalam aplikasi JavaScript, fail ini menyenaraikan kebergantungan yang diperlukan oleh aplikasi. Dalam **requirements.txt** tambahkan baris:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Kini, jalankan fail ini dengan menavigasi ke _web-app_:

    ```bash
    cd web-app
    ```

1. Dalam terminal anda taip `pip install`, untuk memasang pustaka yang disenaraikan dalam _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Kini, anda bersedia untuk mencipta tiga lagi fail untuk melengkapkan aplikasi:

    1. Cipta **app.py** di root.
    2. Cipta **index.html** dalam direktori _templates_.
    3. Cipta **styles.css** dalam direktori _static/css_.

1. Bina fail _styles.css_ dengan beberapa gaya:

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

1. Seterusnya, bina fail _index.html_:

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

    Perhatikan templat dalam fail ini. Perhatikan sintaks 'mustache' di sekitar pembolehubah yang akan disediakan oleh aplikasi, seperti teks ramalan: `{{}}`. Terdapat juga borang yang menghantar ramalan ke laluan `/predict`.

    Akhirnya, anda bersedia untuk membina fail python yang memacu penggunaan model dan paparan ramalan:

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

    > 💡 Tip: apabila anda menambah [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) semasa menjalankan aplikasi web menggunakan Flask, sebarang perubahan yang anda buat pada aplikasi anda akan tercermin serta-merta tanpa perlu memulakan semula pelayan. Berhati-hati! Jangan aktifkan mod ini dalam aplikasi pengeluaran.

Jika anda menjalankan `python app.py` atau `python3 app.py` - pelayan web anda akan bermula secara tempatan, dan anda boleh mengisi borang pendek untuk mendapatkan jawapan kepada soalan anda tentang di mana UFO telah dilihat!

Sebelum melakukannya, lihat bahagian `app.py`:

1. Pertama, kebergantungan dimuatkan dan aplikasi dimulakan.
1. Kemudian, model diimport.
1. Kemudian, index.html dirender pada laluan utama.

Pada laluan `/predict`, beberapa perkara berlaku apabila borang dihantar:

1. Pembolehubah borang dikumpulkan dan ditukar kepada array numpy. Ia kemudian dihantar ke model dan ramalan dikembalikan.
2. Negara-negara yang ingin dipaparkan dirender semula sebagai teks yang boleh dibaca daripada kod negara yang diramalkan, dan nilai tersebut dihantar kembali ke index.html untuk dirender dalam templat.

Menggunakan model dengan cara ini, dengan Flask dan model yang dipickle, adalah agak mudah. Perkara yang paling sukar ialah memahami bentuk data yang mesti dihantar ke model untuk mendapatkan ramalan. Itu semua bergantung pada bagaimana model dilatih. Model ini mempunyai tiga titik data untuk dimasukkan bagi mendapatkan ramalan.

Dalam suasana profesional, anda dapat melihat betapa pentingnya komunikasi yang baik antara mereka yang melatih model dan mereka yang menggunakannya dalam aplikasi web atau mudah alih. Dalam kes kita, ia hanya satu orang, anda!

---

## 🚀 Cabaran

Daripada bekerja dalam notebook dan mengimport model ke aplikasi Flask, anda boleh melatih model terus dalam aplikasi Flask! Cuba tukarkan kod Python anda dalam notebook, mungkin selepas data anda dibersihkan, untuk melatih model dari dalam aplikasi pada laluan yang dipanggil `train`. Apakah kelebihan dan kekurangan menggunakan kaedah ini?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Terdapat banyak cara untuk membina aplikasi web yang menggunakan model ML. Buat senarai cara anda boleh menggunakan JavaScript atau Python untuk membina aplikasi web yang memanfaatkan pembelajaran mesin. Pertimbangkan seni bina: adakah model harus kekal dalam aplikasi atau berada di awan? Jika yang terakhir, bagaimana anda akan mengaksesnya? Lukiskan model seni bina untuk penyelesaian web ML yang diterapkan.

## Tugasan

[Cuba model yang berbeza](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.