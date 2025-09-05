<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T19:54:57+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "ms"
}
-->
# Membina Aplikasi Web Pencadang Masakan

Dalam pelajaran ini, anda akan membina model klasifikasi menggunakan beberapa teknik yang telah dipelajari dalam pelajaran sebelumnya dan dataset masakan yang lazat yang digunakan sepanjang siri ini. Selain itu, anda akan membina aplikasi web kecil untuk menggunakan model yang disimpan, dengan memanfaatkan runtime web Onnx.

Salah satu kegunaan praktikal pembelajaran mesin yang paling berguna ialah membina sistem cadangan, dan anda boleh mengambil langkah pertama ke arah itu hari ini!

[![Mempersembahkan aplikasi web ini](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Klik imej di atas untuk video: Jen Looper membina aplikasi web menggunakan data masakan yang telah diklasifikasikan

## [Kuiz pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

Dalam pelajaran ini, anda akan belajar:

- Cara membina model dan menyimpannya sebagai model Onnx
- Cara menggunakan Netron untuk memeriksa model
- Cara menggunakan model anda dalam aplikasi web untuk inferens

## Bina model anda

Membina sistem ML yang diterapkan adalah bahagian penting dalam memanfaatkan teknologi ini untuk sistem perniagaan anda. Anda boleh menggunakan model dalam aplikasi web anda (dan dengan itu menggunakannya dalam konteks luar talian jika diperlukan) dengan menggunakan Onnx.

Dalam [pelajaran sebelumnya](../../3-Web-App/1-Web-App/README.md), anda telah membina model Regresi tentang penampakan UFO, "pickled" model tersebut, dan menggunakannya dalam aplikasi Flask. Walaupun seni bina ini sangat berguna untuk diketahui, ia adalah aplikasi Python full-stack, dan keperluan anda mungkin termasuk penggunaan aplikasi JavaScript.

Dalam pelajaran ini, anda boleh membina sistem asas berasaskan JavaScript untuk inferens. Namun, pertama sekali, anda perlu melatih model dan menukarkannya untuk digunakan dengan Onnx.

## Latihan - latih model klasifikasi

Pertama, latih model klasifikasi menggunakan dataset masakan yang telah dibersihkan yang kita gunakan.

1. Mulakan dengan mengimport pustaka yang berguna:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Anda memerlukan '[skl2onnx](https://onnx.ai/sklearn-onnx/)' untuk membantu menukar model Scikit-learn anda ke format Onnx.

1. Kemudian, bekerja dengan data anda dengan cara yang sama seperti yang anda lakukan dalam pelajaran sebelumnya, dengan membaca fail CSV menggunakan `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Buang dua lajur pertama yang tidak diperlukan dan simpan data yang tinggal sebagai 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Simpan label sebagai 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Mulakan rutin latihan

Kami akan menggunakan pustaka 'SVC' yang mempunyai ketepatan yang baik.

1. Import pustaka yang sesuai dari Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Pisahkan set latihan dan ujian:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Bina model Klasifikasi SVC seperti yang anda lakukan dalam pelajaran sebelumnya:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Sekarang, uji model anda dengan memanggil `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Cetak laporan klasifikasi untuk memeriksa kualiti model:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Seperti yang kita lihat sebelum ini, ketepatannya adalah baik:

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

### Tukar model anda ke Onnx

Pastikan untuk melakukan penukaran dengan nombor Tensor yang betul. Dataset ini mempunyai 380 bahan yang disenaraikan, jadi anda perlu mencatatkan nombor itu dalam `FloatTensorType`:

1. Tukar menggunakan nombor tensor sebanyak 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Cipta fail onx dan simpan sebagai **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Nota, anda boleh memasukkan [pilihan](https://onnx.ai/sklearn-onnx/parameterized.html) dalam skrip penukaran anda. Dalam kes ini, kami memasukkan 'nocl' sebagai True dan 'zipmap' sebagai False. Oleh kerana ini adalah model klasifikasi, anda mempunyai pilihan untuk membuang ZipMap yang menghasilkan senarai kamus (tidak diperlukan). `nocl` merujuk kepada maklumat kelas yang disertakan dalam model. Kurangkan saiz model anda dengan menetapkan `nocl` kepada 'True'.

Menjalankan keseluruhan notebook kini akan membina model Onnx dan menyimpannya ke folder ini.

## Lihat model anda

Model Onnx tidak begitu kelihatan dalam Visual Studio Code, tetapi terdapat perisian percuma yang sangat baik yang digunakan oleh ramai penyelidik untuk memvisualisasikan model bagi memastikan ia dibina dengan betul. Muat turun [Netron](https://github.com/lutzroeder/Netron) dan buka fail model.onnx anda. Anda boleh melihat model ringkas anda divisualisasikan, dengan 380 input dan pengklasifikasi disenaraikan:

![Visual Netron](../../../../4-Classification/4-Applied/images/netron.png)

Netron adalah alat yang berguna untuk melihat model anda.

Kini anda bersedia untuk menggunakan model yang menarik ini dalam aplikasi web. Mari kita bina aplikasi yang akan berguna apabila anda melihat ke dalam peti sejuk anda dan cuba menentukan kombinasi bahan lebihan yang boleh digunakan untuk memasak masakan tertentu, seperti yang ditentukan oleh model anda.

## Bina aplikasi web pencadang

Anda boleh menggunakan model anda secara langsung dalam aplikasi web. Seni bina ini juga membolehkan anda menjalankannya secara tempatan dan bahkan di luar talian jika diperlukan. Mulakan dengan mencipta fail `index.html` dalam folder yang sama di mana anda menyimpan fail `model.onnx` anda.

1. Dalam fail ini _index.html_, tambahkan markup berikut:

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

1. Sekarang, bekerja dalam tag `body`, tambahkan sedikit markup untuk menunjukkan senarai kotak semak yang mencerminkan beberapa bahan:

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

    Perhatikan bahawa setiap kotak semak diberikan nilai. Ini mencerminkan indeks di mana bahan tersebut ditemui mengikut dataset. Sebagai contoh, epal dalam senarai abjad ini, menduduki lajur kelima, jadi nilainya adalah '4' kerana kita mula mengira dari 0. Anda boleh merujuk kepada [lembaran kerja bahan](../../../../4-Classification/data/ingredient_indexes.csv) untuk mengetahui indeks bahan tertentu.

    Meneruskan kerja anda dalam fail index.html, tambahkan blok skrip di mana model dipanggil selepas penutupan akhir `</div>`.

1. Pertama, import [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime digunakan untuk membolehkan menjalankan model Onnx anda merentasi pelbagai platform perkakasan, termasuk pengoptimuman dan API untuk digunakan.

1. Setelah Runtime tersedia, anda boleh memanggilnya:

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

Dalam kod ini, beberapa perkara berlaku:

1. Anda mencipta array dengan 380 nilai yang mungkin (1 atau 0) untuk ditetapkan dan dihantar ke model untuk inferens, bergantung pada sama ada kotak semak bahan ditandakan.
2. Anda mencipta array kotak semak dan cara untuk menentukan sama ada ia ditandakan dalam fungsi `init` yang dipanggil apabila aplikasi dimulakan. Apabila kotak semak ditandakan, array `ingredients` diubah untuk mencerminkan bahan yang dipilih.
3. Anda mencipta fungsi `testCheckboxes` yang memeriksa sama ada mana-mana kotak semak ditandakan.
4. Anda menggunakan fungsi `startInference` apabila butang ditekan dan, jika mana-mana kotak semak ditandakan, anda memulakan inferens.
5. Rutin inferens termasuk:
   1. Menyediakan muatan asinkron model
   2. Mencipta struktur Tensor untuk dihantar ke model
   3. Mencipta 'feeds' yang mencerminkan input `float_input` yang anda cipta semasa melatih model anda (anda boleh menggunakan Netron untuk mengesahkan nama itu)
   4. Menghantar 'feeds' ini ke model dan menunggu respons

## Uji aplikasi anda

Buka sesi terminal dalam Visual Studio Code dalam folder di mana fail index.html anda berada. Pastikan anda mempunyai [http-server](https://www.npmjs.com/package/http-server) dipasang secara global, dan taip `http-server` pada prompt. Satu localhost akan dibuka dan anda boleh melihat aplikasi web anda. Periksa masakan yang disyorkan berdasarkan pelbagai bahan:

![Aplikasi web bahan](../../../../4-Classification/4-Applied/images/web-app.png)

Tahniah, anda telah mencipta aplikasi web 'cadangan' dengan beberapa medan. Luangkan masa untuk membina sistem ini!

## 🚀Cabaran

Aplikasi web anda sangat minimal, jadi teruskan membangunkannya menggunakan bahan dan indeksnya daripada data [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Kombinasi rasa apa yang berfungsi untuk mencipta hidangan kebangsaan tertentu?

## [Kuiz pasca-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Kajian & Pembelajaran Kendiri

Walaupun pelajaran ini hanya menyentuh tentang kegunaan mencipta sistem cadangan untuk bahan makanan, bidang aplikasi ML ini sangat kaya dengan contoh. Baca lebih lanjut tentang bagaimana sistem ini dibina:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tugasan 

[Bina pencadang baru](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.