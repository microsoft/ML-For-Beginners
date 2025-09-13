<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T18:41:52+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "ms"
}
-->
# Bina model regresi menggunakan Scikit-learn: empat cara regresi

![Infografik regresi linear vs polinomial](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kuiz pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Pengenalan 

Setakat ini, anda telah meneroka apa itu regresi dengan data sampel yang dikumpulkan daripada dataset harga labu yang akan kita gunakan sepanjang pelajaran ini. Anda juga telah memvisualisasikannya menggunakan Matplotlib.

Kini anda bersedia untuk mendalami regresi untuk ML. Walaupun visualisasi membolehkan anda memahami data, kekuatan sebenar Pembelajaran Mesin datang daripada _melatih model_. Model dilatih menggunakan data sejarah untuk secara automatik menangkap pergantungan data, dan ia membolehkan anda meramalkan hasil untuk data baharu yang belum pernah dilihat oleh model.

Dalam pelajaran ini, anda akan mempelajari lebih lanjut tentang dua jenis regresi: _regresi linear asas_ dan _regresi polinomial_, bersama-sama dengan beberapa matematik yang mendasari teknik ini. Model-model ini akan membolehkan kita meramalkan harga labu bergantung pada data input yang berbeza.

[![ML untuk pemula - Memahami Regresi Linear](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML untuk pemula - Memahami Regresi Linear")

> 🎥 Klik imej di atas untuk video ringkas mengenai regresi linear.

> Sepanjang kurikulum ini, kami mengandaikan pengetahuan matematik yang minimum, dan berusaha untuk menjadikannya mudah diakses oleh pelajar dari bidang lain, jadi perhatikan nota, 🧮 panggilan, diagram, dan alat pembelajaran lain untuk membantu pemahaman.

### Prasyarat

Anda seharusnya sudah biasa dengan struktur data labu yang sedang kita kaji. Anda boleh menemukannya telah dimuatkan dan dibersihkan dalam fail _notebook.ipynb_ pelajaran ini. Dalam fail tersebut, harga labu dipaparkan per gantang dalam bingkai data baharu. Pastikan anda boleh menjalankan notebook ini dalam kernel di Visual Studio Code.

### Persediaan

Sebagai peringatan, anda memuatkan data ini untuk bertanya soalan mengenainya.

- Bilakah masa terbaik untuk membeli labu?
- Berapakah harga yang boleh saya jangkakan untuk satu kotak labu mini?
- Haruskah saya membelinya dalam bakul setengah gantang atau dalam kotak 1 1/9 gantang?
Mari kita teruskan menyelidiki data ini.

Dalam pelajaran sebelumnya, anda telah mencipta bingkai data Pandas dan mengisinya dengan sebahagian daripada dataset asal, menstandardkan harga mengikut gantang. Dengan melakukan itu, bagaimanapun, anda hanya dapat mengumpulkan kira-kira 400 titik data dan hanya untuk bulan-bulan musim luruh.

Lihat data yang telah dimuatkan dalam notebook yang disertakan dengan pelajaran ini. Data telah dimuatkan dan plot penyebaran awal telah dibuat untuk menunjukkan data bulan. Mungkin kita boleh mendapatkan sedikit lebih banyak perincian tentang sifat data dengan membersihkannya lebih lanjut.

## Garis regresi linear

Seperti yang anda pelajari dalam Pelajaran 1, tujuan latihan regresi linear adalah untuk dapat memplot garis untuk:

- **Menunjukkan hubungan pemboleh ubah**. Menunjukkan hubungan antara pemboleh ubah
- **Membuat ramalan**. Membuat ramalan yang tepat tentang di mana titik data baharu akan berada dalam hubungan dengan garis tersebut.

Adalah biasa bagi **Regresi Kuadrat Terkecil** untuk melukis jenis garis ini. Istilah 'kuadrat terkecil' bermaksud bahawa semua titik data di sekitar garis regresi dikwadratkan dan kemudian dijumlahkan. Sebaiknya, jumlah akhir itu sekecil mungkin, kerana kita mahukan bilangan kesalahan yang rendah, atau `kuadrat terkecil`.

Kita melakukan ini kerana kita ingin memodelkan garis yang mempunyai jarak kumulatif paling kecil dari semua titik data kita. Kita juga mengkwadratkan istilah sebelum menjumlahkannya kerana kita lebih peduli dengan magnitudnya daripada arahnya.

> **🧮 Tunjukkan matematiknya** 
> 
> Garis ini, yang disebut _garis terbaik_, dapat dinyatakan dengan [persamaan](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` adalah 'pemboleh ubah penjelasan'. `Y` adalah 'pemboleh ubah bergantung'. Kecerunan garis adalah `b` dan `a` adalah pemintas-y, yang merujuk kepada nilai `Y` apabila `X = 0`. 
>
>![mengira kecerunan](../../../../2-Regression/3-Linear/images/slope.png)
>
> Pertama, kira kecerunan `b`. Infografik oleh [Jen Looper](https://twitter.com/jenlooper)
>
> Dalam kata lain, dan merujuk kepada soalan asal data labu kita: "ramalkan harga labu per gantang mengikut bulan", `X` akan merujuk kepada harga dan `Y` akan merujuk kepada bulan jualan. 
>
>![lengkapkan persamaan](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Kira nilai Y. Jika anda membayar sekitar $4, itu mesti bulan April! Infografik oleh [Jen Looper](https://twitter.com/jenlooper)
>
> Matematik yang mengira garis mesti menunjukkan kecerunan garis, yang juga bergantung pada pemintas, atau di mana `Y` terletak apabila `X = 0`.
>
> Anda boleh melihat kaedah pengiraan untuk nilai-nilai ini di laman web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Juga lawati [kalkulator Kuadrat Terkecil](https://www.mathsisfun.com/data/least-squares-calculator.html) untuk melihat bagaimana nilai-nilai angka mempengaruhi garis.

## Korelasi

Satu lagi istilah yang perlu difahami ialah **Pekali Korelasi** antara pemboleh ubah X dan Y yang diberikan. Menggunakan plot penyebaran, anda boleh dengan cepat memvisualisasikan pekali ini. Plot dengan titik data yang tersebar dalam garis rapi mempunyai korelasi tinggi, tetapi plot dengan titik data yang tersebar di mana-mana antara X dan Y mempunyai korelasi rendah.

Model regresi linear yang baik adalah model yang mempunyai Pekali Korelasi tinggi (lebih dekat kepada 1 daripada 0) menggunakan kaedah Regresi Kuadrat Terkecil dengan garis regresi.

✅ Jalankan notebook yang disertakan dengan pelajaran ini dan lihat plot penyebaran Bulan ke Harga. Adakah data yang mengaitkan Bulan ke Harga untuk jualan labu kelihatan mempunyai korelasi tinggi atau rendah, menurut tafsiran visual anda terhadap plot penyebaran? Adakah itu berubah jika anda menggunakan ukuran yang lebih terperinci daripada `Bulan`, contohnya *hari dalam tahun* (iaitu bilangan hari sejak awal tahun)?

Dalam kod di bawah, kita akan mengandaikan bahawa kita telah membersihkan data, dan memperoleh bingkai data yang disebut `new_pumpkins`, serupa dengan yang berikut:

ID | Bulan | HariDalamTahun | Jenis | Bandar | Pakej | Harga Rendah | Harga Tinggi | Harga
---|-------|----------------|-------|--------|-------|--------------|--------------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Kod untuk membersihkan data tersedia dalam [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Kami telah melakukan langkah pembersihan yang sama seperti dalam pelajaran sebelumnya, dan telah mengira lajur `HariDalamTahun` menggunakan ekspresi berikut: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sekarang setelah anda memahami matematik di sebalik regresi linear, mari kita buat model Regresi untuk melihat sama ada kita boleh meramalkan pakej labu mana yang akan mempunyai harga labu terbaik. Seseorang yang membeli labu untuk ladang labu percutian mungkin mahukan maklumat ini untuk mengoptimumkan pembelian pakej labu untuk ladang tersebut.

## Mencari Korelasi

[![ML untuk pemula - Mencari Korelasi: Kunci kepada Regresi Linear](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML untuk pemula - Mencari Korelasi: Kunci kepada Regresi Linear")

> 🎥 Klik imej di atas untuk video ringkas mengenai korelasi.

Daripada pelajaran sebelumnya, anda mungkin telah melihat bahawa harga purata untuk bulan-bulan yang berbeza kelihatan seperti ini:

<img alt="Harga purata mengikut bulan" src="../2-Data/images/barchart.png" width="50%"/>

Ini menunjukkan bahawa mungkin terdapat beberapa korelasi, dan kita boleh mencuba melatih model regresi linear untuk meramalkan hubungan antara `Bulan` dan `Harga`, atau antara `HariDalamTahun` dan `Harga`. Berikut adalah plot penyebaran yang menunjukkan hubungan yang terakhir:

<img alt="Plot penyebaran Harga vs. Hari dalam Tahun" src="images/scatter-dayofyear.png" width="50%" /> 

Mari kita lihat sama ada terdapat korelasi menggunakan fungsi `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Nampaknya korelasi agak kecil, -0.15 mengikut `Bulan` dan -0.17 mengikut `HariDalamTahun`, tetapi mungkin terdapat hubungan penting yang lain. Nampaknya terdapat kelompok harga yang berbeza yang sesuai dengan jenis labu yang berbeza. Untuk mengesahkan hipotesis ini, mari kita plot setiap kategori labu menggunakan warna yang berbeza. Dengan memberikan parameter `ax` kepada fungsi plot `scatter`, kita boleh memplot semua titik pada graf yang sama:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Plot penyebaran Harga vs. Hari dalam Tahun" src="images/scatter-dayofyear-color.png" width="50%" /> 

Penyelidikan kita menunjukkan bahawa jenis labu mempunyai lebih banyak kesan terhadap harga keseluruhan daripada tarikh jualan sebenar. Kita boleh melihat ini dengan graf bar:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Graf bar harga vs jenis labu" src="images/price-by-variety.png" width="50%" /> 

Mari kita fokus buat sementara waktu hanya pada satu jenis labu, 'pie type', dan lihat kesan tarikh terhadap harga:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Plot penyebaran Harga vs. Hari dalam Tahun" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Jika kita kini mengira korelasi antara `Harga` dan `HariDalamTahun` menggunakan fungsi `corr`, kita akan mendapat sesuatu seperti `-0.27` - yang bermaksud bahawa melatih model ramalan masuk akal.

> Sebelum melatih model regresi linear, adalah penting untuk memastikan data kita bersih. Regresi linear tidak berfungsi dengan baik dengan nilai yang hilang, oleh itu masuk akal untuk membuang semua sel kosong:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Pendekatan lain adalah dengan mengisi nilai kosong tersebut dengan nilai purata dari lajur yang sesuai.

## Regresi Linear Mudah

[![ML untuk pemula - Regresi Linear dan Polinomial menggunakan Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML untuk pemula - Regresi Linear dan Polinomial menggunakan Scikit-learn")

> 🎥 Klik imej di atas untuk video ringkas mengenai regresi linear dan polinomial.

Untuk melatih model Regresi Linear kita, kita akan menggunakan perpustakaan **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Kita mulakan dengan memisahkan nilai input (ciri) dan output yang dijangkakan (label) ke dalam array numpy yang berasingan:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Perhatikan bahawa kita perlu melakukan `reshape` pada data input agar pakej Regresi Linear memahaminya dengan betul. Regresi Linear mengharapkan array 2D sebagai input, di mana setiap baris array sesuai dengan vektor ciri input. Dalam kes kita, kerana kita hanya mempunyai satu input - kita memerlukan array dengan bentuk N×1, di mana N adalah saiz dataset.

Kemudian, kita perlu membahagikan data kepada dataset latihan dan ujian, supaya kita boleh mengesahkan model kita selepas latihan:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Akhirnya, melatih model Regresi Linear sebenar hanya memerlukan dua baris kod. Kita mendefinisikan objek `LinearRegression`, dan melatihnya dengan data kita menggunakan kaedah `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objek `LinearRegression` selepas `fit`-ting mengandungi semua pekali regresi, yang boleh diakses menggunakan sifat `.coef_`. Dalam kes kita, hanya ada satu pekali, yang seharusnya sekitar `-0.017`. Ini bermaksud bahawa harga nampaknya menurun sedikit dengan masa, tetapi tidak terlalu banyak, sekitar 2 sen sehari. Kita juga boleh mengakses titik persilangan regresi dengan paksi Y menggunakan `lin_reg.intercept_` - ia akan sekitar `21` dalam kes kita, menunjukkan harga pada awal tahun.

Untuk melihat sejauh mana ketepatan model kita, kita boleh meramalkan harga pada dataset ujian, dan kemudian mengukur sejauh mana ramalan kita mendekati nilai yang dijangkakan. Ini boleh dilakukan menggunakan metrik ralat kuadrat purata (MSE), yang merupakan purata semua perbezaan kuadrat antara nilai yang dijangkakan dan yang diramalkan.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Kesalahan kita nampaknya berkisar pada 2 titik, iaitu ~17%. Tidak begitu baik. Satu lagi penunjuk kualiti model ialah **koefisien penentuan**, yang boleh diperoleh seperti ini:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Jika nilainya 0, ini bermakna model tidak mengambil kira data input, dan bertindak sebagai *peramal linear paling buruk*, iaitu hanya nilai purata hasil. Nilai 1 bermakna kita dapat meramal semua output yang dijangka dengan sempurna. Dalam kes kita, koefisien adalah sekitar 0.06, yang agak rendah.

Kita juga boleh memplot data ujian bersama dengan garis regresi untuk melihat dengan lebih jelas bagaimana regresi berfungsi dalam kes kita:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Regresi linear" src="images/linear-results.png" width="50%" />

## Regresi Polinomial

Satu lagi jenis Regresi Linear ialah Regresi Polinomial. Walaupun kadangkala terdapat hubungan linear antara pemboleh ubah - semakin besar labu dalam isi padu, semakin tinggi harganya - kadangkala hubungan ini tidak boleh diplot sebagai satah atau garis lurus.

✅ Berikut adalah [beberapa contoh lagi](https://online.stat.psu.edu/stat501/lesson/9/9.8) data yang boleh menggunakan Regresi Polinomial.

Lihat semula hubungan antara Tarikh dan Harga. Adakah scatterplot ini kelihatan seperti ia semestinya dianalisis oleh garis lurus? Bukankah harga boleh berubah-ubah? Dalam kes ini, anda boleh mencuba regresi polinomial.

✅ Polinomial adalah ekspresi matematik yang mungkin terdiri daripada satu atau lebih pemboleh ubah dan koefisien.

Regresi polinomial mencipta garis melengkung untuk lebih sesuai dengan data tidak linear. Dalam kes kita, jika kita memasukkan pemboleh ubah `DayOfYear` kuasa dua ke dalam data input, kita sepatutnya dapat menyesuaikan data kita dengan lengkung parabola, yang akan mempunyai minimum pada titik tertentu dalam tahun tersebut.

Scikit-learn termasuk [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) yang berguna untuk menggabungkan pelbagai langkah pemprosesan data bersama-sama. **Pipeline** ialah rangkaian **estimators**. Dalam kes kita, kita akan mencipta pipeline yang pertama menambah ciri polinomial kepada model kita, dan kemudian melatih regresi:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Menggunakan `PolynomialFeatures(2)` bermakna kita akan memasukkan semua polinomial darjah kedua daripada data input. Dalam kes kita, ini hanya bermakna `DayOfYear`<sup>2</sup>, tetapi dengan dua pemboleh ubah input X dan Y, ini akan menambah X<sup>2</sup>, XY dan Y<sup>2</sup>. Kita juga boleh menggunakan polinomial darjah lebih tinggi jika kita mahu.

Pipeline boleh digunakan dengan cara yang sama seperti objek `LinearRegression` asal, iaitu kita boleh `fit` pipeline, dan kemudian menggunakan `predict` untuk mendapatkan hasil ramalan. Berikut adalah graf yang menunjukkan data ujian, dan lengkung anggaran:

<img alt="Regresi polinomial" src="images/poly-results.png" width="50%" />

Menggunakan Regresi Polinomial, kita boleh mendapatkan MSE yang sedikit lebih rendah dan penentuan yang lebih tinggi, tetapi tidak secara signifikan. Kita perlu mengambil kira ciri-ciri lain!

> Anda boleh melihat bahawa harga labu paling rendah diperhatikan sekitar Halloween. Bagaimana anda boleh menjelaskan ini?

🎃 Tahniah, anda baru sahaja mencipta model yang boleh membantu meramal harga labu pai. Anda mungkin boleh mengulangi prosedur yang sama untuk semua jenis labu, tetapi itu akan menjadi membosankan. Mari kita belajar sekarang bagaimana mengambil kira jenis labu dalam model kita!

## Ciri Kategori

Dalam dunia ideal, kita mahu dapat meramal harga untuk pelbagai jenis labu menggunakan model yang sama. Walau bagaimanapun, lajur `Variety` agak berbeza daripada lajur seperti `Month`, kerana ia mengandungi nilai bukan numerik. Lajur seperti ini dipanggil **kategori**.

[![ML untuk pemula - Ramalan Ciri Kategori dengan Regresi Linear](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML untuk pemula - Ramalan Ciri Kategori dengan Regresi Linear")

> 🎥 Klik imej di atas untuk video ringkas mengenai penggunaan ciri kategori.

Di sini anda boleh melihat bagaimana harga purata bergantung pada jenis:

<img alt="Harga purata mengikut jenis" src="images/price-by-variety.png" width="50%" />

Untuk mengambil kira jenis, kita perlu menukarnya kepada bentuk numerik terlebih dahulu, atau **encode**. Terdapat beberapa cara kita boleh melakukannya:

* **Pengekodan numerik** yang mudah akan membina jadual pelbagai jenis, dan kemudian menggantikan nama jenis dengan indeks dalam jadual tersebut. Ini bukan idea terbaik untuk regresi linear, kerana regresi linear mengambil nilai numerik sebenar indeks, dan menambahnya kepada hasil, dengan mendarabkan dengan beberapa koefisien. Dalam kes kita, hubungan antara nombor indeks dan harga jelas tidak linear, walaupun kita memastikan bahawa indeks diatur dalam cara tertentu.
* **Pengekodan one-hot** akan menggantikan lajur `Variety` dengan 4 lajur berbeza, satu untuk setiap jenis. Setiap lajur akan mengandungi `1` jika baris yang sepadan adalah daripada jenis tertentu, dan `0` jika tidak. Ini bermakna akan ada empat koefisien dalam regresi linear, satu untuk setiap jenis labu, yang bertanggungjawab untuk "harga permulaan" (atau lebih tepat "harga tambahan") untuk jenis tertentu.

Kod di bawah menunjukkan bagaimana kita boleh melakukan pengekodan one-hot untuk jenis:

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

Untuk melatih regresi linear menggunakan jenis yang telah dikodkan one-hot sebagai input, kita hanya perlu memulakan data `X` dan `y` dengan betul:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Selebihnya kod adalah sama seperti yang kita gunakan di atas untuk melatih Regresi Linear. Jika anda mencubanya, anda akan melihat bahawa ralat kuasa dua min adalah hampir sama, tetapi kita mendapat koefisien penentuan yang jauh lebih tinggi (~77%). Untuk mendapatkan ramalan yang lebih tepat, kita boleh mengambil kira lebih banyak ciri kategori, serta ciri numerik seperti `Month` atau `DayOfYear`. Untuk mendapatkan satu array besar ciri-ciri, kita boleh menggunakan `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Di sini kita juga mengambil kira `City` dan jenis `Package`, yang memberikan kita MSE 2.84 (10%), dan penentuan 0.94!

## Menggabungkan semuanya

Untuk mencipta model terbaik, kita boleh menggunakan data gabungan (kategori yang dikodkan one-hot + numerik) daripada contoh di atas bersama dengan Regresi Polinomial. Berikut adalah kod lengkap untuk kemudahan anda:

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

Ini sepatutnya memberikan kita koefisien penentuan terbaik hampir 97%, dan MSE=2.23 (~8% ralat ramalan).

| Model | MSE | Penentuan |
|-------|-----|-----------|
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |
| `Variety` Linear | 5.24 (19.7%) | 0.77 |
| Semua ciri Linear | 2.84 (10.5%) | 0.94 |
| Semua ciri Polynomial | 2.23 (8.25%) | 0.97 |

🏆 Syabas! Anda telah mencipta empat model Regresi dalam satu pelajaran, dan meningkatkan kualiti model kepada 97%. Dalam bahagian terakhir mengenai Regresi, anda akan belajar tentang Regresi Logistik untuk menentukan kategori.

---
## 🚀Cabaran

Uji beberapa pemboleh ubah berbeza dalam notebook ini untuk melihat bagaimana korelasi berkait dengan ketepatan model.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Kajian & Pembelajaran Kendiri

Dalam pelajaran ini kita belajar tentang Regresi Linear. Terdapat jenis Regresi lain yang penting. Baca tentang teknik Stepwise, Ridge, Lasso dan Elasticnet. Kursus yang baik untuk belajar lebih lanjut ialah [Kursus Pembelajaran Statistik Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tugasan 

[Bina Model](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.