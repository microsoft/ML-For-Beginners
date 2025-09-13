<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T19:51:45+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "ms"
}
-->
# Pengelas Masakan 1

Dalam pelajaran ini, anda akan menggunakan dataset yang telah disimpan dari pelajaran sebelumnya yang penuh dengan data seimbang dan bersih mengenai masakan.

Anda akan menggunakan dataset ini dengan pelbagai pengelas untuk _meramalkan jenis masakan berdasarkan kumpulan bahan_. Semasa melakukannya, anda akan mempelajari lebih lanjut tentang beberapa cara algoritma boleh digunakan untuk tugas klasifikasi.

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)
# Persediaan

Dengan andaian anda telah menyelesaikan [Pelajaran 1](../1-Introduction/README.md), pastikan fail _cleaned_cuisines.csv_ wujud dalam folder root `/data` untuk keempat-empat pelajaran ini.

## Latihan - ramalkan jenis masakan

1. Bekerja dalam folder _notebook.ipynb_ pelajaran ini, import fail tersebut bersama perpustakaan Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data kelihatan seperti ini:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Sekarang, import beberapa lagi perpustakaan:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Bahagikan koordinat X dan y kepada dua dataframe untuk latihan. `cuisine` boleh menjadi dataframe label:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ia akan kelihatan seperti ini:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Buang lajur `Unnamed: 0` dan lajur `cuisine`, panggil `drop()`. Simpan data yang selebihnya sebagai ciri yang boleh dilatih:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ciri-ciri anda kelihatan seperti ini:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sekarang anda bersedia untuk melatih model anda!

## Memilih pengelas

Sekarang data anda bersih dan sedia untuk latihan, anda perlu memutuskan algoritma mana yang akan digunakan untuk tugas ini.

Scikit-learn mengelompokkan klasifikasi di bawah Pembelajaran Terkawal, dan dalam kategori itu anda akan menemui banyak cara untuk mengelas. [Kepelbagaian](https://scikit-learn.org/stable/supervised_learning.html) ini mungkin kelihatan mengelirukan pada pandangan pertama. Kaedah berikut semuanya termasuk teknik klasifikasi:

- Model Linear
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Kaedah Ensemble (Voting Classifier)
- Algoritma Multiclass dan multioutput (klasifikasi multilabel dan multiclass-multioutput)

> Anda juga boleh menggunakan [rangkaian neural untuk mengelas data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), tetapi itu di luar skop pelajaran ini.

### Pengelas mana yang perlu dipilih?

Jadi, pengelas mana yang patut anda pilih? Selalunya, mencuba beberapa pengelas dan mencari hasil yang baik adalah cara untuk menguji. Scikit-learn menawarkan [perbandingan sisi-sisi](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) pada dataset yang dicipta, membandingkan KNeighbors, SVC dua cara, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB dan QuadraticDiscrinationAnalysis, menunjukkan hasil yang divisualisasikan:

![perbandingan pengelas](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Plot yang dihasilkan dalam dokumentasi Scikit-learn

> AutoML menyelesaikan masalah ini dengan mudah dengan menjalankan perbandingan ini di awan, membolehkan anda memilih algoritma terbaik untuk data anda. Cuba [di sini](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Pendekatan yang lebih baik

Pendekatan yang lebih baik daripada meneka secara rawak adalah dengan mengikuti idea dalam [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) yang boleh dimuat turun ini. Di sini, kita mendapati bahawa, untuk masalah multiclass kita, kita mempunyai beberapa pilihan:

![cheatsheet untuk masalah multiclass](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Bahagian dari Algorithm Cheat Sheet Microsoft, memperincikan pilihan klasifikasi multiclass

✅ Muat turun cheat sheet ini, cetak, dan gantungkan di dinding anda!

### Penalaran

Mari kita lihat jika kita boleh membuat penalaran melalui pendekatan yang berbeza berdasarkan kekangan yang kita ada:

- **Rangkaian neural terlalu berat**. Memandangkan dataset kita bersih tetapi minimal, dan fakta bahawa kita menjalankan latihan secara tempatan melalui notebook, rangkaian neural terlalu berat untuk tugas ini.
- **Tiada pengelas dua kelas**. Kita tidak menggunakan pengelas dua kelas, jadi itu menyingkirkan one-vs-all.
- **Decision tree atau logistic regression boleh berfungsi**. Decision tree mungkin berfungsi, atau logistic regression untuk data multiclass.
- **Multiclass Boosted Decision Trees menyelesaikan masalah yang berbeza**. Multiclass boosted decision tree paling sesuai untuk tugas bukan parametrik, contohnya tugas yang direka untuk membina ranking, jadi ia tidak berguna untuk kita.

### Menggunakan Scikit-learn 

Kita akan menggunakan Scikit-learn untuk menganalisis data kita. Walau bagaimanapun, terdapat banyak cara untuk menggunakan logistic regression dalam Scikit-learn. Lihat [parameter untuk diteruskan](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Secara asasnya terdapat dua parameter penting - `multi_class` dan `solver` - yang perlu kita tentukan, apabila kita meminta Scikit-learn untuk melakukan logistic regression. Nilai `multi_class` menerapkan tingkah laku tertentu. Nilai solver adalah algoritma yang akan digunakan. Tidak semua solver boleh digabungkan dengan semua nilai `multi_class`.

Menurut dokumen, dalam kes multiclass, algoritma latihan:

- **Menggunakan skema one-vs-rest (OvR)**, jika pilihan `multi_class` ditetapkan kepada `ovr`
- **Menggunakan cross-entropy loss**, jika pilihan `multi_class` ditetapkan kepada `multinomial`. (Pada masa ini pilihan `multinomial` hanya disokong oleh solver ‘lbfgs’, ‘sag’, ‘saga’ dan ‘newton-cg’.)"

> 🎓 'Skema' di sini boleh sama ada 'ovr' (one-vs-rest) atau 'multinomial'. Oleh kerana logistic regression sebenarnya direka untuk menyokong klasifikasi binari, skema ini membolehkannya menangani tugas klasifikasi multiclass dengan lebih baik. [sumber](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' ditakrifkan sebagai "algoritma yang digunakan dalam masalah pengoptimuman". [sumber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn menawarkan jadual ini untuk menerangkan bagaimana solver menangani cabaran yang berbeza yang disebabkan oleh struktur data yang berbeza:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Latihan - bahagikan data

Kita boleh fokus pada logistic regression untuk percubaan latihan pertama kita kerana anda baru-baru ini mempelajarinya dalam pelajaran sebelumnya.
Bahagikan data anda kepada kumpulan latihan dan ujian dengan memanggil `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Latihan - gunakan logistic regression

Oleh kerana anda menggunakan kes multiclass, anda perlu memilih _skema_ yang akan digunakan dan _solver_ yang akan ditetapkan. Gunakan LogisticRegression dengan tetapan multiclass dan solver **liblinear** untuk latihan.

1. Cipta logistic regression dengan multi_class ditetapkan kepada `ovr` dan solver ditetapkan kepada `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Cuba solver lain seperti `lbfgs`, yang sering ditetapkan sebagai default
Gunakan fungsi Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) untuk meratakan data anda apabila diperlukan.
Ketepatan adalah baik pada lebih **80%**!

1. Anda boleh melihat model ini berfungsi dengan menguji satu baris data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Hasilnya dicetak:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Cuba nombor baris yang berbeza dan periksa hasilnya

1. Dengan lebih mendalam, anda boleh memeriksa ketepatan ramalan ini:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Hasilnya dicetak - masakan India adalah tekaan terbaiknya, dengan kebarangkalian yang baik:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Bolehkah anda jelaskan mengapa model ini cukup yakin bahawa ini adalah masakan India?

1. Dapatkan lebih banyak perincian dengan mencetak laporan klasifikasi, seperti yang anda lakukan dalam pelajaran regresi:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Cabaran

Dalam pelajaran ini, anda menggunakan data yang telah dibersihkan untuk membina model pembelajaran mesin yang boleh meramalkan masakan kebangsaan berdasarkan siri bahan-bahan. Luangkan masa untuk membaca pelbagai pilihan yang disediakan oleh Scikit-learn untuk mengklasifikasikan data. Selami lebih mendalam konsep 'solver' untuk memahami apa yang berlaku di sebalik tabir.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Selami sedikit lagi matematik di sebalik regresi logistik dalam [pelajaran ini](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tugasan 

[Kaji tentang solvers](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.