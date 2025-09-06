<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T07:58:54+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "tr"
}
-->
# Yemek Kültürü Sınıflandırıcıları 1

Bu derste, önceki derste kaydettiğiniz dengeli ve temiz yemek kültürleri verileriyle dolu veri setini kullanacaksınız.

Bu veri setini, _bir grup malzemeye dayanarak belirli bir ulusal yemek kültürünü tahmin etmek_ için çeşitli sınıflandırıcılarla kullanacaksınız. Bunu yaparken, algoritmaların sınıflandırma görevlerinde nasıl kullanılabileceği hakkında daha fazla bilgi edineceksiniz.

## [Ders Öncesi Testi](https://ff-quizzes.netlify.app/en/ml/)
# Hazırlık

[1. Ders](../1-Introduction/README.md)'i tamamladığınızı varsayarak, bu dört ders için kök `/data` klasöründe _cleaned_cuisines.csv_ dosyasının bulunduğundan emin olun.

## Alıştırma - ulusal bir yemek kültürünü tahmin etme

1. Bu dersin _notebook.ipynb_ klasöründe çalışarak, Pandas kütüphanesiyle birlikte bu dosyayı içe aktarın:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Veriler şu şekilde görünüyor:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

1. Şimdi birkaç kütüphane daha içe aktarın:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X ve y koordinatlarını eğitim için iki veri çerçevesine ayırın. `cuisine` etiketler veri çerçevesi olabilir:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Şöyle görünecek:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` sütununu ve `cuisine` sütununu `drop()` kullanarak kaldırın. Geri kalan verileri eğitilebilir özellikler olarak kaydedin:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Özellikleriniz şu şekilde görünecek:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Artık modelinizi eğitmeye hazırsınız!

## Sınıflandırıcı Seçimi

Verileriniz temiz ve eğitime hazır olduğuna göre, iş için hangi algoritmayı kullanacağınıza karar vermelisiniz.

Scikit-learn, sınıflandırmayı Denetimli Öğrenme altında gruplandırır ve bu kategoride birçok sınıflandırma yöntemi bulabilirsiniz. [Çeşitlilik](https://scikit-learn.org/stable/supervised_learning.html) ilk bakışta oldukça kafa karıştırıcı olabilir. Aşağıdaki yöntemlerin tümü sınıflandırma tekniklerini içerir:

- Doğrusal Modeller
- Destek Vektör Makineleri
- Stokastik Gradyan İnişi
- En Yakın Komşular
- Gauss Süreçleri
- Karar Ağaçları
- Toplu yöntemler (oylama sınıflandırıcı)
- Çok sınıflı ve çok çıkışlı algoritmalar (çok sınıflı ve çok etiketli sınıflandırma, çok sınıflı-çok çıkışlı sınıflandırma)

> Verileri sınıflandırmak için [sinir ağlarını](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) da kullanabilirsiniz, ancak bu dersin kapsamı dışındadır.

### Hangi sınıflandırıcıyı seçmeli?

Peki, hangi sınıflandırıcıyı seçmelisiniz? Çoğu zaman, birkaçını deneyip iyi bir sonuç aramak bir test yöntemi olabilir. Scikit-learn, oluşturulmuş bir veri setinde KNeighbors, SVC iki şekilde, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ve QuadraticDiscriminationAnalysis'ı karşılaştıran bir [yan yana karşılaştırma](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) sunar ve sonuçları görselleştirir:

![sınıflandırıcıların karşılaştırması](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafikler Scikit-learn belgelerinde oluşturulmuştur

> AutoML, bu karşılaştırmaları bulutta çalıştırarak ve verileriniz için en iyi algoritmayı seçmenize olanak tanıyarak bu sorunu kolayca çözer. [Buradan deneyin](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Daha iyi bir yaklaşım

Ancak rastgele tahmin etmekten daha iyi bir yol, bu indirilebilir [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) üzerindeki fikirleri takip etmektir. Burada, çok sınıflı problemimiz için bazı seçeneklerimiz olduğunu keşfediyoruz:

![çok sınıflı problemler için cheat sheet](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Microsoft'un Algoritma Cheat Sheet'inin bir bölümü, çok sınıflı sınıflandırma seçeneklerini detaylandırıyor

✅ Bu cheat sheet'i indirin, yazdırın ve duvarınıza asın!

### Mantık Yürütme

Kısıtlamalarımızı göz önünde bulundurarak farklı yaklaşımları mantıkla değerlendirelim:

- **Sinir ağları çok ağır**. Temiz ama minimal veri setimiz ve eğitimi yerel olarak notebooklar üzerinden çalıştırdığımız gerçeği göz önüne alındığında, sinir ağları bu görev için çok ağırdır.
- **İki sınıflı sınıflandırıcı yok**. İki sınıflı bir sınıflandırıcı kullanmıyoruz, bu nedenle one-vs-all seçeneği eleniyor.
- **Karar ağacı veya lojistik regresyon işe yarayabilir**. Çok sınıflı veriler için bir karar ağacı veya lojistik regresyon işe yarayabilir.
- **Çok sınıflı Boosted Karar Ağaçları farklı bir problemi çözer**. Çok sınıflı Boosted Karar Ağacı, sıralamalar oluşturmak için tasarlanmış parametrik olmayan görevler için en uygundur, bu nedenle bizim için kullanışlı değildir.

### Scikit-learn Kullanımı

Verilerimizi analiz etmek için Scikit-learn kullanacağız. Ancak, Scikit-learn'de lojistik regresyonu kullanmanın birçok yolu vardır. Geçilecek [parametrelere](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) bir göz atın.

Temelde, Scikit-learn'den lojistik regresyon yapmasını istediğimizde belirtmemiz gereken iki önemli parametre vardır - `multi_class` ve `solver`. `multi_class` değeri belirli bir davranışı uygular. Solver'ın değeri ise hangi algoritmanın kullanılacağını belirler. Tüm solver'lar tüm `multi_class` değerleriyle eşleştirilemez.

Belgelerden öğrendiğimize göre, çok sınıflı durumda eğitim algoritması:

- **one-vs-rest (OvR) şemasını kullanır**, eğer `multi_class` seçeneği `ovr` olarak ayarlanmışsa
- **çapraz entropi kaybını kullanır**, eğer `multi_class` seçeneği `multinomial` olarak ayarlanmışsa. (Şu anda `multinomial` seçeneği yalnızca ‘lbfgs’, ‘sag’, ‘saga’ ve ‘newton-cg’ solver'ları tarafından desteklenmektedir.)

> 🎓 Buradaki 'şema', 'ovr' (one-vs-rest) veya 'multinomial' olabilir. Lojistik regresyon aslında ikili sınıflandırmayı desteklemek için tasarlandığından, bu şemalar onun çok sınıflı sınıflandırma görevlerini daha iyi ele almasına olanak tanır. [kaynak](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver', "optimizasyon probleminde kullanılacak algoritma" olarak tanımlanır. [kaynak](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn, solver'ların farklı veri yapılarının sunduğu zorlukları nasıl ele aldığını açıklamak için şu tabloyu sunar:

![solver'lar](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Alıştırma - veriyi bölmek

Son dersinizde lojistik regresyon hakkında bilgi edindiğiniz için, ilk eğitim denemeniz için lojistik regresyona odaklanabiliriz.
Verilerinizi `train_test_split()` çağırarak eğitim ve test gruplarına ayırın:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Alıştırma - lojistik regresyon uygulamak

Çok sınıflı durumu kullandığınız için hangi _şemayı_ kullanacağınızı ve hangi _solver'ı_ ayarlayacağınızı seçmeniz gerekiyor. Multi_class ayarı `ovr` ve solver ayarı `liblinear` olan LogisticRegression kullanarak eğitin.

1. Multi_class `ovr` ve solver `liblinear` olarak ayarlanmış bir lojistik regresyon oluşturun:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Varsayılan olarak sıkça ayarlanan `lbfgs` gibi farklı bir solver deneyin
Not: Verilerinizi düzleştirmeniz gerektiğinde Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) fonksiyonunu kullanın.
Doğruluk oranı **%80**'in üzerinde oldukça iyi!

1. Bu modeli bir veri satırını test ederek (#50) çalışırken görebilirsiniz:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Sonuç yazdırılır:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Farklı bir satır numarası deneyin ve sonuçları kontrol edin.

1. Daha derine inerek, bu tahminin doğruluğunu kontrol edebilirsiniz:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Sonuç yazdırılır - Hint mutfağı en iyi tahmin olarak, yüksek bir olasılıkla:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Modelin neden Hint mutfağı olduğundan bu kadar emin olduğunu açıklayabilir misiniz?

1. Regresyon derslerinde yaptığınız gibi bir sınıflandırma raporu yazdırarak daha fazla ayrıntı elde edin:

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

## 🚀Meydan Okuma

Bu derste, temizlenmiş verilerinizi kullanarak bir dizi malzemeye dayanarak ulusal bir mutfağı tahmin edebilen bir makine öğrenimi modeli oluşturdunuz. Scikit-learn'ün verileri sınıflandırmak için sunduğu birçok seçeneği incelemek için biraz zaman ayırın. Sahne arkasında neler olduğunu anlamak için 'solver' kavramını daha derinlemesine araştırın.

## [Ders sonrası test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Lojistik regresyonun matematiğini biraz daha derinlemesine inceleyin: [bu derste](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Ödev 

[Solver'ları inceleyin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.