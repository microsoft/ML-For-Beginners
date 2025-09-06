<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T08:00:12+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "tr"
}
-->
# Mutfak Sınıflandırıcıları 2

Bu ikinci sınıflandırma dersinde, sayısal verileri sınıflandırmanın daha fazla yolunu keşfedeceksiniz. Ayrıca, bir sınıflandırıcıyı diğerine tercih etmenin sonuçlarını öğreneceksiniz.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

### Ön Koşul

Önceki dersleri tamamladığınızı ve bu 4 derslik klasörün kök dizininde `data` klasörünüzde _cleaned_cuisines.csv_ adlı temizlenmiş bir veri kümesine sahip olduğunuzu varsayıyoruz.

### Hazırlık

_Notebook.ipynb_ dosyanız temizlenmiş veri kümesiyle yüklendi ve model oluşturma sürecine hazır olacak şekilde X ve y veri çerçevelerine bölündü.

## Bir sınıflandırma haritası

Daha önce, Microsoft'un hile sayfasını kullanarak verileri sınıflandırırken sahip olduğunuz çeşitli seçenekleri öğrenmiştiniz. Scikit-learn, sınıflandırıcılarınızı (diğer bir deyişle tahmin ediciler) daraltmanıza yardımcı olabilecek benzer, ancak daha ayrıntılı bir hile sayfası sunar:

![Scikit-learn'den ML Haritası](../../../../4-Classification/3-Classifiers-2/images/map.png)
> İpucu: [bu haritayı çevrimiçi ziyaret edin](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ve belgeleri okumak için yol boyunca tıklayın.

### Plan

Bu harita, verilerinizi net bir şekilde anladığınızda çok yardımcı olur, çünkü yollarında 'yürüyerek' bir karara varabilirsiniz:

- 50'den fazla örneğimiz var
- Bir kategori tahmin etmek istiyoruz
- Etiketlenmiş verilerimiz var
- 100.000'den az örneğimiz var
- ✨ Linear SVC seçebiliriz
- Bu işe yaramazsa, sayısal verilerimiz olduğu için
    - ✨ KNeighbors Classifier deneyebiliriz
      - Bu işe yaramazsa, ✨ SVC ve ✨ Ensemble Classifiers deneyin

Bu, takip edilmesi çok faydalı bir yol.

## Alıştırma - verileri bölün

Bu yolu takip ederek, kullanmak için bazı kütüphaneleri içe aktarmalıyız.

1. Gerekli kütüphaneleri içe aktarın:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Eğitim ve test verilerinizi bölün:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC sınıflandırıcısı

Support-Vector Clustering (SVC), ML tekniklerinin Support-Vector Machines ailesinin bir alt dalıdır (aşağıda bunlar hakkında daha fazla bilgi edinin). Bu yöntemde, etiketleri nasıl kümelendireceğinize karar vermek için bir 'kernel' seçebilirsiniz. 'C' parametresi, parametrelerin etkisini düzenleyen 'düzenleme' anlamına gelir. Kernel, [birkaç seçenekten](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) biri olabilir; burada, Linear SVC'den yararlanmak için 'linear' olarak ayarlıyoruz. Olasılık varsayılan olarak 'false'dur; burada olasılık tahminleri toplamak için 'true' olarak ayarlıyoruz. Rastgele durumu '0' olarak ayarlıyoruz, böylece veriler karıştırılarak olasılıklar elde ediliyor.

### Alıştırma - bir Linear SVC uygulayın

Bir sınıflandırıcı dizisi oluşturarak başlayın. Test ettikçe bu diziye kademeli olarak ekleme yapacaksınız.

1. Linear SVC ile başlayın:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Modelinizi Linear SVC kullanarak eğitin ve bir rapor yazdırın:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Sonuç oldukça iyi:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-Neighbors sınıflandırıcısı

K-Neighbors, hem denetimli hem de denetimsiz öğrenme için kullanılabilen ML yöntemlerinin "komşular" ailesinin bir parçasıdır. Bu yöntemde, önceden tanımlanmış bir nokta sayısı oluşturulur ve veriler bu noktaların etrafında toplanır, böylece veriler için genelleştirilmiş etiketler tahmin edilebilir.

### Alıştırma - K-Neighbors sınıflandırıcısını uygulayın

Önceki sınıflandırıcı iyiydi ve verilerle iyi çalıştı, ancak belki daha iyi bir doğruluk elde edebiliriz. Bir K-Neighbors sınıflandırıcısını deneyin.

1. Sınıflandırıcı dizinize bir satır ekleyin (Linear SVC öğesinden sonra bir virgül ekleyin):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Sonuç biraz daha kötü:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ [K-Neighbors hakkında bilgi edinin](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector sınıflandırıcıları, sınıflandırma ve regresyon görevleri için kullanılan ML yöntemlerinin [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) ailesinin bir parçasıdır. SVM'ler, "eğitim örneklerini iki kategori arasındaki mesafeyi en üst düzeye çıkarmak için uzaydaki noktalara eşler." Sonraki veriler bu uzaya eşlenir, böylece kategorileri tahmin edilebilir.

### Alıştırma - bir Support Vector Classifier uygulayın

Biraz daha iyi doğruluk elde etmek için bir Support Vector Classifier deneyelim.

1. K-Neighbors öğesinden sonra bir virgül ekleyin ve ardından bu satırı ekleyin:

    ```python
    'SVC': SVC(),
    ```

    Sonuç oldukça iyi!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ [Support-Vectors hakkında bilgi edinin](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Önceki test oldukça iyi olmasına rağmen, yolun sonuna kadar gidelim. Bazı 'Ensemble Classifiers' deneyelim, özellikle Random Forest ve AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Sonuç özellikle Random Forest için çok iyi:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ [Ensemble Classifiers hakkında bilgi edinin](https://scikit-learn.org/stable/modules/ensemble.html)

Bu Makine Öğrenimi yöntemi, modelin kalitesini artırmak için birkaç temel tahmin edicinin tahminlerini birleştirir. Örneğimizde, Random Trees ve AdaBoost kullandık.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), bir 'karar ağaçları' 'ormanı' oluşturan ve aşırı uyumu önlemek için rastgelelik ekleyen bir ortalama yöntemi. N_estimators parametresi, ağaç sayısını belirler.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), bir veri kümesine bir sınıflandırıcı uyarlar ve ardından aynı veri kümesine bu sınıflandırıcının kopyalarını uyarlar. Yanlış sınıflandırılmış öğelerin ağırlıklarına odaklanır ve bir sonraki sınıflandırıcıyı düzeltmek için uyumu ayarlar.

---

## 🚀Meydan Okuma

Bu tekniklerin her birinin ayarlayabileceğiniz çok sayıda parametresi vardır. Her birinin varsayılan parametrelerini araştırın ve bu parametreleri ayarlamanın modelin kalitesi için ne anlama geleceğini düşünün.

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu derslerde çok fazla terim var, bu yüzden [bu listeyi](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) gözden geçirmek için bir dakikanızı ayırın!

## Ödev 

[Parametrelerle Oynama](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dilindeki hali, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.