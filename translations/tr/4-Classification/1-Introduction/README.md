<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-06T08:00:41+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "tr"
}
-->
# Sınıflandırmaya Giriş

Bu dört derste, klasik makine öğreniminin temel odak noktalarından biri olan _sınıflandırmayı_ keşfedeceksiniz. Asya ve Hindistan'ın tüm muhteşem mutfakları hakkında bir veri seti kullanarak çeşitli sınıflandırma algoritmalarını inceleyeceğiz. Umarım acıkmışsınızdır!

![bir tutam yeter!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Bu derslerde pan-Asya mutfaklarını kutlayın! Görsel: [Jen Looper](https://twitter.com/jenlooper)

Sınıflandırma, regresyon teknikleriyle birçok ortak noktası olan bir [denetimli öğrenme](https://wikipedia.org/wiki/Supervised_learning) türüdür. Makine öğrenimi, veri setlerini kullanarak değerleri veya şeylere isimler tahmin etmekle ilgiliyse, sınıflandırma genellikle iki gruba ayrılır: _ikili sınıflandırma_ ve _çok sınıflı sınıflandırma_.

[![Sınıflandırmaya giriş](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Sınıflandırmaya giriş")

> 🎥 Yukarıdaki görsele tıklayarak bir video izleyin: MIT'den John Guttag sınıflandırmayı tanıtıyor

Unutmayın:

- **Doğrusal regresyon**, değişkenler arasındaki ilişkileri tahmin etmenize ve yeni bir veri noktasının bu çizgiyle ilişkili olarak nerede yer alacağını doğru bir şekilde tahmin etmenize yardımcı oldu. Örneğin, _Eylül ve Aralık aylarında bir kabağın fiyatını_ tahmin edebilirsiniz.
- **Lojistik regresyon**, "ikili kategorileri" keşfetmenize yardımcı oldu: bu fiyat noktasında, _bu kabak turuncu mu yoksa turuncu değil mi_?

Sınıflandırma, bir veri noktasının etiketini veya sınıfını belirlemenin diğer yollarını belirlemek için çeşitli algoritmalar kullanır. Bu mutfak verileriyle çalışarak, bir grup malzemeyi gözlemleyerek hangi mutfağa ait olduğunu belirleyip belirleyemeyeceğimizi görelim.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bu ders R dilinde de mevcut!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Giriş

Sınıflandırma, makine öğrenimi araştırmacısının ve veri bilimcisinin temel faaliyetlerinden biridir. Basit bir ikili değerin sınıflandırılmasından ("bu e-posta spam mi değil mi?"), bilgisayarla görme kullanarak karmaşık görüntü sınıflandırma ve segmentasyona kadar, verileri sınıflara ayırmak ve onlara sorular sormak her zaman faydalıdır.

Bu süreci daha bilimsel bir şekilde ifade etmek gerekirse, sınıflandırma yöntemi, giriş değişkenleri ile çıkış değişkenleri arasındaki ilişkiyi haritalamanızı sağlayan bir tahmin modeli oluşturur.

![ikili vs. çok sınıflı sınıflandırma](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Sınıflandırma algoritmalarının ele alması gereken ikili ve çok sınıflı problemler. Görsel: [Jen Looper](https://twitter.com/jenlooper)

Verilerimizi temizleme, görselleştirme ve makine öğrenimi görevlerimize hazırlama sürecine başlamadan önce, makine öğreniminin verileri sınıflandırmak için kullanılabileceği çeşitli yollar hakkında biraz bilgi edinelim.

[İstatistikten](https://wikipedia.org/wiki/Statistical_classification) türetilen klasik makine öğrenimi ile sınıflandırma, `sigara içen`, `kilo` ve `yaş` gibi özellikleri kullanarak _X hastalığını geliştirme olasılığını_ belirler. Daha önce gerçekleştirdiğiniz regresyon egzersizlerine benzer bir denetimli öğrenme tekniği olarak, verileriniz etiketlenir ve makine öğrenimi algoritmaları bu etiketleri bir veri setinin sınıflarını (veya 'özelliklerini') sınıflandırmak ve tahmin etmek ve bunları bir gruba veya sonuca atamak için kullanır.

✅ Bir mutfak hakkında bir veri seti hayal etmek için bir dakikanızı ayırın. Çok sınıflı bir model neyi cevaplayabilir? İkili bir model neyi cevaplayabilir? Belirli bir mutfağın çemen otu kullanma olasılığını belirlemek isteseydiniz ne olurdu? Ya yıldız anason, enginar, karnabahar ve yaban turpu dolu bir market çantası hediye edilseydi ve tipik bir Hint yemeği yapıp yapamayacağınızı görmek isteseydiniz?

[![Çılgın gizemli sepetler](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Çılgın gizemli sepetler")

> 🎥 Yukarıdaki görsele tıklayarak bir video izleyin. 'Chopped' adlı programın tüm temeli, şeflerin rastgele seçilmiş malzemelerden bir yemek yapması gereken 'gizemli sepet' üzerine kuruludur. Kesinlikle bir makine öğrenimi modeli yardımcı olurdu!

## Merhaba 'sınıflandırıcı'

Bu mutfak veri setine sormak istediğimiz soru aslında bir **çok sınıflı soru**, çünkü çalışabileceğimiz birkaç potansiyel ulusal mutfak var. Bir grup malzeme verildiğinde, bu birçok sınıftan hangisine veri uyacak?

Scikit-learn, çözmek istediğiniz problemin türüne bağlı olarak verileri sınıflandırmak için kullanabileceğiniz birkaç farklı algoritma sunar. Önümüzdeki iki derste, bu algoritmalardan birkaçını öğreneceksiniz.

## Egzersiz - Verilerinizi temizleyin ve dengeleyin

Bu projeye başlamadan önceki ilk görev, daha iyi sonuçlar elde etmek için verilerinizi temizlemek ve **dengelemektir**. Bu klasörün kökündeki boş _notebook.ipynb_ dosyasıyla başlayın.

İlk olarak yüklemeniz gereken şey [imblearn](https://imbalanced-learn.org/stable/). Bu, verileri daha iyi dengelemenizi sağlayacak bir Scikit-learn paketidir (bu görev hakkında birazdan daha fazla bilgi edineceksiniz).

1. `imblearn` yüklemek için, aşağıdaki gibi `pip install` komutunu çalıştırın:

    ```python
    pip install imblearn
    ```

1. Verilerinizi içe aktarmak ve görselleştirmek için gereken paketleri içe aktarın, ayrıca `imblearn`'den `SMOTE`'yi içe aktarın.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Şimdi verileri içe aktarmaya hazırsınız.

1. Bir sonraki görev, verileri içe aktarmaktır:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   `read_csv()` kullanarak _cusines.csv_ dosyasının içeriğini okuyabilir ve bunu `df` değişkenine yerleştirebilirsiniz.

1. Verilerin şekline bakın:

    ```python
    df.head()
    ```

   İlk beş satır şöyle görünür:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Bu veri hakkında bilgi almak için `info()` çağırın:

    ```python
    df.info()
    ```

    Çıktınız şu şekilde görünür:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Egzersiz - Mutfaklar hakkında bilgi edinme

Şimdi işler daha ilginç hale gelmeye başlıyor. Verilerin dağılımını, mutfak başına keşfedelim.

1. Verileri çubuklar olarak çizmek için `barh()` çağırın:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![mutfak veri dağılımı](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Sınırlı sayıda mutfak var, ancak veri dağılımı eşit değil. Bunu düzeltebilirsiniz! Bunu yapmadan önce biraz daha keşfedin.

1. Mutfak başına ne kadar veri olduğunu öğrenin ve yazdırın:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    Çıktı şu şekilde görünür:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Malzemeleri Keşfetme

Şimdi verileri daha derinlemesine inceleyebilir ve her mutfak için tipik malzemelerin neler olduğunu öğrenebilirsiniz. Mutfaklar arasında kafa karışıklığı yaratan tekrarlayan verileri temizlemelisiniz, bu sorunu öğrenelim.

1. Python'da bir `create_ingredient()` fonksiyonu oluşturun. Bu fonksiyon, yararsız bir sütunu kaldırarak ve malzemeleri sayısına göre sıralayarak bir malzeme veri çerçevesi oluşturacaktır:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Şimdi bu fonksiyonu kullanarak her mutfak için en popüler on malzeme hakkında bir fikir edinebilirsiniz.

1. `create_ingredient()` çağırın ve `barh()` çağırarak çizin:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Japon verileri için aynısını yapın:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Şimdi Çin malzemeleri için:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Hint malzemelerini çizin:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Son olarak, Kore malzemelerini çizin:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Şimdi, `drop()` çağırarak farklı mutfaklar arasında kafa karışıklığı yaratan en yaygın malzemeleri kaldırın:

   Herkes pirinç, sarımsak ve zencefili sever!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Veri Setini Dengeleme

Verileri temizledikten sonra, [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Sentetik Azınlık Aşırı Örnekleme Tekniği" - kullanarak dengeleyin.

1. `fit_resample()` çağırın, bu strateji interpolasyon yoluyla yeni örnekler oluşturur.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Verilerinizi dengeleyerek, sınıflandırırken daha iyi sonuçlar elde edersiniz. İkili bir sınıflandırmayı düşünün. Verilerinizin çoğu bir sınıfsa, bir makine öğrenimi modeli bu sınıfı daha sık tahmin edecektir, çünkü bu sınıf için daha fazla veri vardır. Verilerin dengelenmesi, herhangi bir dengesizliği alır ve bu dengesizliği ortadan kaldırmaya yardımcı olur.

1. Şimdi malzeme başına etiket sayısını kontrol edebilirsiniz:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Çıktınız şu şekilde görünür:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Veriler güzel, temiz, dengeli ve çok lezzetli!

1. Son adım, dengelenmiş verilerinizi, etiketler ve özellikler dahil olmak üzere, bir dosyaya aktarılabilecek yeni bir veri çerçevesine kaydetmektir:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Bu verileri `transformed_df.head()` ve `transformed_df.info()` kullanarak bir kez daha inceleyebilirsiniz. Gelecek derslerde kullanmak üzere bu verilerin bir kopyasını kaydedin:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Bu yeni CSV artık kök veri klasöründe bulunabilir.

---

## 🚀Meydan Okuma

Bu müfredat birkaç ilginç veri seti içeriyor. `data` klasörlerini inceleyin ve ikili veya çok sınıflı sınıflandırmaya uygun veri setleri içerip içermediğini görün. Bu veri setine hangi soruları sorardınız?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

SMOTE'nin API'sini keşfedin. Hangi kullanım durumları için en iyi şekilde kullanılır? Hangi sorunları çözer?

## Ödev 

[Sınıflandırma yöntemlerini keşfedin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.