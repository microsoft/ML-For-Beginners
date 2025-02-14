# Maşın öyrənməsində modellərin klasterləşdirilməsi

Klasterləşdirmə maşın öyrənməsi tapşırığı olub, məqsədi bir-birinə bənzəyən obyektləri tapmaq və onları klasterlər adlanan qruplarda toplamaqdır. Klasterləşdirməni digər maşın öyrənməsi yanaşmalarından fərqləndirən cəhət ondan ibarətdir ki, burada hadisələr avtomatik olaraq baş verir, hətta bunu da demək doğru olar ki, klasterləşdirmə yönləndirilmiş öyrənmənin əksidir.

## Regional mövzu: Nigeriyalı auditoriyanın musiqi zövqünə görə modellərin klasterləşdirilməsi 🎧

Nigeriyanın çeşidli dinləyici auditoriyası çeşidli musiqi zövqünə sahibdir. Gəlin Spotify-dan əldə olunan datadan istifadə edərək ([bu məqalədən](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) ilham alınmışdır) Nigeriyada məşhur olan musiqiyə nəzər salaq. Bu data massivdə müxtəlif mahnıların 'oynaqlıq' dərəcəsi, 'akustikliyi', gurluğu, 'sözlülüyü', məşhurluğu və enerjisi haqqında məlumat var. Burada təkrarlanan nümunələri kəşf etmək maraqlı olacaq!

![turntable aləti](../images/turntable.jpg)

> <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> platformasında <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> tərəfindən çəkilmiş şəkil

Bu dərslərdə siz, klasterləşdirmə texnikalarından istifadə edərək datanı analiz etməyin yeni yolları ilə tanış olacaqsınız. Klasterləşdirmə, datasetdə etiketlərin çatışmayan hallarında xüsusilə faydalıdır. Əgər etiketlər mövcuddursa, o zaman əvvəlki dərslərdə öyrəndiyiniz təsnifatlandırma texnikaları daha faydalı ola bilər. Lakin məsələ etiketlənməmiş datanı qruplaşdıran zaman təkrarlanan nümunələri tapmaqdırsa, o zaman klasterləşdirmə əla üsuldur.

> Klasterləşdirmə modelləri ilə işləmək haqqında məlumat almanıza kömək olacaq az-kod yanaşmalı alətlər də mövcuddur. Bu tapşırıq üçün olan [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) sınaqdan keçirin.

## Dərslər

1. [Klasterləşdirmə bölməsinə giriş](../1-Visualize/translations/README.az.md)
2. [K-Ortalama klasterləşməsi](../2-K-Means/translations/README.az.md)

## Təşəkkürlər

Bu dərslər [Jen Looper](https://www.twitter.com/jenlooper) tərəfindən 🎶 ilə və [Rishit Dagli](https://www.twitter.com/rishit_dagli) və [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) tərəfindən gələn faydalı rəylərin köməkliyi ilə yazılmışdır.

[Nigeria Mahnıları](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) dataseti Spotify-dan əldə olunduğu formatda Kaggle-dan götürülmüşdür.

Bu dərsin yaradılmasına kömək olan faydalı K-Ortalama nümunələrinə [süsənin tədqiqi](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), bu [başlanğıc üçün praktika notbuku](https://www.kaggle.com/prashant111/k-means-clustering-with-python), və bu [xəyali QHT nümunəsi](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) daxildir.
