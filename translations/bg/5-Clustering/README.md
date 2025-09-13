<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-04T23:57:22+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "bg"
}
-->
# Модели за клъстеризация в машинното обучение

Клъстеризацията е задача в машинното обучение, която се стреми да открие обекти, които си приличат, и да ги групира в групи, наречени клъстери. Това, което отличава клъстеризацията от другите подходи в машинното обучение, е, че процесът се случва автоматично. Всъщност, може да се каже, че това е противоположността на обучението с учител.

## Регионална тема: модели за клъстеризация на музикалните вкусове на нигерийската аудитория 🎧

Разнообразната аудитория в Нигерия има разнообразни музикални вкусове. Използвайки данни, събрани от Spotify (вдъхновено от [тази статия](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), нека разгледаме някои популярни песни в Нигерия. Този набор от данни включва информация за различни песни, като например техния 'danceability' рейтинг, 'acousticness', сила на звука, 'speechiness', популярност и енергия. Ще бъде интересно да открием модели в тези данни!

![Грамофон](../../../5-Clustering/images/turntable.jpg)

> Снимка от <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> на <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
В тази серия от уроци ще откриете нови начини за анализиране на данни, използвайки техники за клъстеризация. Клъстеризацията е особено полезна, когато вашият набор от данни няма етикети. Ако има етикети, тогава техниките за класификация, които сте научили в предишните уроци, може да са по-полезни. Но в случаите, когато искате да групирате данни без етикети, клъстеризацията е отличен начин за откриване на модели.

> Съществуват полезни инструменти с нисък код, които могат да ви помогнат да научите повече за работата с модели за клъстеризация. Опитайте [Azure ML за тази задача](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Уроци

1. [Въведение в клъстеризацията](1-Visualize/README.md)
2. [Клъстеризация с K-Means](2-K-Means/README.md)

## Кредити

Тези уроци бяха написани с 🎶 от [Jen Looper](https://www.twitter.com/jenlooper) с полезни ревюта от [Rishit Dagli](https://rishit_dagli) и [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Наборът от данни [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) беше взет от Kaggle и събран от Spotify.

Полезни примери за K-Means, които помогнаха при създаването на този урок, включват това [изследване на ириси](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), този [въвеждащ ноутбук](https://www.kaggle.com/prashant111/k-means-clustering-with-python) и този [хипотетичен пример за НПО](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.