<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T12:08:59+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "uk"
}
-->
# Моделі кластеризації для машинного навчання

Кластеризація — це завдання машинного навчання, яке спрямоване на пошук об'єктів, схожих один на одного, і групування їх у групи, які називаються кластерами. Що відрізняє кластеризацію від інших підходів у машинному навчанні, так це те, що процес відбувається автоматично. Насправді, можна сказати, що це протилежність до навчання з учителем.

## Регіональна тема: моделі кластеризації для музичних уподобань аудиторії Нігерії 🎧

Різноманітна аудиторія Нігерії має різноманітні музичні уподобання. Використовуючи дані, отримані зі Spotify (натхненні [цією статтею](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), давайте розглянемо деяку популярну музику в Нігерії. Цей набір даних містить інформацію про такі показники пісень, як "танцювальність", "акустичність", гучність, "мовність", популярність і енергійність. Буде цікаво виявити закономірності в цих даних!

![Програвач](../../../5-Clustering/images/turntable.jpg)

> Фото від <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> на <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
У цій серії уроків ви відкриєте нові способи аналізу даних за допомогою технік кластеризації. Кластеризація особливо корисна, коли ваш набір даних не має міток. Якщо мітки є, тоді техніки класифікації, які ви вивчали в попередніх уроках, можуть бути більш корисними. Але в тих випадках, коли ви хочете згрупувати дані без міток, кластеризація — чудовий спосіб виявити закономірності.

> Існують корисні інструменти з низьким рівнем кодування, які можуть допомогти вам навчитися працювати з моделями кластеризації. Спробуйте [Azure ML для цього завдання](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Уроки

1. [Вступ до кластеризації](1-Visualize/README.md)
2. [Кластеризація методом K-Means](2-K-Means/README.md)

## Авторство

Ці уроки були написані з 🎶 [Jen Looper](https://www.twitter.com/jenlooper) за допомогою корисних рецензій від [Rishit Dagli](https://rishit_dagli) та [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Набір даних [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) був отриманий з Kaggle, як дані зі Spotify.

Корисні приклади K-Means, які допомогли створити цей урок, включають [дослідження ірисів](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [вступний ноутбук](https://www.kaggle.com/prashant111/k-means-clustering-with-python) та [гіпотетичний приклад НГО](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Відмова від відповідальності**:  
Цей документ був перекладений за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, будь ласка, майте на увазі, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ на його рідній мові слід вважати авторитетним джерелом. Для критичної інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникають внаслідок використання цього перекладу.