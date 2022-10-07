# 머신러닝을 위한 Clustering 모델

Clustering 은 서로 비슷한 오브젝트를 찾고 clusters 라고 불린 그룹으로 묶는 머신러닝 작업입니다. Clustering 이 머신러닝의 다른 접근법과 다른 점은, 자동으로 어떤 일이 생긴다는 것이며, 사실은, supervised learning 의 반대라고 말하는 게 맞습니다.

## 지역 토픽: 나이지리아 사람들의 음악 취향을 위한 clustering 모델 🎧

나이지리아의 다양한 사람들은 다양한 음악 취향이 있습니다. Spotify 에서 긁어온 데이터를 사용해서 ([this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) 에서 영감받았습니다), 나이지니아에서 인기있는 음악을 알아보겠습니다. 데이터셋에 다양한 노래의 'danceability' 점수, 'acousticness', loudness, 'speechiness', 인기도와 에너지 데이터가 포함됩니다. 데이터에서 패턴을 찾는 것은 흥미로울 예정입니다!

![A turntable](../images/turntable.jpg)

> Photo by <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> on <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
이 강의의 시리즈에서, clustering 기술로 데이터를 분석하는 새로운 방식을 찾아볼 예정입니다. Clustering 은 데이터셋에 라벨이 없으면 더욱 더 유용합니다. 만약 라벨이 있다면, 이전 강의에서 배운대로 classification 기술이 더 유용할 수 있습니다. 그러나 라벨링되지 않은 데이터를 그룹으로 묶으려면, clustering 은 패턴을 발견하기 위한 좋은 방식입니다.

> clustering 모델 작업을 배울 때 도움을 받을 수 있는 유용한 low-code 도구가 있습니다. [Azure ML for this task](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)를 시도해봅니다.

## 강의

1. [clustering 소개하기](../1-Visualize/translations/README.ko.md)
2. [K-Means clustering](../2-K-Means/translations/README.ko.md)

## 크레딧

These lessons were written with 🎶 by [Jen Looper](https://www.twitter.com/jenlooper) with helpful reviews by [Rishit Dagli](https://rishit_dagli) and [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 데이터셋은 Spotify 스크랩해서 Kaggle 에서 가져왔습니다.

이 강의를 만들 때 도움된 유용한 K-Means 예시는 [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [introductory notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), 과 [hypothetical NGO example](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)이 포함됩니다.
