# 기계 학습을 위한 클러스터링 모델

클러스터링은 서로 닮은 객체를 찾아 그룹으로 묶는 기계 학습 작업입니다. 클러스터링이 기계 학습의 다른 방법들과 다른 점은 모든 일이 자동으로 일어난다는 것이며, 사실 이는 감독 학습(supervised learning)과는 정반대라고 할 수 있습니다.

## 지역 주제: 나이지리아 청중의 음악 취향을 위한 클러스터링 모델 🎧

나이지리아의 다양한 청중은 다양한 음악 취향을 가지고 있습니다. Spotify에서 스크랩한 데이터([이 기사](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)에서 영감을 받음)를 사용하여 나이지리아에서 인기 있는 음악 몇 가지를 살펴보겠습니다. 이 데이터셋은 여러 노래의 'danceability' 점수, 'acousticness', 음량, 'speechiness', 인기 및 에너지에 대한 데이터를 포함합니다. 이 데이터에서 패턴을 발견하는 것은 흥미롭습니다!

![턴테이블](../../../translated_images/ko/turntable.f2b86b13c53302dc.webp)

> 사진 제공: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> / <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
이 연속 강의에서, 클러스터링 기법을 사용하여 데이터를 분석하는 새로운 방법을 발견하게 될 것입니다. 클러스터링은 데이터셋에 라벨이 없을 때 특히 유용합니다. 라벨이 있는 경우 이전 강의에서 배운 분류 기법이 더 유용할 수 있습니다. 그러나 라벨이 없는 데이터를 그룹화하려고 할 때 클러스터링은 패턴을 발견하는 훌륭한 방법입니다.

> 클러스터링 모델 작업을 배우는 데 도움이 되는 유용한 로우코드 도구들이 있습니다. 이 작업을 위해 [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)을 사용해 보세요.

## 강의

1. [클러스터링 소개](1-Visualize/README.md)
2. [K-평균 클러스터링](2-K-Means/README.md)

## 크레딧

이 강의들은 🎶 [Jen Looper](https://www.twitter.com/jenlooper)가 작성했으며, [Rishit Dagli](https://rishit_dagli/)와 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)의 유용한 리뷰가 있었습니다.

[나이지리아 노래](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 데이터셋은 Spotify에서 스크랩한 내용을 Kaggle에서 출처로 제공받았습니다.

이 강의를 만드는 데 도움이 된 유용한 K-평균 예제로는 이 [아이리스 탐색](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), 이 [초보자 노트북](https://www.kaggle.com/prashant111/k-means-clustering-with-python), 그리고 이 [가상 NGO 예제](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)가 있습니다.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**면책 조항**:
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 기하기 위해 노력하고 있으나, 자동 번역은 오류나 부정확한 부분이 있을 수 있음을 유의하시기 바랍니다. 원본 문서의 원어본이 권위 있는 자료로 간주되어야 합니다. 중요한 정보의 경우, 전문가의 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->