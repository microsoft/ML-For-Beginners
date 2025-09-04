<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:55:54+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ko"
}
-->
# 머신 러닝을 위한 클러스터링 모델

클러스터링은 서로 비슷한 객체를 찾아 클러스터라고 불리는 그룹으로 묶는 머신 러닝 작업입니다. 클러스터링이 머신 러닝의 다른 접근 방식과 다른 점은 모든 것이 자동으로 이루어진다는 점입니다. 사실, 이는 지도 학습(supervised learning)의 반대라고 할 수 있습니다.

## 지역 주제: 나이지리아 청중의 음악 취향을 위한 클러스터링 모델 🎧

나이지리아의 다양한 청중은 다양한 음악 취향을 가지고 있습니다. [이 기사](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)에서 영감을 받아 Spotify에서 수집한 데이터를 사용하여 나이지리아에서 인기 있는 음악을 살펴보겠습니다. 이 데이터셋에는 여러 곡의 '댄스 가능성(danceability)' 점수, '어쿠스틱(acousticness)', 음량(loudness), '스피치니스(speechiness)', 인기(popularity), 에너지(energy)에 대한 데이터가 포함되어 있습니다. 이 데이터에서 패턴을 발견하는 것은 흥미로울 것입니다!

![턴테이블](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.ko.jpg)

> 사진 제공: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> on <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
이 강의 시리즈에서는 클러스터링 기법을 사용하여 데이터를 분석하는 새로운 방법을 배우게 됩니다. 클러스터링은 데이터셋에 레이블이 없는 경우 특히 유용합니다. 만약 레이블이 있다면, 이전 강의에서 배운 분류(classification) 기법이 더 유용할 수 있습니다. 하지만 레이블이 없는 데이터를 그룹화하려는 경우, 클러스터링은 패턴을 발견하는 훌륭한 방법입니다.

> 클러스터링 모델 작업을 배우는 데 도움이 되는 유용한 로우코드 도구가 있습니다. [Azure ML을 사용해 이 작업을 시도해 보세요](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## 강의

1. [클러스터링 소개](1-Visualize/README.md)
2. [K-Means 클러스터링](2-K-Means/README.md)

## 크레딧

이 강의는 [Jen Looper](https://www.twitter.com/jenlooper)가 🎶와 함께 작성했으며, [Rishit Dagli](https://rishit_dagli)와 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)의 유용한 리뷰를 통해 완성되었습니다.

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 데이터셋은 Kaggle에서 Spotify에서 수집된 데이터를 기반으로 제공되었습니다.

이 강의를 작성하는 데 도움을 준 유용한 K-Means 예제로는 [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [입문 노트북](https://www.kaggle.com/prashant111/k-means-clustering-with-python), 그리고 [가상의 NGO 예제](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)가 있습니다.

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생할 수 있는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  