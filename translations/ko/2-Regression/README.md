# 머신 러닝을 위한 회귀 모델
## 지역 주제: 북미의 호박 가격을 위한 회귀 모델 🎃

북미에서는 호박을 할로윈 때 무서운 얼굴로 조각하는 경우가 많습니다. 이 매력적인 채소에 대해 더 알아봅시다!

![jack-o-lanterns](../../../translated_images/jack-o-lanterns.181c661a9212457d7756f37219f660f1358af27554d856e5a991f16b4e15337c.ko.jpg)
> 사진 출처: <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> on <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## 학습 내용

[![회귀 소개](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Regression Introduction video - Click to Watch!")
> 🎥 위 이미지를 클릭하면 이 강의의 간단한 소개 영상을 볼 수 있습니다

이 섹션의 강의는 머신 러닝의 맥락에서 회귀의 유형을 다룹니다. 회귀 모델은 변수 간의 _관계_를 결정하는 데 도움을 줄 수 있습니다. 이 유형의 모델은 길이, 온도, 나이와 같은 값을 예측할 수 있으며, 데이터 포인트를 분석하면서 변수 간의 관계를 밝혀냅니다.

이 시리즈의 강의에서는 선형 회귀와 로지스틱 회귀의 차이점과 어느 상황에서 어느 것을 선호해야 하는지 알아볼 것입니다.

[![ML for beginners - 머신 러닝을 위한 회귀 모델 소개](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML for beginners - Introduction to Regression models for Machine Learning")

> 🎥 위 이미지를 클릭하면 회귀 모델을 소개하는 짧은 영상을 볼 수 있습니다.

이 강의 그룹에서는 머신 러닝 작업을 시작하기 위한 설정 방법, 특히 데이터 과학자들이 주로 사용하는 노트북을 관리하기 위한 Visual Studio Code 설정 방법을 배웁니다. Scikit-learn이라는 머신 러닝 라이브러리를 발견하고, 이 장에서는 회귀 모델에 초점을 맞춘 첫 번째 모델을 구축할 것입니다.

> 회귀 모델 작업을 배우는 데 도움이 되는 유용한 로우코드 도구들이 있습니다. [Azure ML을 사용해 보세요](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### 강의 목록

1. [필수 도구들](1-Tools/README.md)
2. [데이터 관리](2-Data/README.md)
3. [선형 및 다항 회귀](3-Linear/README.md)
4. [로지스틱 회귀](4-Logistic/README.md)

---
### 크레딧

"회귀를 이용한 머신 러닝"은 [Jen Looper](https://twitter.com/jenlooper)가 ♥️를 담아 작성했습니다.

♥️ 퀴즈 기여자들: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 및 [Ornella Altunyan](https://twitter.com/ornelladotcom)

호박 데이터셋은 [Kaggle의 이 프로젝트](https://www.kaggle.com/usda/a-year-of-pumpkin-prices)에서 제안되었으며, 데이터는 미국 농무부에서 배포한 [특수 작물 터미널 시장 표준 보고서](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)에서 가져왔습니다. 우리는 분포를 정규화하기 위해 품종에 따라 색상에 대한 몇 가지 포인트를 추가했습니다. 이 데이터는 공공 도메인에 있습니다.

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 우리는 정확성을 위해 노력하지만, 자동 번역에는 오류나 부정확성이 포함될 수 있음을 유의하시기 바랍니다. 원본 문서의 원어를 권위 있는 출처로 간주해야 합니다. 중요한 정보에 대해서는 전문적인 인간 번역을 권장합니다. 이 번역의 사용으로 인해 발생하는 오해나 오역에 대해 당사는 책임을 지지 않습니다.