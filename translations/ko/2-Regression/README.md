<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-03T22:15:17+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "ko"
}
-->
# 머신 러닝을 위한 회귀 모델
## 지역 주제: 북미 지역 호박 가격을 위한 회귀 모델 🎃

북미에서는 호박을 종종 할로윈을 위해 무서운 얼굴로 조각합니다. 이 매력적인 채소에 대해 더 알아봅시다!

![jack-o-lanterns](../../../translated_images/jack-o-lanterns.181c661a9212457d7756f37219f660f1358af27554d856e5a991f16b4e15337c.ko.jpg)
> 사진 제공: <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> on <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## 학습 내용

[![회귀 소개](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "회귀 소개 영상 - 클릭하여 시청하세요!")
> 🎥 위 이미지를 클릭하면 이 레슨에 대한 간단한 소개 영상을 볼 수 있습니다.

이 섹션의 레슨에서는 머신 러닝의 맥락에서 다양한 회귀 유형을 다룹니다. 회귀 모델은 변수 간의 _관계_를 결정하는 데 도움을 줄 수 있습니다. 이러한 유형의 모델은 길이, 온도, 나이와 같은 값을 예측할 수 있으며, 데이터 포인트를 분석하면서 변수 간의 관계를 밝혀냅니다.

이 레슨 시리즈에서는 선형 회귀와 로지스틱 회귀의 차이점을 발견하고, 언제 각각을 사용하는 것이 더 적합한지 배울 수 있습니다.

[![초보자를 위한 머신 러닝 - 회귀 모델 소개](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "초보자를 위한 머신 러닝 - 회귀 모델 소개")

> 🎥 위 이미지를 클릭하면 회귀 모델에 대한 짧은 소개 영상을 볼 수 있습니다.

이 레슨 그룹에서는 머신 러닝 작업을 시작하기 위한 준비를 하게 됩니다. 여기에는 데이터 과학자들이 자주 사용하는 환경인 노트북을 관리하기 위해 Visual Studio Code를 설정하는 과정이 포함됩니다. 또한 Scikit-learn이라는 머신 러닝 라이브러리를 발견하고, 이 챕터에서 회귀 모델에 초점을 맞춘 첫 번째 모델을 구축하게 됩니다.

> 회귀 모델 작업을 배우는 데 도움이 되는 유용한 로우코드 도구가 있습니다. [Azure ML을 사용해 보세요](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### 레슨

1. [필수 도구](1-Tools/README.md)
2. [데이터 관리](2-Data/README.md)
3. [선형 및 다항 회귀](3-Linear/README.md)
4. [로지스틱 회귀](4-Logistic/README.md)

---
### 크레딧

"회귀를 활용한 머신 러닝"은 [Jen Looper](https://twitter.com/jenlooper)가 ♥️를 담아 작성했습니다.

♥️ 퀴즈 기여자: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 및 [Ornella Altunyan](https://twitter.com/ornelladotcom)

호박 데이터셋은 [이 Kaggle 프로젝트](https://www.kaggle.com/usda/a-year-of-pumpkin-prices)에서 제안되었으며, 데이터는 미국 농무부가 배포한 [특수 작물 터미널 시장 표준 보고서](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)에서 가져왔습니다. 우리는 분포를 정규화하기 위해 품종에 따라 색상 관련 데이터를 추가했습니다. 이 데이터는 퍼블릭 도메인에 속합니다.

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전이 권위 있는 출처로 간주되어야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 책임을 지지 않습니다.