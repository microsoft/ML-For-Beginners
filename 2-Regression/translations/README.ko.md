# 머신러닝을 위한 Regression 모델

## 지역 토픽: 북미의 호박 가격을 위한 Regression 모델 🎃

북미에서, Halloween을 위해서 호박을 소름돋게 무서운 얼굴로 조각하는 경우가 자주 있습니다. 매혹적인 채소에 대하여 찾아봅시다!

![jack-o-lanterns](../images/jack-o-lanterns.jpg)
> Photo by <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> on <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## 무엇을 배우나요

이 섹션의 강의는 머신러닝의 컨텍스트에서 regression 타입을 다루게 됩니다. Regression 모델은 변수 사이 _relationship_ 을 결정하도록 도울 수 있습니다. 모델의 타입은 길이, 온도, 또는 나이와 같은 값을 예측할 수 있으므로, 데이터 포인트를 분석하는 순간 변수 사이의 관계를 알 수 있습니다.

이 강의의 시리즈에서, linear와 logistic regression 간의 다른 점을 찾을 수 있고, 둘 중 하나를 언제 사용해야 될 지 알 수 있습니다.

이 강의의 그룹에서, 데이터 사이언티스트를 위한 일반적 환경의, 노트북을 관리할 Visual Studio code 구성을 포함해서, 머신러닝 작업을 시작하도록 맞춥니다. 머신러닝을 위한 라이브러리인, Scikit-learn을 찾고, 이 챕터의 Regression 모델에 초점을 맞추어, 첫 모델을 만들 예정입니다.

> Regression 모델을 작업할 때 배울 수 있는 유용한 low-code 도구가 있습니다. [Azure ML for this task](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)를 시도해보세요.

### Lessons

1. [무역의 도구](../1-Tools/translations/README.ko.md)
2. [데이터 관리](../2-Data/translations/README.ko.md)
3. [Linear와 polynomial regression](../3-Linear/translations/README.ko.md)
4. [Logistic regression](../4-Logistic/translations/README.ko.md)

---
### 크레딧

"ML with regression" was written with ♥️ by [Jen Looper](https://twitter.com/jenlooper)

♥️ Quiz contributors include: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) and [Ornella Altunyan](https://twitter.com/ornelladotcom)

호박 데이터셋은 [this project on Kaggle](https://www.kaggle.com/usda/a-year-of-pumpkin-prices)에서 제안되었고 데이터는 United States Department of Agriculture에서 배포한 [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)에 기반합니다. 분포를 정규화하기 위하여 다양성을 기반으로 색상을 주변에 몇 포인트 더 했습니다. 데이터는 공개 도메인에 존재합니다.
 
