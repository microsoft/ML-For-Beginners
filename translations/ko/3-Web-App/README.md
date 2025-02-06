# 웹 앱을 구축하여 ML 모델 사용하기

이 커리큘럼의 이 섹션에서는 적용된 머신러닝 주제에 대해 소개합니다: Scikit-learn 모델을 파일로 저장하여 웹 애플리케이션 내에서 예측에 사용할 수 있는 방법입니다. 모델을 저장한 후에는 Flask로 구축된 웹 앱에서 이를 사용하는 방법을 배우게 됩니다. 먼저 UFO 목격에 관한 데이터를 사용하여 모델을 생성합니다! 그런 다음, 위도와 경도 값을 입력하여 어느 나라에서 UFO를 목격했는지 예측할 수 있는 웹 앱을 구축합니다.

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.ko.jpg)

<a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a>의 사진, <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>에서 제공

## 레슨

1. [웹 앱 구축하기](1-Web-App/README.md)

## 크레딧

"웹 앱 구축하기"는 [Jen Looper](https://twitter.com/jenlooper)의 ♥️와 함께 작성되었습니다.

♥️ 퀴즈는 Rohan Raj가 작성했습니다.

데이터셋은 [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)에서 제공되었습니다.

웹 앱 아키텍처는 부분적으로 [이 기사](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4)와 Abhinav Sagar의 [이 저장소](https://github.com/abhinavsagar/machine-learning-deployment)의 제안을 참고했습니다.

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 정확성을 위해 노력하지만, 자동 번역에는 오류나 부정확성이 있을 수 있습니다. 원본 문서의 모국어 버전이 권위 있는 소스로 간주되어야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.