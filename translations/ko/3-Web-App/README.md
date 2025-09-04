<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T23:43:15+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "ko"
}
-->
# ML 모델을 활용한 웹 앱 만들기

이 커리큘럼 섹션에서는 Scikit-learn 모델을 파일로 저장하여 웹 애플리케이션 내에서 예측에 사용할 수 있는 방법을 배우는 실용적인 머신러닝 주제를 소개합니다. 모델을 저장한 후에는 Flask로 구축된 웹 앱에서 이를 사용하는 방법을 배우게 됩니다. 먼저 UFO 목격 데이터를 사용하여 모델을 생성합니다! 그런 다음, 특정 시간(초)과 위도 및 경도 값을 입력하여 어느 국가에서 UFO를 목격했는지 예측할 수 있는 웹 앱을 구축합니다.

![UFO 주차장](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.ko.jpg)

사진 제공: <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> on <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## 강의 목록

1. [웹 앱 만들기](1-Web-App/README.md)

## 크레딧

"웹 앱 만들기"는 [Jen Looper](https://twitter.com/jenlooper)가 ♥️를 담아 작성했습니다.

♥️ 퀴즈는 Rohan Raj가 작성했습니다.

데이터셋은 [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)에서 제공되었습니다.

웹 앱 아키텍처는 [이 기사](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4)와 [이 저장소](https://github.com/abhinavsagar/machine-learning-deployment)에서 Abhinav Sagar가 제안한 내용을 일부 참고했습니다.

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전이 권위 있는 출처로 간주되어야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 책임을 지지 않습니다.