[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

# Machine Learning for Beginners (입문자를 위한 머신러닝) - 커리큘럼

> 🌍 세계의 문화로 머신러닝을 알아가면서 전 세계를 여행합니다 🌍

Microsoft의 Azure Cloud Advocates는 **Machine Learning**에 대한 모든 12-주, 24-강의 (하나 더!) 커리큘럼을 제공하게 된 것을 기쁘게 생각합니다. 이 교육 과정에서는 주로 Scikit-learn을 라이브러리로 사용하고 향후에 다룰 딥 러닝을 제외한 **classic machine learning**에 대해 배우게 됩니다. 본 수업과 '입문자를 위한 데이터 과학' 커리큘럼과 연계하여 학습해도 좋습니다.

이러한 고전적인 기술을 세계 여러 지역의 데이터에 적용하는 동안 우리와 함께 세계 여행을 떠나보십시오. 각 레슨에는 예습 및 복습 퀴즈, 레슨을 완료하기 위한 서면 지침, 해결책 및 과제가 포함됩니다. 프로젝트 기반 교육학을 통해 새로운 기술을 익힐 수 있는 검증된 방법으로 학습 할 수 있습니다.

**✍️ Hearty thanks to our authors** Jen Looper, Stephen Howell, Francesca Lazzeri, Tomomi Imura, Cassie Breviu, Dmitry Soshnikov, Chris Noring, Ornella Altunyan, and Amy Boyd

**🎨 Thanks as well to our illustrators** Tomomi Imura, Dasani Madipalli, and Jen Looper

**🙏 Special thanks 🙏 to our Microsoft Student Ambassador authors, reviewers and content contributors**, notably Rishit Dagli, Muhammad Sakib Khan Inan, Rohan Raj, Alexandru Petrescu, Abhishek Jaiswal, Nawrin Tabassum, Ioan Samuila, and Snigdha Agarwal

**🤩 Extra gratitude to Microsoft Student Ambassador Eric Wanjau for our R lessons!** 

---

# 시작하기

**학생**은, 이 커리큘럼을 사용하기 위해서, 전체 저장소를 자신의 GitHub 계정으로 포크하고 혼자 또는 그룹으로 같이 학습합니다:

- 강의 전 퀴즈를 시작합니다.
- 강의를 읽고, 각 지식 점검에서 멈추고 습득해서 활동을 끝냅니다.
- 솔루션 코드를 실행하는 것보다 강의를 이해해서 프로젝트를 만들어봅니다. 해답 코드는 각 프로젝트-지향 강의 별 `/solution` 폴더에 위치합니다.
- 강의 후 퀴즈를 해봅니다.
- 도전을 끝내봅니다.
- 과제를 끝내봅니다.
- 강의 그룹을 끝내면, [Discussion board](https://github.com/microsoft/ML-For-Beginners/discussions)를 방문하고 적절한 PAT rubric를 채워서 "learn out loud" 합니다. 'PAT'은 심화적으로 배우려고 작성하는 rubric인 Progress Assessment 도구 입니다. 같이 배울 수 있게 다른 PAT으로도 할 수 있습니다.

> 더 배우기 위해서, [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) 모듈과 학습 경로를 따르는 것을 추천합니다.

**선생님**은, 이 커리큘럼의 사용 방법에 대해 [일부 제안사항](../for-teachers.md)이 있습니다.

---

## Team 만나기

[![Promo video](../ml-for-beginners.png)](https://youtu.be/Tj1XWrDSYJU "Promo video")

> 🎥 프로젝트와 이 내용을 작성한 사람들에 대한 영상을 보려면 위 이미지를 클릭합니다!

---

## 교육학

이 커리큘럼을 만드는 동안 2가지 교육학 원칙을 선택했습니다: **project-based**에서 실습하고 **frequent quizzes**가 포함되었는지 확인합니다. 추가적으로, 이 커리큘럼은 통합적으로 보이기 위해서 공통적인 **theme**가 있습니다.

컨텐츠가 프로젝트와 맞게 유지되므로, 프로세스는 학생들이 더 끌리고 개념의 집중도가 높아집니다. 추가적으로, 강의 전 가벼운 퀴즈는 학생들이 공부에 집중하게 해주고, 강의 후 두 번째 퀴즈는 계속 집중하게 합니다. 이 커리큘럼은 유연하고 재밌게 디자인되었으며 다 배우거나 일부만 배울 수 있습니다. 프로젝트는 작게 시작해서 12주 사이클로 끝날 때까지 점점 복잡해집니다. 이 커리큘럼은 추가 크레딧이나 토론의 기초로 사용할 수 있는, ML의 현실에 적용한 postscript도 포함되어 있습니다.

> [Code of Conduct](../CODE_OF_CONDUCT.md), [Contributing](../CONTRIBUTING.md)과, [Translation](../TRANSLATIONS.md) 가이드라인을 확인해봅니다. 건설적인 피드백을 환영합니다!

## 각 강의에 포함된 내용:

- 취사선택 스케치노트
- 취사선택 추가 영상
- 강의 전 준비 퀴즈
- 강의 내용
- 프로젝트-기반 강의라면, 프로젝트 제작 방식 step-by-step 지도
- 지식 점검
- 도전
- 보충 내용
- 과제
- 강의 후 퀴즈

> **퀴즈 참고사항**: 모든 퀴즈는 [이 앱](https://gray-sand-07a10f403.1.azurestaticapps.net/)에 포함되어 있으며, 각각 3문제씩 총 50개의 퀴즈가 있습니다. 퀴즈 앱은 교육 과정과 연결되어 있지만, 원하는 경우 따로 퀴즈 앱을 실행할 수도 있습니다. 자세한 사항은 퀴즈 앱 폴더 내의 지침을 따르십시오.


| 강의 번호 |                           토픽                            |                   강의 그룹                   | 학습 목표                                                                                                             |                     연결 강의                     |     저자     |
| :-----------: | :--------------------------------------------------------: | :-------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------: | :------------: |
|      01       |              머신러닝 소개              |      [소개](../1-Introduction/translations/README.ko.md)       | 머신러닝의 기초 컨셉을 배웁니다                                                                                |   [강의](../1-Introduction/1-intro-to-ML/translations/README.ko.md)    |    Muhammad    |
|      02       |              머신러닝의 역사               |      [소개](../1-Introduction/translations/README.ko.md)       | 이 필드의 역사를 배웁니다                                                                                         |  [강의](../1-Introduction/2-history-of-ML/translations/README.ko.md)   |  Jen and Amy   |
|      03       |               공정과 머신러닝                |      [소개](../1-Introduction/translations/README.ko.md)       | 학생들이 ML 모델을 만들고 적용할 때 고려해야 할 공정과 관련한 중요 철학적인 이슈는 무엇인가요? |     [강의](../1-Introduction/3-fairness/translations/README.ko.md)     |     Tomomi     |
|      04       |              머신러닝의 기술               |      [소개](../1-Introduction/translations/README.ko.md)       | ML 연구원들이 ML 모델을 만들 때 사용할 기술은 무엇인가요?                                                                       | [강의](../1-Introduction/4-techniques-of-ML/translations/README.ko.md) | Chris and Jen  |
|      05       |                 regression 소개                |        [Regression](../2-Regression/translations/README.ko.md)         | regression 모델을 위한 Python과 Scikit-learn으로 시작합니다                                                                  |       [강의](../2-Regression/1-Tools/translations/README.ko.md)        |      Jen       |
|      06       |              북미의 호박 가격 🎃               |        [Regression](../2-Regression/translations/README.ko.md)         | ML을 준비하기 위해서 데이터를 시각화하고 정리합니다                                                                                  |        [강의](../2-Regression/2-Data/translations/README.ko.md)        |      Jen       |
|      07       |              북미의 호박 가격 🎃               |        [Regression](../2-Regression/translations/README.ko.md)         | linear와 polynomial regression 모델을 만듭니다                                                                                   |       [강의](2-Regression/3-Linear/translations/README.ko.md)       |      Jen       |
|      08       |              북미의 호박 가격 🎃               |        [Regression](../2-Regression/translations/README.ko.md)         | logistic regression 모델을 만듭니다                                                                                               |      [강의](../2-Regression/4-Logistic/translations/README.ko.md)      |      Jen       |
|      09       |                        웹 앱 🔌                         |           [웹 앱](../3-Web-App/translations/README.ko.md)            | 훈련된 모델로 웹 앱을 만듭니다                                                                                       |        [강의](../3-Web-App/1-Web-App/translations/README.ko.md)        |      Jen       |
|      10       |               classification 소개               |    [Classification](../4-Classification/translations/README.ko.md)     | 데이터 정리, 준비, 그리고 시각화; classification을 소개합니다                                                            |  [강의](../4-Classification/1-Introduction/translations/README.ko.md)  | Jen and Cassie |
|      11       |           맛있는 아시아와 인도 요리 🍜            |    [Classification](../4-Classification/translations/README.ko.md)     | classifier를 소개합니다                                                                                                     | [강의](../4-Classification/2-Classifiers-1/translations/README.ko.md)  | Jen and Cassie |
|      12       |           맛있는 아시아와 인도 요리 🍜            |    [Classification](../4-Classification/translations/README.ko.md)     | 더 많은 classifier                                                                                                                | [강의](../4-Classification/3-Classifiers-2/translations/README.ko.md)  | Jen and Cassie |
|      13       |           맛있는 아시아와 인도 요리 🍜            |    [Classification](../4-Classification/translations/README.ko.md)     | 모델로 추천 웹 앱을 만듭니다                                                                                    |    [강의](../4-Classification/4-Applied/translations/README.ko.md)     |      Jen       |
|      14       |                 clustering 소개                 |        [Clustering](../5-Clustering/translations/README.ko.md)         | 데이터 정리, 준비, 그리고 시각화; clustering을 소개합니다                                                                |     [강의](../5-Clustering/1-Visualize/translations/README.ko.md)      |      Jen       |
|      15       |            나이지리아인의 음악 취향 알아보기 🎧             |        [Clustering](../5-Clustering/translations/README.ko.md)         | K-Means clustering 메소드를 탐색합니다                                                                                           |      [강의](../5-Clustering/2-K-Means/translations/README.ko.md)       |      Jen       |
|      16       |       natural language processing 소개 ☕️        |   [Natural language processing](../6-NLP/translations/README.ko.md)    | 간단한 봇을 만들면서 NLP에 대하여 기본을 배웁니다                                                                             |    [강의](../6-NLP/1-Introduction-to-NLP/translations/README.ko.md)    |    Stephen     |
|      17       |                     일반적인 NLP 작업 ☕️                     |   [Natural language processing](../6-NLP/translations/README.ko.md)    | 언어 구조를 다룰 때 필요한 일반 작업을 이해하면서 NLP 지식을 깊게 팝니다                          |           [강의](../6-NLP/2-Tasks/translations/README.ko.md)           |    Stephen     |
|      18       |            번역과 감정 분석 ♥️            |   [Natural language processing](../6-NLP/translations/README.ko.md)    | Jane Austen을 통한 번역과 감정 분석                                                                             |   [강의](../6-NLP/3-Translation-Sentiment/translations/README.ko.md)   |    Stephen     |
|      19       |                유럽의 로맨틱 호텔 ♥️                 |   [Natural language processing](../6-NLP/translations/README.ko.md)    | 호텔 리뷰를 통한 감정 분석 1                                                                                         |      [강의](../6-NLP/4-Hotel-Reviews-1/translations/README.ko.md)      |    Stephen     |
|      20       |                유럽의 로맨틱 호텔 ♥️                 |   [Natural language processing](../6-NLP/translations/README.ko.md)    | 호텔 리뷰를 통한 감정 분석 2                                                                                         |      [강의](../6-NLP/5-Hotel-Reviews-2/translations/README.ko.md)      |    Stephen     |
|      21       |          time series forecasting 소개           |        [Time series](../7-TimeSeries/translations/README.ko.md)        | time series forecasting을 소개합니다                                                                                         |    [강의](../7-TimeSeries/1-Introduction/translations/README.ko.md)    |   Francesca    |
|      22       | ⚡️ 세계 전력 사용량 ⚡️ - ARIMA의 time series forecasting |        [Time series](../7-TimeSeries/translations/README.ko.md)        | ARIMA의 Time series forecasting                                                                                              |       [강의](../7-TimeSeries/2-ARIMA/translations/README.ko.md)        |   Francesca    |
|      23       |           reinforcement learning 소개           | [Reinforcement learning](../8-Reinforcement/translations/README.ko.md) | Q-Learning의 reinforcement learning을 소개합니다                                                                          |    [강의](../8-Reinforcement/1-QLearning/translations/README.ko.md)    |     Dmitry     |
|      24       |                늑대를 피하는 Peter 도와주기! 🐺                | [Reinforcement learning](../8-Reinforcement/translations/README.ko.md) | Gym에서 Reinforcement learning                                                                                                       |       [강의](../8-Reinforcement/2-Gym/translations/README.ko.md)       |     Dmitry     |
|  Postscript   |          실생활 ML 시나리오와 애플리케이션           |      [야생의 ML](../9-Real-World/translations/README.ko.md)       | classical ML의 흥미롭게 드러나는 현실 애플리케이션                                                               |    [강의](../9-Real-World/1-Applications/translations/README.ko.md)    |      Team      |

## 오프라인 액세스

[Docsify](https://docsify.js.org/#/)를 사용하여 이 문서를 오프라인으로 실행할 수 있습니다. 이 repo를 포크하여 로컬 컴퓨터에 [Docsify (설치)](https://docsify.js.org/#/quickstart)를 설치한 다음 이 repo의 루트 폴더에 'docsify serve'를 입력하면 됩니다. 웹 사이트는 로컬 호스트의 포트 3000에서 제공됩니다: 'localhost:3000'.

## PDF

[여기](../pdf/readme.pdf) 링크를 통해 커리큘럼의 PDF를 찾아보십시오.

## 도와주세요!

번역에 기여하고 싶으신가요? [translation guidelines](../TRANSLATIONS.md)를 읽고 [here](https://github.com/microsoft/ML-For-Beginners/issues/71)에 입력해주세요. 

## 기타 커리큘럼

우리 팀에서는 다른 커리큘럼도 제작합니다! 확인해보세요!

- [Web Dev for Beginners](https://aka.ms/webdev-beginners)
- [IoT for Beginners](https://aka.ms/iot-beginners)
