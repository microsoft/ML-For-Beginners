# 머신러닝 소개

[![ML, AI, deep learning - What's the difference?](https://img.youtube.com/vi/lTd9RSxS9ZE/0.jpg)](https://youtu.be/lTd9RSxS9ZE "ML, AI, deep learning - What's the difference?")

> 🎥 머신러닝, AI 그리고 딥러닝의 차이를 설명하는 영상을 보려면 위 이미지를 클릭합니다.

## [강의 전 퀴즈](https://white-water-09ec41f0f.azurestaticapps.net/quiz/1/)

### 소개

입문자를 위한 classical 머신러닝 코스에 오신 것을 환영합니다! 이 토픽에 완벽하게 새로 접해보거나, 한 분야에 완벽해지고 싶어하는 ML 실무자도 저희와 함께하게 되면 좋습니다! ML 연구를 위한 친숙한 시작점을 만들고 싶고, 당신의 [feedback](https://github.com/microsoft/ML-For-Beginners/discussions)을 평가, 응답하고 반영하겠습니다.

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 동영상을 보려면 위 이미지 클릭: MIT의 John Guttag가 머신러닝을 소개합니다.
### 머신러닝 시작하기

이 커리큘럼을 시작하기 전, 컴퓨터를 세팅하고 노트북을 로컬에서 실행할 수 있게 준비해야 합니다.

- **이 영상으로 컴퓨터 세팅하기**. [set of videos](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6)에서 컴퓨터를 세팅하는 방법에 대하여 자세히 알아봅니다.
- **Python 배우기**. 이 코스에서 사용할 데이터 사이언티스트에게 유용한 프로그래밍 언어인 [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-15963-cxa)에 대한 기본적인 이해를 해야 좋습니다.
- **Node.js 와 JavaScript 배우기**. 이 코스에서 웹앱을 빌드할 때 몇 번 JavaScript를 사용하므로, [node](https://nodejs.org) 와 [npm](https://www.npmjs.com/)을 설치해야 합니다, Python 과 JavaScript를 개발하며 모두 쓸 수 있는 [Visual Studio Code](https://code.visualstudio.com/)도 있습니다.
- **GitHub 계정 만들기**. [GitHub](https://github.com)에서 찾았으므로, 이미 계정이 있을 수 있습니다, 혹시 없다면, 계정을 만든 뒤에 이 커리큘럼을 포크해서 직접 쓸 수 있습니다. (star 주셔도 됩니다 😊)
- **Scikit-learn 찾아보기**. 이 강의에서 참조하고 있는 ML 라이브러리 셋인 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)을 숙지합니다.

### 머신러닝은 무엇인가요?

'머신러닝'은 최근 가장 인기있고 자주 언급되는 용어입니다. 어떤 분야든 기술에 어느 정도 익숙해지면 이러한 용어를 한 번즈음 들어본 적이 있었을 것입니다, 그러나, 머신러닝의 구조는 대부분의 사람들에겐 미스테리입니다. 머신러닝 입문자에겐 주제가 때때로 숨막힐 수 있습니다, 그래서 머신러닝이 실제로 어떤 지 이해하고, 실제 적용된 예시로, 단계별 학습을 진행하는 것이 중요합니다.

![ml hype curve](../images/hype.png)

> Google Trends는 '머신러닝' 용어의 최근 'hype curve'로 보여줍니다

우리는 매우 신비한 우주에 살고 있습니다. Stephen Hawking, Albert Einstein과 같은 위대한 과학자들은 주변 세계의 신비를 밝혀낼 의미있는 정보를 찾는 데 일생을 바쳤습니다. 이건 사람의 학습 조건입니다: 아이는 자라면서 해마다 새로운 것을 배우고 세계 구조를 발견합니다.

어린이의 뇌와 센스는 주변의 사실을 인식하고 점차 숨겨진 생활 패턴을 학습하여 학습된 패턴을 식별할 논리 규칙을 만드는 데 도움을 줍니다. 뇌의 논리 프로세스는 사람을 가장 정교한 생명체로 만듭니다. 숨겨진 패턴을 발견하고 개선하여 지속해서 학습하면 평생 발전할 수 있습니다. 이런 학습 능력과 진화력은 [brain plasticity](https://www.simplypsychology.org/brain-plasticity.html)로 불리는 컨셉과 관련있습니다. 표면적으로, 뇌의 학습 과정과 머신러닝의 개념 사이에 motivational similarities를 그릴 수 있습니다.

[human brain](https://www.livescience.com/29365-human-brain.html)은 실제 세계에서 사물을 인식하고, 인식된 정보를 처리하며, 합리적인 결정과, 상황에 따른 행동을 합니다. 이걸 지능적으로 행동한다고 합니다. 기계에 지능적인 행동 복사본를 프로그래밍할 때, 인공 지능 (AI)라고 부릅니다.

용어가 햇갈릴 수 있지만, 머신러닝(ML)은 중요한 인공 지능의 서브넷입니다. **ML은 특수한 알고리즘을 써서 의미있는 정보를 찾고 인식한 데이터에서 숨겨진 패턴을 찾아 합리적으로 판단할 프로세스를 확실하게 수행하는 것에 관심있습니다**.

![AI, ML, deep learning, data science](../images/ai-ml-ds.png)

> AI, ML, 딥러닝, 그리고 데이터 사이언티스트 간의 관계를 보여주는 다이어그램. [this graphic](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)에서 영감을 받은 [Jen Looper](https://twitter.com/jenlooper)의 인포그래픽

## 이 코스에서 배우는 것

이 커리큘럼에서, 입문자가 반드시 알아야 할 머신러닝의 핵심적인 개념만 다룰 것입니다. 많은 학생들이 기초를 배우기 위해 사용하는 훌륭한 라이브러리인, Scikit-learn으로 'classical machine learning'이라고 부르는 것을 다룹니다. 인공 지능 또는 딥러닝의 대략적인 개념을 이해하려면, 머신러닝에 대한 강력한 기초 지식이 꼭 필요하므로, 여기에서 제공하고자 합니다.

이 코스에서 다음 사항을 배웁니다:

- 머신러닝의 핵심 컨셉
- ML 의 역사
- ML 과 공정성
- regression ML 기술
- classification ML 기술
- clustering ML 기술
- natural language processing ML 기술
- time series forecasting ML 기술
- 강화 학습
- real-world 애플리케이션 for ML

## 다루지 않는 것

- 딥러닝
- 신경망
- AI
  
더 좋은 학습 환경을 만들기 위해서, 신경망, '딥러닝' - many-layered model-building using neural networks - 과 AI의 복잡도를 피할 것이며, 다른 커리큘럼에서 논의할 것입니다. 또한 더 큰 필드에 초점을 맞추기 위하여 향후 데이터 사이언스 커리큘럼을 제공할 예정입니다. 
## 왜 머신러닝을 배우나요?

시스템 관점에서 보는 머신러닝은, 지능적인 결정하도록 데이터에서 숨겨진 패턴을 학습할 수 있는 자동화 시스템 생성으로 정의합니다.

동기 부여는 뇌가 다른 세계에서 보는 데이터를 기반으로 특정한 무언가들을 학습하는 방식에서 살짝 영감을 받았습니다.

✅  비지니스에서 머신러닝 전략 대신 하드-코딩된 룰-베이스 엔진을 만드려는 이유를 잠시 생각해봅시다.

### 머신러닝의 애플리케이션

머신러닝의 애플리케이션은 이제 거의 모든 곳에서, 스마트 폰, 연결된 기기, 그리고 다른 시스템에 의하여 생성된 주변의 흐르는 데이터만큼 어디에나 존재합니다. 첨단 머신러닝 알고리즘의 큰 잠재력을 고려한, 연구원들은 긍정적인 결과로 multi-dimensional과 multi-disciplinary적인 실-생활 문제를 해결하는 능력을 찾고 있습니다.

**다양한 방식으로 머신러닝을 사용할 수 있습니다**:

- 환자의 병력이나 보고서를 기반으로 질병 가능성을 예측합니다.
- 날씨 데이터로 계절 이벤트를 예측합니다.
- 문장의 감정을 이해합니다.
- 가짜 뉴스를 감지하고 선동을 막습니다.

금융, 경제학, 지구 과학, 우주 탐험, 생물 공학, 인지 과학, 그리고 인문학까지 머신러닝을 적용하여 힘들고, 데이터 처리가 버거운 이슈를 해결했습니다.

머신러닝은 실제-환경이거나 생성된 데이터에서 의미를 찾아 패턴-발견하는 프로세스를 자동화합니다. 비즈니스, 건강과 금용 애플리케이션에서 높은 가치가 있다고 증명되었습니다.

가까운 미래에, 머신러닝의 기본을 이해하는 건 광범위한 선택으로 인하여 모든 분야의 사람들에게 필수적으로 다가올 것 입니다.

---
## 🚀 도전

종이에 그리거나, [Excalidraw](https://excalidraw.com/)처럼 온라인 앱을 이용하여 AI, ML, 딥러닝, 그리고 데이터 사이언스의 차이를 이해합시다. 각 기술들이 잘 해결할 수 있는 문제에 대해 아이디어를 합쳐보세요.

## [강의 후 퀴즈](https://white-water-09ec41f0f.azurestaticapps.net/quiz/2/)

## 리뷰 & 자기주도 학습

클라우드에서 ML 알고리즘을 어떻게 사용하는 지 자세히 알아보려면, [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-15963-cxa)를 따릅니다.

ML의 기초에 대한 [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-15963-cxa)를 봅니다.

## 과제

[Get up and running](../assignment.md)
