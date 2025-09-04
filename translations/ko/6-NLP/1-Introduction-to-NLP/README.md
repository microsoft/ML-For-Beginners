<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "89e923cf3e8bdff9662536e8bf9516e6",
  "translation_date": "2025-09-04T00:46:48+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "ko"
}
-->
# 자연어 처리 소개

이 강의에서는 *계산 언어학*의 하위 분야인 *자연어 처리*의 간략한 역사와 중요한 개념을 다룹니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## 소개

일반적으로 NLP라고 알려진 자연어 처리는 머신 러닝이 적용되고 실제 소프트웨어에서 사용된 가장 잘 알려진 분야 중 하나입니다.

✅ 매일 사용하는 소프트웨어 중 일부에 NLP가 포함되어 있을 가능성이 있는 것을 생각해볼 수 있나요? 워드 프로세싱 프로그램이나 자주 사용하는 모바일 앱은 어떨까요?

여러분은 다음을 배우게 됩니다:

- **언어의 개념**. 언어가 어떻게 발전했는지와 주요 연구 분야에 대해 알아봅니다.
- **정의와 개념**. 컴퓨터가 텍스트를 처리하는 방법, 구문 분석, 문법, 명사와 동사를 식별하는 방법에 대한 정의와 개념을 배우게 됩니다. 이 강의에서는 몇 가지 코딩 작업이 포함되어 있으며, 이후 강의에서 코딩을 배우게 될 중요한 개념들이 소개됩니다.

## 계산 언어학

계산 언어학은 수십 년 동안 컴퓨터가 언어를 작업하고, 이해하고, 번역하고, 심지어 소통할 수 있는 방법을 연구하는 분야입니다. 자연어 처리(NLP)는 컴퓨터가 '자연어', 즉 인간의 언어를 처리하는 방법에 초점을 맞춘 관련 분야입니다.

### 예시 - 전화 음성 입력

전화로 타이핑 대신 음성을 입력하거나 가상 비서에게 질문을 한 적이 있다면, 여러분의 음성은 텍스트 형태로 변환된 후 여러분이 말한 언어에서 *구문 분석*됩니다. 감지된 키워드는 전화나 비서가 이해하고 실행할 수 있는 형식으로 처리됩니다.

![이해](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.ko.png)
> 실제 언어적 이해는 어렵습니다! 이미지 제공: [Jen Looper](https://twitter.com/jenlooper)

### 이 기술은 어떻게 가능할까요?

이 기술은 누군가가 이를 수행하는 컴퓨터 프로그램을 작성했기 때문에 가능합니다. 몇십 년 전, 일부 공상 과학 작가들은 사람들이 주로 컴퓨터와 대화하고 컴퓨터가 항상 그들이 의미하는 바를 정확히 이해할 것이라고 예측했습니다. 안타깝게도 이는 많은 사람들이 상상했던 것보다 더 어려운 문제로 밝혀졌으며, 오늘날에는 훨씬 더 잘 이해되는 문제이지만 문장의 의미를 이해하는 '완벽한' 자연어 처리를 달성하는 데는 여전히 상당한 어려움이 있습니다. 특히 유머를 이해하거나 문장에서 풍자와 같은 감정을 감지하는 것은 매우 어려운 문제입니다.

이 시점에서 학교 수업에서 문장의 문법 부분을 배운 기억이 떠오를 수도 있습니다. 일부 국가에서는 학생들이 문법과 언어학을 전용 과목으로 배우지만, 많은 국가에서는 이러한 주제가 언어 학습의 일부로 포함됩니다. 이는 초등학교에서 첫 번째 언어를 배우는 과정(읽고 쓰는 법 배우기)과 중등학교 또는 고등학교에서 두 번째 언어를 배우는 과정에서 이루어질 수 있습니다. 명사와 동사 또는 부사와 형용사를 구분하는 데 전문가가 아니더라도 걱정하지 마세요!

*단순 현재 시제*와 *현재 진행형*의 차이를 구분하는 데 어려움을 겪는다면, 여러분은 혼자가 아닙니다. 이는 많은 사람들에게, 심지어 모국어를 사용하는 사람들에게도 어려운 일입니다. 좋은 소식은 컴퓨터가 공식적인 규칙을 적용하는 데 매우 능숙하다는 것이며, 여러분은 문장을 인간처럼 *구문 분석*할 수 있는 코드를 작성하는 방법을 배우게 될 것입니다. 이후에 살펴볼 더 큰 도전은 문장의 *의미*와 *감정*을 이해하는 것입니다.

## 사전 요구 사항

이 강의를 위해 주요 요구 사항은 이 강의의 언어를 읽고 이해할 수 있는 능력입니다. 수학 문제나 방정식을 풀 필요는 없습니다. 원래 저자가 이 강의를 영어로 작성했지만, 다른 언어로 번역되었을 수도 있으므로 번역본을 읽고 있을 가능성도 있습니다. 여러 언어의 문법 규칙을 비교하기 위해 다양한 언어가 사용된 예제가 있습니다. 이러한 예제는 번역되지 않지만, 설명 텍스트는 번역되므로 의미는 명확할 것입니다.

코딩 작업을 위해 Python을 사용하며, 예제는 Python 3.8을 사용합니다.

이 섹션에서 필요하고 사용할 내용은 다음과 같습니다:

- **Python 3 이해**. Python 3 프로그래밍 언어 이해, 이 강의에서는 입력, 루프, 파일 읽기, 배열을 사용합니다.
- **Visual Studio Code + 확장 프로그램**. Visual Studio Code와 Python 확장 프로그램을 사용합니다. 원하는 Python IDE를 사용할 수도 있습니다.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob)은 Python용 간단한 텍스트 처리 라이브러리입니다. TextBlob 사이트의 지침에 따라 시스템에 설치하세요(아래에 표시된 대로 corpora도 설치하세요):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 팁: VS Code 환경에서 직접 Python을 실행할 수 있습니다. 자세한 내용은 [문서](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott)를 확인하세요.

## 기계와 대화하기

컴퓨터가 인간의 언어를 이해하도록 만드는 시도의 역사는 수십 년 전으로 거슬러 올라가며, 자연어 처리를 고려한 초기 과학자 중 한 명은 *앨런 튜링*입니다.

### '튜링 테스트'

튜링은 1950년대에 *인공지능*을 연구하면서, 인간과 컴퓨터가 (타이핑된 대화로) 대화를 나누는 테스트를 제안했습니다. 이 테스트에서 인간은 자신이 대화하고 있는 상대가 인간인지 컴퓨터인지 확신할 수 없게 됩니다.

만약 일정 시간 동안의 대화 후에 인간이 상대가 컴퓨터인지 아닌지를 판단할 수 없다면, 컴퓨터가 *생각*한다고 말할 수 있을까요?

### 영감 - '모방 게임'

이 아이디어는 *모방 게임*이라는 파티 게임에서 영감을 받았습니다. 이 게임에서 조사자는 방에 혼자 있고, 다른 방에 있는 두 사람 중 각각 남성과 여성의 성별을 결정해야 합니다. 조사자는 메모를 보낼 수 있으며, 작성된 답변이 신비한 사람의 성별을 드러내는 질문을 생각해야 합니다. 물론, 다른 방에 있는 플레이어들은 정직하게 답변하는 것처럼 보이면서도 조사자를 혼란스럽게 하거나 오도하려고 노력합니다.

### 엘리자 개발

1960년대 MIT의 과학자인 *조셉 와이젠바움*은 [*엘리자*](https://wikipedia.org/wiki/ELIZA)라는 컴퓨터 '치료사'를 개발했습니다. 엘리자는 인간에게 질문을 하고 인간의 답변을 이해하는 것처럼 보였습니다. 그러나 엘리자는 문장을 구문 분석하고 특정 문법 구조와 키워드를 식별하여 합리적인 답변을 제공할 수 있었지만, 문장을 *이해*한다고 말할 수는 없었습니다. 예를 들어, 엘리자에게 "**나는** <u>슬프다</u>"라는 형식의 문장이 주어지면, 문장의 시제를 변경하고 몇 가지 단어를 추가하여 "얼마나 오랫동안 **당신은** <u>슬펐나요</u>"라는 답변을 생성할 수 있었습니다.

이 답변은 엘리자가 문장을 이해하고 후속 질문을 하고 있는 것처럼 보이게 했지만, 실제로는 시제를 변경하고 몇 가지 단어를 추가한 것뿐이었습니다. 엘리자가 응답할 키워드를 식별할 수 없으면, 대신 여러 다른 문장에 적용될 수 있는 무작위 응답을 제공했습니다. 예를 들어, 사용자가 "**당신은** <u>자전거</u>입니다"라고 작성하면, 엘리자는 "얼마나 오랫동안 **나는** <u>자전거</u>였나요?"라고 응답할 수 있었으며, 더 합리적인 응답을 제공하지 못했습니다.

[![엘리자와 대화하기](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "엘리자와 대화하기")

> 🎥 위 이미지를 클릭하면 원래 엘리자 프로그램에 대한 비디오를 볼 수 있습니다.

> 참고: [엘리자](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)의 원래 설명을 1966년에 발표된 ACM 계정으로 읽을 수 있습니다. 또는 [위키피디아](https://wikipedia.org/wiki/ELIZA)에서 엘리자에 대해 읽어보세요.

## 연습 - 기본 대화형 봇 코딩하기

엘리자와 같은 대화형 봇은 사용자 입력을 유도하고 이해하고 지능적으로 응답하는 것처럼 보이는 프로그램입니다. 엘리자와 달리, 우리의 봇은 지능적인 대화를 하는 것처럼 보이게 하는 여러 규칙을 갖고 있지는 않을 것입니다. 대신, 우리의 봇은 단순한 대화에서 거의 모든 경우에 작동할 수 있는 무작위 응답을 통해 대화를 계속 이어가는 한 가지 능력만 가질 것입니다.

### 계획

대화형 봇을 구축할 때의 단계:

1. 사용자에게 봇과 상호작용하는 방법에 대한 지침을 출력합니다.
2. 루프를 시작합니다.
   1. 사용자 입력을 받습니다.
   2. 사용자가 종료를 요청했는지 확인하고, 요청했다면 종료합니다.
   3. 사용자 입력을 처리하고 응답을 결정합니다(이 경우 응답은 가능한 일반적인 응답 목록에서 무작위로 선택됩니다).
   4. 응답을 출력합니다.
3. 2단계로 다시 돌아갑니다.

### 봇 만들기

이제 봇을 만들어봅시다. 몇 가지 문구를 정의하는 것으로 시작하겠습니다.

1. 다음과 같은 무작위 응답을 사용하는 Python에서 직접 이 봇을 만들어보세요:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    아래는 샘플 출력입니다(사용자 입력은 `>`로 시작하는 줄에 표시됩니다):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    작업에 대한 하나의 가능한 솔루션은 [여기](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)에 있습니다.

    ✅ 멈추고 생각해보기

    1. 무작위 응답이 봇이 실제로 사용자를 이해한다고 생각하게 만들 수 있을까요?
    2. 봇이 더 효과적이 되기 위해 필요한 기능은 무엇일까요?
    3. 봇이 문장의 의미를 실제로 '이해'할 수 있다면, 대화에서 이전 문장의 의미를 '기억'해야 할까요?

---

## 🚀도전

위의 "멈추고 생각해보기" 요소 중 하나를 선택하여 코드로 구현하거나 의사 코드로 종이에 솔루션을 작성해보세요.

다음 강의에서는 자연어를 구문 분석하고 머신 러닝을 사용하는 여러 접근법에 대해 배우게 될 것입니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## 복습 및 자기 학습

아래 참고 자료를 살펴보며 추가 학습 기회를 가져보세요.

### 참고 자료

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 과제 

[봇 검색하기](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  