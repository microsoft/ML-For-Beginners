# Natural language processing 소개하기

이 강의애서 *computational linguistics* 하위인, *natural language processing*의 간단한 역사와 중요 컨셉을 다룹니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## 소개

일반적으로 알고있는, NLP는, 머신러닝이 프로덕션 소프트웨어에 적용되어 사용하는 잘-알려진 영역 중 하나입니다.

✅ 항상 사용하는 소프트웨어에서 NLP가 들어갈 수 있는지 생각할 수 있나요? 규칙적으로 사용하는 워드 프로그램이나 모바일 앱은 어떤가요?

해당 내용을 배우게 됩니다:

- **언어의 아이디어**. 언어가 어떻게 발전했고 어떤 주요 연구 영역인가요?
- **정의와 컨셉**. 또한 파싱, 문법, 그리고 명사와 동사를 식별하는 것을 합쳐서, 컴퓨터가 텍스트를 처리하는 방식에 대한 정의와 개념을 배우게 됩니다. 이 강의에서 약간의 코딩 작업을 하며, 다음 강의 뒤에 배울 코드에서 중요한 개념을 소개합니다.

## 전산 언어학

전산 언어학은 컴퓨터가 언어와 합쳐서 이해, 번역, 그리고 커뮤니케이션 방식을 연구하는 수십 년을 넘어 연구 개발하고 있는 영역입니다. natural language processing (NLP)은 컴퓨터가 인간 언어를, 'natural'하게, 처리할 수 있는 것에 초점을 맞춘 관련 필드입니다.

### 예시 - 전화번호 받아쓰기

만약 핸드폰에 타이핑하거나 가상 어시스턴트에 질문을 했다면, 음성은 텍스트 형태로 변환되고 언급한 언어에서 처리되거나 *파싱*됩니다. 감지된 키워드는 핸드폰이나 어시스턴트가 이해하고 행동할 수 있는 포맷으로 처리됩니다.

![comprehension](../images/comprehension.png)
> Real linguistic comprehension is hard! Image by [Jen Looper](https://twitter.com/jenlooper)

### 이 기술은 어떻게 만들어지나요?

누군가 컴퓨터 프로그램을 작성했기 때문에 가능합니다. 수십 년 전에, 과학소설 작가는 사람들이 컴퓨터와 이야기하며, 컴퓨터가 그 의미를 항상 정확히 이해할 것이라고 예측했습니다. 슬프게, 많은 사람들이 상상했던 내용보다 더 어려운 문제로 밝혀졌고, 이제 더 잘 이해되는 문제이지만, 문장 의미를 이해함에 있어서 'perfect'한 natural language processing을 성공하기에는 상당히 어렵습니다. 유머를 이해하거나 문장에서 풍자처럼 감정을 알아차릴 때 특히 어렵습니다.

이 포인트에서, 학교 수업에서 선생님이 문장의 문법 파트를 가르쳤던 기억을 회상할 수 있습니다. 일부 국가에서는, 학생에게 문법과 언어학을 전공 과목으로 가르치지만, 많은 곳에서, 이 주제를 언어 학습의 일부로 합칩니다: 초등학교에서 모국어(읽고 쓰는 방식 배우기)와 중학교나, 고등학교에서 제2 외국어를 배울 수 있습니다. 만약 명사와 형용사 또는 부사를 구분하는 전문가가 아니라고 해도 걱정하지 맙시다!

만약 *simple present*와 *present progressive* 사이에서 몸부림치면, 혼자가 아닙니다. 모국어를 언어로 쓰는, 많은 사람들에게 도전입니다. 좋은 소식은 컴퓨터가 형식적인 규칙을 적용하는 것은 매우 좋고, 사람말고 문장을 *parse*할 수 있는 코드로 작성하는 방식을 배우게 됩니다. 나중에 할 큰 도전은 문장의 *meaning*과, *sentiment*를 이해하는 것입니다. 

## 전제 조건

이 강의에서, 주요 전제 조건은 이 강의의 언어를 읽고 이해해야 합니다. 풀 수 있는 수학 문제나 방정식이 아닙니다. 원작자가 영어로 이 강의를 작성했지만, 다른 언어로 번역되었으므로, 번역본으로 읽을 수 있게 되었습니다. 다른 언어로 사용된 (다른 언어의 다른 문법을 비교하는) 예시가 있습니다. 번역을 *하지 않았어도*, 설명 텍스트는, 의미가 명확해야 합니다.

코딩 작업이면, Python으로 Python 3.8 버전을 사용할 예정입니다.

이 섹션에서, 필요하고, 사용할 예정입니다:

- **Python 3 이해**.  Python 3의 프로그래밍 언어 이해. 이 강의에서는 입력, 반복, 파일 입력, 배열을 사용합니다.
- **Visual Studio Code + 확장**. Visual Studio Code와 Python 확장을 사용할 예정입니다. 선택에 따라 Python IDE를 사용할 수 있습니다.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob)은 간단한 Python 텍스트 처리 라이브러리입니다. TextBlob 사이트 설명을 따라서 시스템에 설치합니다 (보이는 것처럼, corpora도 설치합니다):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 팁: VS Code 환경에서 Python을 바로 실행할 수 있습니다. [docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott)으로 정보를 더 확인합니다.

## 기계와 대화하기

컴퓨터가 인간 언어를 이해하려 시도한 역사는 수십 년전으로, natural language processing을 고려한 초창기 사이언티스트 중 한 사람이 *Alan Turing*입니다.

### 'Turing test'

Turing이 1950년에 *artificial intelligence*를 연구하고 있을 때, 만약 대화하고 있는 사람이 다른 사람이나 컴퓨터와 대화하고 있는지 확신할 수 없다면, 사람과 컴퓨터의 대화를 (타이핑된 통신으로) 테스트할 수 있는지 고려했습니다.

만약, 일정 대화 이후에, 사람이 컴퓨터에서 나온 대답인지 결정할 수 없다면, 컴퓨터가 *thinking*하고 있다고 말할 수 있나요?

### 영감 - 'the imitation game'

*The Imitation Game*으로 불리는 파티 게임에서 유래된 아이디어로 질문자가 방에 혼자있고 (다른 방의) 두 사람 중 남성과 여성을 결정할 일을 맡게 됩니다. 질문하는 사람은 노트를 보낼 수 있으며, 성별을 알 수 없는 사람이 작성해서 보낼 답변을 생각하고 질문해야 합니다. 당연하게, 다른 방에 있는 사람도 잘 못 이끌거나 혼동하는 방식으로 답변하며, 정직하게 대답해주는 모습을 보여 질문하는 사람을 속이려고 합니다.

### Eliza 개발

1960년에 *Joseph Weizenbaum*으로 불린 MIT 사이언티스트는, 사람의 질문을 답변하고 답변을 이해하는 모습을 주는 컴퓨터 'therapist' [*Eliza*](https://wikipedia.org/wiki/ELIZA)를 개발했습니다. 하지만, Eliza는 문장을 파싱하고 특정 문법 구조와 키워드를 식별하여 이유있는 답변을 준다고 할 수 있지만, 문장을 *understand*한다고 말할 수 없습니다. 만약 Eliza가 "**I am** <u>sad</u>" 포맷과 유사한 문장을 제시받으면 문장에서 단어를 재배열하고 대치해서 "How long have **you been** <u>sad</u>" 형태로 응답할 수 있습니다.

Eliza가 문장을 이해하고 다음 질문을 대답하는 것처럼 인상을 줬지만, 실제로는, 시제를 바꾸고 일부 단어를 추가했을 뿐입니다. 만약 Eliza가 응답할 키워드를 식별하지 못하는 경우, 여러 다른 문장에 적용할 수 있는 랜덤 응답으로 대신합니다. 만약 사용자가 "**You are** a <u>bicycle</u>"라고 작성하면 더 이유있는 응답 대신에, "How long have **I been** a <u>bicycle</u>?"처럼 답변하므로, Eliza는 쉽게 속을 수 있습니다.

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 original ELIZA program에 대한 영상보려면 이미지 클릭

> 노트: ACM 계정을 가지고 있다면 출판된 [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) 원본 설명을 읽을 수 있습니다. 대신, [wikipedia](https://wikipedia.org/wiki/ELIZA)에서 Eliza에 대한 내용을 읽을 수도 있습니다.

## 연습 - 기초 대화 봇 코드 작성하기

Eliza와 같은, 대화 봇은, 사용자 입력을 유도해서 지능적으로 이해하고 답변하는 프로그램입니다. Eliza와 다르게, 봇은 지능적 대화 형태를 띄는 여러 룰이 없습니다. 대신, 봇은 대부분 사소한 대화에서 작동할 랜덤으로 응답해서 대화를 지속하는, 하나의 능력만 가지고 있을 것입니다.

### 계획

대화 봇을 만들 몇 단계가 있습니다:

1. 봇과 상호작용하는 방식을 사용자에 알려줄 명령 출력
2. 반복 시작
   1. 사용자 입력 승인
   2. 만약 사용자 종료 요청하면, 종료
   3. 사용자 입력 처리 및 응답 결정 (이 케이스는, 응답할 수 있는 일반적인 대답 목록에서 랜덤 선택)
   4. 응답 출력
3. 2 단계로 돌아가서 반복

### 봇 만들기

다음으로 봇을 만듭니다. 약간의 구문을 정의해서 시작해볼 예정입니다.

1. 해당 랜덤 응답으로 Python에서 봇을 만듭니다:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    가이드하는 약간의 샘플 출력이 있습니다 (사용자 입력은 라인의 시작점에 `>` 있습니다):

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

    작업에 맞는 하나의 솔루션은 [here](../solution/bot.py) 입니다

    ✅ 잠시 멈추고 생각합니다

    1. 랜덤 응답이 실제로 누군가를 이해했다고 생각하게 'trick'을 쓴다고 생각하나요?
    2. 봇이 더 효과있으려면 어떤 기능을 해야 될까요?
    3. 만약 봇이 문장의 의미를 정말 'understand' 했다면, 대화에서 이전 문장의 의미도 'remember'할 필요가 있을까요?

---

## 🚀 도전

"잠시 멈추고 생각합니다" 항목 중 하나를 골라서 코드를 구현하거나 의사 코드로 종이에 솔루션을 작성합니다.

다음 강의에서, natural language와 머신러닝을 분석하는 여러 다른 접근 방식에 대해 배울 예정입니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## 검토 & 자기주도 학습

더 읽을 수 있는 틈에 아래 레퍼런스를 찾아봅니다.

### 레퍼런스

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 과제 

[Search for a bot](../assignment.md)
