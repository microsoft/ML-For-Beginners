<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6534e145d52a3890590d27be75386e5d",
  "translation_date": "2025-09-04T00:34:52+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "ko"
}
-->
# 자연어 처리의 일반적인 작업과 기술

대부분의 *자연어 처리* 작업에서는 처리할 텍스트를 분해하고, 분석하며, 결과를 저장하거나 규칙 및 데이터 세트와 교차 참조해야 합니다. 이러한 작업을 통해 프로그래머는 텍스트에서 _의미_, _의도_, 또는 단순히 _단어 빈도_를 도출할 수 있습니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/33/)

텍스트를 처리하는 데 사용되는 일반적인 기술을 알아봅시다. 이러한 기술은 머신 러닝과 결합하여 대량의 텍스트를 효율적으로 분석할 수 있도록 도와줍니다. 그러나 ML을 이러한 작업에 적용하기 전에 NLP 전문가가 직면하는 문제를 이해해 봅시다.

## NLP의 일반적인 작업

텍스트를 분석하는 방법에는 여러 가지가 있습니다. 수행할 수 있는 작업이 있으며, 이를 통해 텍스트를 이해하고 결론을 도출할 수 있습니다. 이러한 작업은 일반적으로 순차적으로 수행됩니다.

### 토큰화

대부분의 NLP 알고리즘이 가장 먼저 해야 할 일은 텍스트를 토큰, 즉 단어로 분리하는 것입니다. 간단해 보이지만, 문장 부호와 다양한 언어의 단어 및 문장 구분자를 고려해야 하므로 까다로울 수 있습니다. 구분을 결정하기 위해 다양한 방법을 사용해야 할 수도 있습니다.

![토큰화](../../../../translated_images/tokenization.1641a160c66cd2d93d4524e8114e93158a9ce0eba3ecf117bae318e8a6ad3487.ko.png)
> **Pride and Prejudice**의 문장을 토큰화하는 과정. 인포그래픽: [Jen Looper](https://twitter.com/jenlooper)

### 임베딩

[단어 임베딩](https://wikipedia.org/wiki/Word_embedding)은 텍스트 데이터를 숫자로 변환하는 방법입니다. 임베딩은 비슷한 의미를 가진 단어 또는 함께 사용되는 단어들이 서로 가까이 모이도록 수행됩니다.

![단어 임베딩](../../../../translated_images/embedding.2cf8953c4b3101d188c2f61a5de5b6f53caaa5ad4ed99236d42bc3b6bd6a1fe2.ko.png)
> "I have the highest respect for your nerves, they are my old friends." - **Pride and Prejudice**의 문장에 대한 단어 임베딩. 인포그래픽: [Jen Looper](https://twitter.com/jenlooper)

✅ [이 흥미로운 도구](https://projector.tensorflow.org/)를 사용하여 단어 임베딩을 실험해 보세요. 단어를 클릭하면 'toy'가 'disney', 'lego', 'playstation', 'console'과 같은 유사 단어 클러스터와 연결되는 것을 볼 수 있습니다.

### 구문 분석 및 품사 태깅

토큰화된 각 단어는 명사, 동사, 형용사와 같은 품사로 태깅될 수 있습니다. 예를 들어, `the quick red fox jumped over the lazy brown dog`라는 문장은 fox = 명사, jumped = 동사로 품사 태깅될 수 있습니다.

![구문 분석](../../../../translated_images/parse.d0c5bbe1106eae8fe7d60a183cd1736c8b6cec907f38000366535f84f3036101.ko.png)

> **Pride and Prejudice**의 문장을 구문 분석하는 과정. 인포그래픽: [Jen Looper](https://twitter.com/jenlooper)

구문 분석은 문장에서 어떤 단어들이 서로 관련되어 있는지 인식하는 과정입니다. 예를 들어, `the quick red fox jumped`는 형용사-명사-동사 시퀀스로 `lazy brown dog` 시퀀스와는 별개입니다.

### 단어 및 구문 빈도

대량의 텍스트를 분석할 때 유용한 절차는 관심 있는 모든 단어 또는 구문의 사전을 작성하고 그것이 얼마나 자주 나타나는지 기록하는 것입니다. 예를 들어, `the quick red fox jumped over the lazy brown dog`라는 문장에서 `the`의 단어 빈도는 2입니다.

다음은 단어 빈도를 계산하는 예제 텍스트입니다. 러디어드 키플링의 시 **The Winners**는 다음 구절을 포함합니다:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

구문 빈도는 필요에 따라 대소문자를 구분하거나 구분하지 않을 수 있습니다. 예를 들어, `a friend`는 빈도가 2이고, `the`는 빈도가 6이며, `travels`는 빈도가 2입니다.

### N-그램

텍스트는 단일 단어(유니그램), 두 단어(바이그램), 세 단어(트라이그램) 또는 임의의 단어 수(n-그램)로 구성된 시퀀스로 분할될 수 있습니다.

예를 들어, `the quick red fox jumped over the lazy brown dog`를 n-그램 점수 2로 나누면 다음과 같은 n-그램이 생성됩니다:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

이를 문장 위에 슬라이딩 박스로 시각화하면 더 쉽게 이해할 수 있습니다. 다음은 3단어 n-그램의 예입니다. 각 문장에서 n-그램은 굵게 표시됩니다:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-그램 슬라이딩 윈도우](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-그램 값 3: 인포그래픽: [Jen Looper](https://twitter.com/jenlooper)

### 명사구 추출

대부분의 문장에는 주어 또는 목적어 역할을 하는 명사가 있습니다. 영어에서는 종종 'a', 'an', 'the'가 앞에 오는 것으로 식별할 수 있습니다. 문장의 의미를 이해하려고 할 때 '명사구를 추출'하는 것은 NLP에서 일반적인 작업입니다.

✅ "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun."라는 문장에서 명사구를 식별할 수 있나요?

`the quick red fox jumped over the lazy brown dog`라는 문장에는 **quick red fox**와 **lazy brown dog**라는 두 개의 명사구가 있습니다.

### 감정 분석

문장이나 텍스트는 *긍정적*인지 *부정적*인지 감정을 분석할 수 있습니다. 감정은 *극성*과 *객관성/주관성*으로 측정됩니다. 극성은 -1.0에서 1.0(부정적에서 긍정적)까지, 객관성은 0.0에서 1.0(가장 객관적에서 가장 주관적)까지 측정됩니다.

✅ 나중에 머신 러닝을 사용하여 감정을 결정하는 다양한 방법을 배우겠지만, 한 가지 방법은 인간 전문가가 긍정적 또는 부정적으로 분류한 단어와 구문 목록을 사용하여 텍스트에 모델을 적용하고 극성 점수를 계산하는 것입니다. 이 방법이 어떤 상황에서는 잘 작동하고 다른 상황에서는 덜 작동하는 이유를 이해할 수 있나요?

### 굴절

굴절은 단어를 단수형 또는 복수형으로 변환할 수 있도록 합니다.

### 어간 추출

*어간*은 단어 집합의 기본 또는 중심 단어입니다. 예를 들어, *flew*, *flies*, *flying*의 어간은 동사 *fly*입니다.

NLP 연구자에게 유용한 데이터베이스도 있습니다. 특히:

### WordNet

[WordNet](https://wordnet.princeton.edu/)은 다양한 언어의 모든 단어에 대한 동의어, 반의어 및 기타 세부 정보를 포함한 데이터베이스입니다. 번역, 맞춤법 검사기 또는 언어 도구를 구축할 때 매우 유용합니다.

## NLP 라이브러리

다행히도 이러한 기술을 직접 구축할 필요는 없습니다. 자연어 처리나 머신 러닝에 전문적이지 않은 개발자도 접근할 수 있도록 훌륭한 Python 라이브러리가 제공됩니다. 다음 레슨에서는 이러한 라이브러리의 더 많은 예제를 다루겠지만, 여기서는 다음 작업에 도움이 되는 몇 가지 유용한 예제를 배웁니다.

### 연습 - `TextBlob` 라이브러리 사용

TextBlob이라는 라이브러리를 사용해 봅시다. 이 라이브러리는 이러한 유형의 작업을 처리하는 데 유용한 API를 포함하고 있습니다. TextBlob은 "[NLTK](https://nltk.org)와 [pattern](https://github.com/clips/pattern)의 거대한 어깨 위에 서 있으며, 두 라이브러리와 잘 호환됩니다." API에 상당한 양의 머신 러닝이 포함되어 있습니다.

> 참고: TextBlob에 대한 유용한 [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) 가이드가 제공되며, 경험 많은 Python 개발자에게 추천됩니다.

명사구를 식별하려고 할 때 TextBlob은 명사구를 찾기 위한 여러 추출 옵션을 제공합니다.

1. `ConllExtractor`를 살펴보세요.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > 여기서 무슨 일이 일어나고 있나요? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor)는 "ConLL-2000 학습 코퍼스를 사용하여 청크 구문 분석으로 학습된 명사구 추출기"입니다. ConLL-2000은 2000년 Computational Natural Language Learning 컨퍼런스를 의미합니다. 매년 컨퍼런스는 어려운 NLP 문제를 해결하기 위한 워크숍을 개최했으며, 2000년에는 명사 청킹이 주제였습니다. 모델은 Wall Street Journal의 "섹션 15-18을 학습 데이터(211727 토큰)로, 섹션 20을 테스트 데이터(47377 토큰)로 사용하여" 학습되었습니다. 사용된 절차는 [여기](https://www.clips.uantwerpen.be/conll2000/chunking/)에서, 결과는 [여기](https://ifarm.nl/erikt/research/np-chunking.html)에서 확인할 수 있습니다.

### 챌린지 - NLP로 봇 개선하기

이전 레슨에서 매우 간단한 Q&A 봇을 만들었습니다. 이제 입력을 감정 분석하여 감정에 맞는 응답을 출력함으로써 Marvin을 조금 더 공감할 수 있도록 만들어 봅시다. 또한 `noun_phrase`를 식별하고 이에 대해 질문해야 합니다.

더 나은 대화형 봇을 구축할 때의 단계:

1. 사용자에게 봇과 상호작용하는 방법에 대한 지침 출력
2. 루프 시작 
   1. 사용자 입력 수락
   2. 사용자가 종료를 요청하면 종료
   3. 사용자 입력을 처리하고 적절한 감정 응답 결정
   4. 감정에서 명사구가 감지되면 복수형으로 변환하고 해당 주제에 대해 추가 입력 요청
   5. 응답 출력
3. 2단계로 다시 루프

다음은 TextBlob을 사용하여 감정을 결정하는 코드 스니펫입니다. 감정 응답에는 네 가지 *그라디언트*만 있습니다(원한다면 더 추가할 수도 있습니다):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

다음은 샘플 출력입니다(사용자 입력은 >로 시작하는 줄에 표시됩니다):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

이 작업에 대한 가능한 솔루션은 [여기](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)에 있습니다.

✅ 지식 점검

1. 공감하는 응답이 봇이 실제로 사용자를 이해한다고 '속일' 수 있다고 생각하나요?
2. 명사구를 식별하는 것이 봇을 더 '믿을 수 있게' 만들까요?
3. 문장에서 '명사구'를 추출하는 것이 왜 유용할까요?

---

이전 지식 점검에서 봇을 구현하고 친구에게 테스트해 보세요. 친구를 속일 수 있나요? 봇을 더 '믿을 수 있게' 만들 수 있나요?

## 🚀챌린지

이전 지식 점검에서 작업을 선택하여 구현해 보세요. 봇을 친구에게 테스트해 보세요. 친구를 속일 수 있나요? 봇을 더 '믿을 수 있게' 만들 수 있나요?

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/34/)

## 복습 및 자기 학습

다음 몇 가지 레슨에서는 감정 분석에 대해 더 배울 것입니다. [KDNuggets](https://www.kdnuggets.com/tag/nlp)와 같은 기사에서 이 흥미로운 기술을 연구해 보세요.

## 과제 

[봇이 대답하도록 만들기](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  