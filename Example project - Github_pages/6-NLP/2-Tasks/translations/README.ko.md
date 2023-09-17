# 일반적인 natural language processing 작업과 기술

대부분 *natural language processing* 작업으로, 처리한 텍스트를 분해하고, 검사하고, 그리고 결과를 저장하거나 룰과 데이터셋을 서로 참조했습니다. 이 작업들로, 프로그래머가 _meaning_ 또는 _intent_ 또는 오직 텍스트에 있는 용어와 단어의 _frequency_ 만 끌어낼 수 있게 합니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/33/)

텍스트를 처리하며 사용했던 일반적인 기술을 찾아봅니다. 머신러닝에 결합된, 이 기술은 효율적으로 많은 텍스트를 분석하는데 도와줍니다. 그러나, 이 작업에 ML을 적용하기 전에, NLP 스페셜리스트가 일으킨 문제를 이해합니다.

## NLP의 공통 작업

작업하고 있는 텍스트를 분석하는 다양한 방식이 있습니다. 진행할 작업과 이 작업으로 텍스트 이해도로 측정하고 결론을 지을 수 있습니다. 대부분 순서대로 작업합니다.

### Tokenization

아마 많은 NLP 알고리즘으로 처음 할 일은 토큰이나, 단어로 텍스트를 나누는 것입니다. 간단하게 들리지만, 문장 부호와 다른 언어의 단어와 문장 구분 기호를 고려하는 건 까다로울 수 있습니다.

![tokenization](../images/tokenization.png)
> Tokenizing a sentence from **Pride and Prejudice**. Infographic by [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding)은 텍스트 데이터를 숫자처럼 변환하는 방식입니다. Embeddings은 의미가 비슷한 단어이거나 cluster와 단어를 함께 쓰는 방식으로 이루어집니다.

![word embeddings](../images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings for a sentence in **Pride and Prejudice**. Infographic by [Jen Looper](https://twitter.com/jenlooper)

✅ [this interesting tool](https://projector.tensorflow.org/)로 단어 embeddings를 실험해봅니다. 하나의 단어를 클릭하면 비슷한 단어의 클러스터를 보여줍니다: 'disney', 'lego', 'playstation', 그리고 'console'이 'toy' 클러스터에 있있습니다.

### 파싱 & Part-of-speech Tagging

토큰화된 모든 단어는 품사를 명사, 동사, 형용사로 테그할 수 있습니다. `the quick red fox jumped over the lazy brown dog` 문장은 fox = noun, jumped = verb로 POS 태그될 수 있습니다.

![parsing](../images/parse.png)

> Parsing a sentence from **Pride and Prejudice**. Infographic by [Jen Looper](https://twitter.com/jenlooper)

파싱은 문장에서 각자 단어들이 관련있는지 인식하는 것입니다. 예시로 `the quick red fox jumped`는 `lazy brown dog` 시퀀스와 나눠진 형용사-명사-동사 시퀀스 입니다.

### 단어와 구문 빈도

텍스트의 많은 분량을 분석할 때 유용한 순서는 흥미있는 모든 단어 또는 자주 나오는 사전을 만드는 것입니다. `the quick red fox jumped over the lazy brown dog` 문구는 단어 빈도가 2 입니다.

단어 빈도를 세는 예시를 찾아봅니다. Rudyard Kipling의 시인 The Winners는 다음을 담고 있습니다:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

구문 빈도는 필요에 의해서 대소문자를 구분하지 않거나 구분하므로, `a friend`는 빈도 2이고 `the`는 빈도 6, 그리고 `travels`는 2입니다.

### N-grams

텍스트는 지정한 길이의 단어 시퀀스, 한 단어(unigram), 두 단어(bigrams), 세 단어(trigrams) 또는 모든 수의 단어(n-grams)로 나눌 수 있습니다.

예시로 n-gram 2점인 `the quick red fox jumped over the lazy brown dog`는 다음 n-grams을 만듭니다:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

문장 위 슬라이드 박스로 시각화하는 게 쉬울 수 있습니다. 여기는 3 단어로 이루어진 n-grams이며, n-gram은 각 문장에서 볼드체로 있습니다:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams sliding window](../images/n-grams.gif)

> N-gram value of 3: Infographic by [Jen Looper](https://twitter.com/jenlooper)

### Noun phrase 추출 

대부분 문장에서, 문장의 주어나, 목적어인 명사가 있습니다. 영어에서, 자주  'a' 또는 'an' 또는 'the'가 앞에 오게 가릴 수 있습니다. "noun phrase 추출'로 문장의 주어 또는 목적어를 가려내려 하는 것은 NLP에서 문장의 의미를 이해할 때 일반적인 작업입니다.

✅ "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun." 문장에서, noun phrases를 가려낼 수 있나요?

`the quick red fox jumped over the lazy brown dog` 문장에서 noun phrases 2개가 있습니다: **quick red fox** 와 **lazy brown dog**.

### 감정 분석

문장이나 텍스트는 감정이나, *positive* 또는 *negative*인지 분석할 수 있습니다. 감정은 *polarity* 와 *objectivity/subjectivity*로 측정됩니다. Polarity는 -1.0 에서 1.0 (negative 에서 positive) 이며 0.0 에서 1.0 (가장 객관적에서 가장 주관적으로)으로 측정됩니다.

✅ 나중에 머신러닝으로 감정을 판단하는 다른 방식을 배울 수 있지만, 하나의 방식은 전문가가 positive 또는 negative로 분류된 단어와 구분의 리스트를 가지고 polarity 점수를 계산한 텍스트로 모델을 적용하는 것입니다. 일부 상황에서 어떻게 작동하고 다른 상황에서도 잘 동작하는지 볼 수 있나요?

### Inflection

Inflection은 단어를 가져와서 단수나 복수 단어를 얻게 됩니다.

### Lemmatization

*lemma*는 단어 세트에서 어원이나 표제어고, 예시로 *flew*, *flies*, *flying*은 *fly* 동사의 lemma를 가지고 있습니다.

특히, NLP 연구원이 사용할 수 있는 유용한 데이터베이스도 있습니다:

### WordNet

[WordNet](https://wordnet.princeton.edu/)은 다양한 언어로 모든 단어를 단어, 동의어, 반의어 그리고 다양한 기타 내용으로 이룬 데이터베이스입니다. 번역, 맞춤법 검사, 또는 모든 타입의 언어 도구를 만드려고 시도할 때 매우 유용합니다.

## NLP 라이브러리

운 좋게, 훌륭한 Python 라이브러리로 natural language processing이나 머신러닝에 전문적이지 않은 개발자도 쉽게 접근할 수 있으므로, 이 기술을 스스로 다 만들지 않아도 됩니다. 다음 강의에서 더 많은 예시를 포함하지만, 여기에서 다음 작업에 도움이 될 몇 유용한 예시를 배울 예정입니다.

### 연습 - `TextBlob` 라이브러리 사용

이 타입의 작업을 처리하는 유용한 API를 포함한 TextBlob이라고 불리는 라이브리를 사용합니다. TextBlob은 "stands on the giant shoulders of [NLTK](https://nltk.org) and [pattern](https://github.com/clips/pattern), and plays nicely with both."이며  API에서 상당히 많이 ML이 녹아들어졌습니다.

> 노트: 잘하는 Python 개발자를 위해서 추천하는 TextBlob의 유용한 [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) 가이드가 존재합니다

*noun phrases* 식별하려고 시도하는 순간, TextBlob은 noun phrases를 찾고자 몇 추출 옵션을 제공합니다.

1. `ConllExtractor` 봅니다.

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

    > 어떤 일이 생기나요? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor)는 "A noun phrase extractor that uses chunk parsing trained with the ConLL-2000 training corpus."입니다. ConLL-2000은 2000 Conference on Computational Natural Language Learning을 의미합니다. 매년 까다로운 NLP 문제를 해결하기 위한 워크숍을 호스트하는 컨퍼런스이며, 2000년에는 noun chunking이었습니다. 모델은 "sections 15-18 as training data (211727 tokens) and section 20 as test data (47377 tokens)"로 Wall Street Journal에서 훈련되었습니다. [here](https://www.clips.uantwerpen.be/conll2000/chunking/)에서 사용한 순서와 [results](https://ifarm.nl/erikt/research/np-chunking.html)를 볼 수 있습니다.

### 도전 - NLP로 봇 개선하기

이전 강의에서 매우 간단한 Q&A 봇을 만들었습니다. 이제, 감정을 넣어서 분석하고 감정과 맞는 응답을 출력하여 Marvin을 좀 더 감성적으로 만듭니다. 또 `noun_phrase`를 식별하고 물어볼 필요가 있습니다.

더 좋은 대화 봇을 만들 때 단계가 있습니다:

1. 사용자에게 봇과 상호작용하는 방식 출력
2. 반복 시작
   1. 사용자 입력 승인
   2. 만약 사용자가 종료 요청하면, 종료
   3. 사용자 입력 처리하고 적절한 감정 응답 결정
   4. 만약 감정에서 noun phrase 탐지되면, 복수형 변경하고 이 토픽에서 입력 추가 요청
   5. 응답 출력
3. 2 단계로 돌아가서 반복

여기 TextBlob으로 감정을 탐지하는 코드 스니펫이 있습니다. 감정 응답에 4개 *gradients*만 있다는 점을 참고합니다 (좋아하는 경우 더 가질 수 있음):

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

여기는 가이드할 약간의 샘플 출력이 있습니다 (사용자 입력은 라인 시작에 > 있습니다):

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

작업에 대한 하나의 가능한 솔루션은 [here](../solution/bot.py) 있습니다.

✅ 지식 점검

1. 봇이 그 사람들을 실제로 이해했다고 생각할 수 있게 감성적인 반응으로 'trick'할 수 있다고 생각하나요?
2. noun phrase를 식별하면 봇을 더 '믿을' 수 있나요?
3. 문장에서 'noun phrase'를 추출하는 이유는 무엇인가요?

---

이전의 지식 점검에서 봇을 구현하고 친구에게 테스트해봅니다. 그들을 속일 수 있나요? 좀 더 '믿을 수'있게 봇을 만들 수 있나요?

## 🚀 도전

이전의 지식 점검에서 작업하고 구현합니다. 친구에게 봇을 테스트합니다. 그들을 속일 수 있나요? 좀 더 '믿을 수'있게 봇을 만들 수 있나요?

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/34/)

## 검토 & 자기주도 학습

다음 몇 강의에서 감정 분석에 대하여 더 배울 예정입니다. [KDNuggets](https://www.kdnuggets.com/tag/nlp) 같은 아티클에서 흥미로운 기술을 연구합니다.

## 과제 

[Make a bot talk back](../assignment.md)
