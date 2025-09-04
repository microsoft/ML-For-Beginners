<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-09-04T00:50:55+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "ko"
}
-->
# 기계 학습을 활용한 번역 및 감정 분석

이전 강의에서는 `TextBlob` 라이브러리를 사용하여 기본적인 NLP 작업(예: 명사구 추출)을 수행하는 간단한 봇을 만드는 방법을 배웠습니다. 컴퓨터 언어학에서 또 다른 중요한 과제는 문장을 한 언어에서 다른 언어로 정확하게 _번역_하는 것입니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

번역은 수천 개의 언어가 존재하고 각 언어마다 문법 규칙이 매우 다를 수 있다는 점에서 매우 어려운 문제입니다. 한 가지 접근 방식은 영어와 같은 한 언어의 문법 규칙을 언어에 의존하지 않는 구조로 변환한 다음, 이를 다시 다른 언어로 변환하는 것입니다. 이 접근 방식은 다음 단계를 포함합니다:

1. **식별**. 입력 언어의 단어를 명사, 동사 등으로 태깅합니다.
2. **번역 생성**. 대상 언어 형식으로 각 단어를 직접 번역합니다.

### 예문: 영어에서 아일랜드어로

'영어'에서 문장 _I feel happy_는 다음과 같은 순서로 세 단어로 구성됩니다:

- **주어** (I)
- **동사** (feel)
- **형용사** (happy)

하지만 '아일랜드어'에서는 같은 문장이 매우 다른 문법 구조를 가집니다. "*happy*"나 "*sad*"와 같은 감정은 *나에게 있다*는 방식으로 표현됩니다.

영어 문장 `I feel happy`는 아일랜드어로 `Tá athas orm`입니다. 이를 *직역*하면 `Happy is upon me`가 됩니다.

아일랜드어 화자가 영어로 번역할 때는 `Happy is upon me`가 아니라 `I feel happy`라고 말합니다. 이는 단어와 문장 구조가 다르더라도 문장의 의미를 이해하기 때문입니다.

아일랜드어 문장의 공식적인 순서는 다음과 같습니다:

- **동사** (Tá 또는 is)
- **형용사** (athas, 또는 happy)
- **주어** (orm, 또는 upon me)

## 번역

단순한 번역 프로그램은 문장 구조를 무시하고 단어만 번역할 수 있습니다.

✅ 성인이 되어 두 번째(또는 세 번째 이상의) 언어를 배운 적이 있다면, 처음에는 모국어로 생각한 후 개념을 단어 단위로 머릿속에서 번역하고, 그 번역을 말로 표현했을 가능성이 높습니다. 이는 단순한 번역 컴퓨터 프로그램이 하는 것과 비슷합니다. 유창함을 얻으려면 이 단계를 넘어서는 것이 중요합니다!

단순한 번역은 종종 잘못된(때로는 웃긴) 번역을 초래합니다. 예를 들어, `I feel happy`를 아일랜드어로 직역하면 `Mise bhraitheann athas`가 됩니다. 이는 문자 그대로 `me feel happy`를 의미하며, 올바른 아일랜드어 문장이 아닙니다. 영어와 아일랜드어는 서로 가까운 섬에서 사용되는 언어임에도 불구하고, 문법 구조가 매우 다릅니다.

> 아일랜드어 언어 전통에 대한 비디오를 [여기](https://www.youtube.com/watch?v=mRIaLSdRMMs)에서 시청할 수 있습니다.

### 기계 학습 접근법

지금까지 자연어 처리를 위한 공식 규칙 접근법에 대해 배웠습니다. 또 다른 접근법은 단어의 의미를 무시하고 _대신 기계 학습을 사용하여 패턴을 감지_하는 것입니다. 원본 언어와 대상 언어로 된 많은 텍스트(코퍼스)가 있다면, 이 방법은 번역에 효과적일 수 있습니다.

예를 들어, 1813년에 제인 오스틴이 쓴 유명한 영어 소설 *오만과 편견*을 생각해 봅시다. 영어 원본과 *프랑스어*로 된 인간 번역본을 참고하면, 한 언어에서 다른 언어로 _관용적으로_ 번역된 구문을 감지할 수 있습니다. 곧 이를 실습해 볼 것입니다.

예를 들어, 영어 문장 `I have no money`를 프랑스어로 직역하면 `Je n'ai pas de monnaie`가 될 수 있습니다. "Monnaie"는 프랑스어에서 '거스름돈'을 의미하는 'false cognate'로, 'money'와 동의어가 아닙니다. 인간 번역자가 더 나은 번역을 한다면 `Je n'ai pas d'argent`가 될 것입니다. 이는 '거스름돈'이 아니라 '돈이 없다'는 의미를 더 잘 전달합니다.

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.ko.png)

> 이미지 제공: [Jen Looper](https://twitter.com/jenlooper)

충분한 인간 번역본을 기반으로 모델을 구축하면, ML 모델은 이전에 전문가가 번역한 텍스트에서 공통 패턴을 식별하여 번역 정확도를 향상시킬 수 있습니다.

### 실습 - 번역

`TextBlob`을 사용하여 문장을 번역할 수 있습니다. **오만과 편견**의 유명한 첫 문장을 시도해 보세요:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob`은 번역을 꽤 잘 수행합니다: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!".

1932년 V. Leconte와 Ch. Pressoir가 번역한 프랑스어 번역본보다 TextBlob의 번역이 훨씬 정확하다고 주장할 수 있습니다:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

이 경우, ML에 기반한 번역이 원작자의 의도를 명확히 하기 위해 불필요하게 단어를 추가한 인간 번역자보다 더 나은 결과를 제공합니다.

> TextBlob이 번역을 잘하는 이유는 무엇일까요? TextBlob은 Google 번역을 사용하며, 이는 수백만 개의 구문을 분석하여 작업에 가장 적합한 문자열을 예측할 수 있는 정교한 AI입니다. 여기에는 수작업이 전혀 포함되지 않으며, `blob.translate`를 사용하려면 인터넷 연결이 필요합니다.

✅ 몇 가지 문장을 더 시도해 보세요. ML 번역과 인간 번역 중 어느 것이 더 나은가요? 어떤 경우에 더 나은가요?

## 감정 분석

기계 학습이 매우 효과적으로 작동할 수 있는 또 다른 영역은 감정 분석입니다. 비-ML 접근법으로 감정을 분석하려면 '긍정적' 또는 '부정적'인 단어와 구문을 식별합니다. 그런 다음, 새로운 텍스트를 주어진 경우, 긍정적, 부정적, 중립적 단어의 총 값을 계산하여 전체 감정을 식별합니다.

이 접근법은 쉽게 속을 수 있습니다. 예를 들어, Marvin 과제에서 본 것처럼 `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road`라는 문장은 비꼬는 부정적 감정의 문장이지만, 간단한 알고리즘은 'great', 'wonderful', 'glad'를 긍정적으로, 'waste', 'lost', 'dark'를 부정적으로 감지합니다. 상반된 단어들로 인해 전체 감정이 왜곡됩니다.

✅ 인간 화자로서 우리가 비꼬는 표현을 어떻게 전달하는지 잠시 생각해 보세요. 억양이 큰 역할을 합니다. "Well, that film was awesome"이라는 문장을 다양한 방식으로 말해 보며, 목소리가 의미를 어떻게 전달하는지 알아보세요.

### ML 접근법

ML 접근법은 부정적, 긍정적 텍스트(예: 트윗, 영화 리뷰 등)를 수동으로 수집하는 것입니다. 인간이 점수와 의견을 제공한 데이터를 기반으로 NLP 기술을 적용하여 패턴을 도출합니다(예: 긍정적 영화 리뷰에는 'Oscar worthy'라는 표현이 부정적 리뷰보다 더 자주 등장).

> ⚖️ **예시**: 정치인의 사무실에서 새로운 법안에 대한 찬반 의견을 담은 이메일을 분류하는 작업을 맡았다고 가정해 봅시다. 이메일이 많다면, 모든 이메일을 읽는 것이 부담스러울 수 있습니다. 봇이 이메일을 읽고 찬성 또는 반대 쪽으로 분류해 준다면 얼마나 좋을까요? 
> 
> 이를 달성하는 한 가지 방법은 기계 학습을 사용하는 것입니다. 모델을 일부 찬성 이메일과 일부 반대 이메일로 훈련시킵니다. 모델은 특정 단어와 패턴이 찬성 또는 반대 이메일에 더 자주 나타나는 경향이 있음을 학습합니다. 그런 다음, 훈련에 사용되지 않은 이메일로 테스트하여 모델이 동일한 결론에 도달하는지 확인합니다. 모델의 정확성에 만족하면, 이후 이메일을 읽지 않고도 처리할 수 있습니다.

✅ 이전 강의에서 사용한 프로세스와 유사한 점이 있나요?

## 실습 - 감정 문장

감정은 -1에서 1까지의 *극성*으로 측정됩니다. -1은 가장 부정적인 감정, 1은 가장 긍정적인 감정을 나타냅니다. 감정은 또한 0(객관적)에서 1(주관적)까지의 점수로 측정됩니다.

제인 오스틴의 *오만과 편견*을 다시 살펴보세요. 텍스트는 [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm)에서 확인할 수 있습니다. 아래 샘플은 책의 첫 문장과 마지막 문장의 감정을 분석하고, 감정 극성과 주관성/객관성 점수를 표시하는 짧은 프로그램을 보여줍니다.

다음 작업에서는 `TextBlob` 라이브러리를 사용하여 `sentiment`를 결정하세요(직접 감정 계산기를 작성할 필요는 없습니다).

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

다음과 같은 출력이 표시됩니다:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## 도전 과제 - 감정 극성 확인

*오만과 편견*에서 절대적으로 긍정적인 문장이 절대적으로 부정적인 문장보다 더 많은지 감정 극성을 사용하여 확인하세요. 이 작업에서는 극성 점수가 1 또는 -1인 경우를 절대적으로 긍정적이거나 부정적이라고 가정합니다.

**단계:**

1. [오만과 편견](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm)의 사본을 .txt 파일로 다운로드합니다. 파일 시작과 끝의 메타데이터를 제거하고 원본 텍스트만 남깁니다.
2. Python에서 파일을 열고 내용을 문자열로 추출합니다.
3. 책 문자열을 사용하여 TextBlob을 생성합니다.
4. 책의 각 문장을 루프에서 분석합니다.
   1. 극성이 1 또는 -1인 경우, 문장을 긍정적 또는 부정적 메시지의 배열 또는 목록에 저장합니다.
5. 마지막으로, 긍정적 문장과 부정적 문장을 각각 출력하고 개수를 표시합니다.

다음은 샘플 [솔루션](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)입니다.

✅ 지식 점검

1. 감정은 문장에서 사용된 단어를 기반으로 하지만, 코드가 단어를 *이해*하나요?
2. 감정 극성이 정확하다고 생각하나요? 즉, 점수에 *동의*하나요?
   1. 특히, 다음 문장의 절대 **긍정적** 극성에 동의하나요?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. 다음 세 문장은 절대적으로 긍정적인 감정으로 점수가 매겨졌지만, 자세히 읽어보면 긍정적인 문장이 아닙니다. 감정 분석이 왜 긍정적이라고 판단했을까요?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. 다음 문장의 절대 **부정적** 극성에 동의하나요?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ 제인 오스틴의 팬이라면 그녀가 종종 책을 통해 영국 섭정 시대의 더 우스꽝스러운 측면을 비판한다는 것을 이해할 것입니다. *오만과 편견*의 주인공 엘리자베스 베넷은 예리한 사회 관찰자(작가와 마찬가지)이며, 그녀의 언어는 종종 매우 미묘합니다. 심지어 이야기의 사랑스러운 상대인 Mr. Darcy조차도 엘리자베스의 장난스럽고 놀리는 언어 사용을 언급합니다: "나는 당신이 가끔 자신의 의견이 아닌 의견을 즐겁게 표현하는 것을 좋아한다는 것을 알 만큼 충분히 오래 당신을 알고 있습니다."

---

## 🚀도전 과제

사용자 입력에서 다른 특징을 추출하여 Marvin을 더 개선할 수 있나요?

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## 복습 및 자기 학습
텍스트에서 감정을 추출하는 방법은 여러 가지가 있습니다. 이 기술을 활용할 수 있는 비즈니스 응용 프로그램을 생각해 보세요. 또한, 이 기술이 어떻게 잘못될 수 있는지에 대해서도 고민해 보세요. 감정을 분석하는 정교한 기업용 시스템에 대해 더 알아보세요. 예를 들어 [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)와 같은 시스템이 있습니다. 위의 '오만과 편견' 문장 중 일부를 테스트해 보고, 미묘한 뉘앙스를 감지할 수 있는지 확인해 보세요.

## 과제

[Poetic license](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있지만, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전을 권위 있는 출처로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 책임을 지지 않습니다.