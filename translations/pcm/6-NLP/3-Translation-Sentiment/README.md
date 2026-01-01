<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-11-18T18:36:10+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "pcm"
}
-->
# Translation and sentiment analysis with ML

For di previous lessons, you don learn how to build one basic bot wey dey use `TextBlob`, one library wey get ML for di background to do basic NLP work like noun phrase extraction. Another big wahala for computational linguistics na how to translate sentence well from one language to another.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Translation na one hard problem because plenty languages dey and di grammar rules for each language fit dey very different. One way wey dem dey try solve am na to change di grammar rules for one language, like English, into one structure wey no depend on language, then translate am back to another language. Dis method mean say you go follow dis steps:

1. **Identification**. Mark di words for di input language as noun, verb, etc.
2. **Create translation**. Translate each word directly into di target language format.

### Example sentence, English to Irish

For 'English', di sentence _I feel happy_ na three words wey dey follow dis order:

- **subject** (I)
- **verb** (feel)
- **adjective** (happy)

But for 'Irish' language, di same sentence get different grammar structure - emotions like "*happy*" or "*sad*" dey expressed as something wey dey *upon* you.

Di English sentence `I feel happy` for Irish go be `Tá athas orm`. If you wan translate am word for word, e go mean `Happy is upon me`.

Person wey sabi Irish wey dey translate to English go talk `I feel happy`, no be `Happy is upon me`, because dem understand di meaning of di sentence, even if di words and sentence structure dey different.

Di formal order for di sentence for Irish na:

- **verb** (Tá or is)
- **adjective** (athas, or happy)
- **subject** (orm, or upon me)

## Translation

One naive translation program fit just dey translate words only, e no go look di sentence structure.

✅ If you don learn another language as adult, you fit start by dey think for your native language, dey translate di idea word by word for your head to di second language, then talk di translation. Na dis kind thing naive translation computer programs dey do. To sabi di language well, you go need pass dis stage!

Naive translation dey lead to bad (and sometimes funny) translations: `I feel happy` go translate word for word to `Mise bhraitheann athas` for Irish. Dis one mean (word for word) `me feel happy` and e no be correct Irish sentence. Even though English and Irish na languages wey dem dey speak for two islands wey dey near each other, dem be very different languages wey get different grammar structures.

> You fit watch some videos about Irish language tradition like [dis one](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Machine learning approaches

So far, you don learn about di formal rules method for natural language processing. Another method na to no look di meaning of di words, but use machine learning to find patterns. Dis one fit work for translation if you get plenty text (a *corpus*) or texts (*corpora*) for both di original and target languages.

For example, look di case of *Pride and Prejudice*, one popular English novel wey Jane Austen write for 1813. If you check di book for English and one human translation of di book for *French*, you fit see phrases for one wey dem translate *idiomatically* into di other. You go try am soon.

For example, if English phrase like `I have no money` dey translate word for word to French, e fit turn `Je n'ai pas de monnaie`. "Monnaie" na tricky French 'false cognate', because 'money' and 'monnaie' no mean di same thing. Better translation wey human fit do na `Je n'ai pas d'argent`, because e go explain di meaning well say you no get money (no be 'loose change' wey be di meaning of 'monnaie').

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b.pcm.png)

> Image by [Jen Looper](https://twitter.com/jenlooper)

If ML model get enough human translations to build model on, e fit make di translations better by finding common patterns for texts wey expert human speakers of di two languages don translate before.

### Exercise - translation

You fit use `TextBlob` to translate sentences. Try di popular first line of **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` dey do di translation well: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

You fit argue say TextBlob translation dey more correct, sef, than di 1932 French translation of di book by V. Leconte and Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

For dis case, di translation wey ML dey do dey better pass di human translator wey dey add extra words for di original author mouth for 'clarity'.

> Wetin dey happen here? And why TextBlob dey translate well? Well, for di background, e dey use Google translate, one advanced AI wey fit check millions of phrases to predict di best strings for di work wey e dey do. Nothing manual dey happen here and you need internet connection to use `blob.translate`.

✅ Try more sentences. Which one better, ML or human translation? For which cases?

## Sentiment analysis

Another area wey machine learning dey work well na sentiment analysis. One method wey no use ML to find sentiment na to mark words and phrases wey be 'positive' and 'negative'. Then, if you get new text, calculate di total value of di positive, negative and neutral words to find di overall sentiment. 

Dis method fit dey tricked as you don see for di Marvin task - di sentence `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` na sarcastic, negative sentiment sentence, but di simple algorithm go see 'great', 'wonderful', 'glad' as positive and 'waste', 'lost' and 'dark' as negative. Di overall sentiment go dey confused by dis conflicting words.

✅ Stop small and think how we dey use sarcasm as human speakers. Di way we dey talk (tone) dey play big role. Try talk di sentence "Well, that film was awesome" for different ways to see how your voice dey show di meaning.

### ML approaches

Di ML method go be to gather negative and positive texts manually - tweets, or movie reviews, or anything wey di human don give score *and* write opinion. Then NLP techniques go dey applied to di opinions and scores, so dat patterns go show (e.g., positive movie reviews dey use di phrase 'Oscar worthy' pass negative movie reviews, or positive restaurant reviews dey use 'gourmet' pass 'disgusting').

> ⚖️ **Example**: If you dey work for politician office and dem dey talk about new law, di people fit write emails wey dey support or dey against di law. Imagine say dem tell you to read di emails and sort dem into 2 groups, *for* and *against*. If di emails plenty, e fit tire you to read all of dem. E go sweet if bot fit read all di emails for you, understand dem and tell you which group each email belong. 
> 
> One way to do am na to use Machine Learning. You go train di model with some *against* emails and some *for* emails. Di model go dey associate phrases and words with di against side and di for side, *but e no go understand di content*, e go just know say some words and patterns dey appear more for *against* or *for* emails. You fit test am with some emails wey you no use train di model, and see if e go get di same result as you. Then, if you dey happy with di accuracy of di model, you fit process future emails without reading each one.

✅ Dis process dey similar to wetin you don use for previous lessons?

## Exercise - sentimental sentences

Sentiment dey measured with *polarity* of -1 to 1, meaning -1 na di most negative sentiment, and 1 na di most positive. Sentiment dey also measured with 0 - 1 score for objectivity (0) and subjectivity (1).

Look Jane Austen *Pride and Prejudice* again. Di text dey available here for [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Di sample below dey show one short program wey dey analyse di sentiment of di first and last sentences from di book and show di sentiment polarity and subjectivity/objectivity score.

You go use di `TextBlob` library (we don talk about am before) to find `sentiment` (you no need write your own sentiment calculator) for di task wey dey below.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

You go see dis output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Challenge - check sentiment polarity

Your task na to find, using sentiment polarity, if *Pride and Prejudice* get more absolutely positive sentences pass absolutely negative ones. For dis task, you fit assume say polarity score of 1 or -1 na absolutely positive or negative.

**Steps:**

1. Download one [copy of Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) from Project Gutenberg as .txt file. Remove di metadata for di start and end of di file, leave only di original text.
2. Open di file for Python and extract di contents as string.
3. Create one TextBlob using di book string.
4. Analyse each sentence for di book for loop.
   1. If di polarity na 1 or -1, keep di sentence for one array or list of positive or negative messages.
5. For di end, print all di positive sentences and negative sentences (separately) and di number of each.

Here na one sample [solution](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Knowledge Check

1. Di sentiment dey based on di words wey dey di sentence, but di code *understand* di words?
2. You think say di sentiment polarity dey accurate, or you *agree* with di scores?
   1. Especially, you agree or disagree with di absolute **positive** polarity of di following sentences?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Di next 3 sentences get absolute positive sentiment, but if you read dem well, dem no be positive sentences. Why sentiment analysis think say dem be positive sentences?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. You agree or disagree with di absolute **negative** polarity of di following sentences?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Any person wey sabi Jane Austen go understand say she dey use her books to talk about di funny parts of English Regency society. Elizabeth Bennett, di main character for *Pride and Prejudice*, na sharp observer of society (like di author) and di way she dey talk dey full of meaning. Even Mr. Darcy (di love interest for di story) notice Elizabeth playful and teasing way of talking: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Challenge

You fit make Marvin better by extracting other features from di user input?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
## Review & Self Study

Plenty ways dey to take find sentiment for text. Make you think about di business wey fit use dis kain technique. Also think about how e fit go wrong. Read more about di advanced enterprise systems wey dey analyze sentiment like [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Try test some of di Pride and Prejudice sentences wey dey above and see if e fit catch di small small meaning.

## Assignment 

[Poetic license](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator) do di translation. Even as we dey try make am accurate, abeg sabi say automated translations fit get mistake or no dey correct well. Di original dokyument for im native language na di main source wey you go fit trust. For important information, e good make professional human translation dey use. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because you use dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->