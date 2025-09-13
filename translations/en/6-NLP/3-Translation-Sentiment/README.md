<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-06T11:02:34+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "en"
}
-->
# Translation and sentiment analysis with ML

In the previous lessons, you learned how to create a basic bot using `TextBlob`, a library that incorporates machine learning behind the scenes to perform basic NLP tasks like extracting noun phrases. Another significant challenge in computational linguistics is accurately _translating_ a sentence from one spoken or written language to another.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Translation is a very complex problem, made even harder by the fact that there are thousands of languages, each with its own unique grammar rules. One approach is to convert the formal grammar rules of one language, such as English, into a structure that is not dependent on any specific language, and then translate it by converting it back into another language. This approach involves the following steps:

1. **Identification**: Identify or tag the words in the input language as nouns, verbs, etc.
2. **Create translation**: Generate a direct translation of each word in the format of the target language.

### Example sentence, English to Irish

In 'English,' the sentence _I feel happy_ consists of three words in the following order:

- **subject** (I)
- **verb** (feel)
- **adjective** (happy)

However, in the 'Irish' language, the same sentence follows a very different grammatical structure—emotions like "*happy*" or "*sad*" are expressed as being *upon* you.

The English phrase `I feel happy` in Irish would be `Tá athas orm`. A *literal* translation would be `Happy is upon me`.

An Irish speaker translating to English would say `I feel happy`, not `Happy is upon me`, because they understand the meaning of the sentence, even though the words and sentence structure differ.

The formal order for the sentence in Irish is:

- **verb** (Tá or is)
- **adjective** (athas, or happy)
- **subject** (orm, or upon me)

## Translation

A simple translation program might translate words individually, ignoring the sentence structure.

✅ If you've learned a second (or third or more) language as an adult, you might have started by thinking in your native language, translating a concept word by word in your head to the second language, and then speaking out your translation. This is similar to what simple translation computer programs do. It's important to move beyond this phase to achieve fluency!

Simple translation often results in poor (and sometimes amusing) mistranslations: `I feel happy` translates literally to `Mise bhraitheann athas` in Irish. That means (literally) `me feel happy` and is not a valid Irish sentence. Even though English and Irish are spoken on two closely neighboring islands, they are very different languages with distinct grammar structures.

> You can watch some videos about Irish linguistic traditions, such as [this one](https://www.youtube.com/watch?v=mRIaLSdRMMs).

### Machine learning approaches

So far, you've learned about the formal rules approach to natural language processing. Another approach is to ignore the meaning of the words and _instead use machine learning to detect patterns_. This can work in translation if you have a large amount of text (a *corpus*) or texts (*corpora*) in both the source and target languages.

For example, consider the case of *Pride and Prejudice*, a famous English novel written by Jane Austen in 1813. If you compare the book in English with a human translation of the book in *French*, you could identify phrases in one language that are _idiomatically_ translated into the other. You'll try this shortly.

For instance, when an English phrase like `I have no money` is translated literally into French, it might become `Je n'ai pas de monnaie`. "Monnaie" is a tricky French 'false cognate,' as 'money' and 'monnaie' are not synonymous. A better translation that a human might make would be `Je n'ai pas d'argent`, because it better conveys the meaning that you have no money (rather than 'loose change,' which is the meaning of 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Image by [Jen Looper](https://twitter.com/jenlooper)

If a machine learning model has access to enough human translations to build a model, it can improve the accuracy of translations by identifying common patterns in texts that have been previously translated by expert human speakers of both languages.

### Exercise - translation

You can use `TextBlob` to translate sentences. Try the famous first line of **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` does a pretty good job at the translation: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

It can be argued that TextBlob's translation is far more precise, in fact, than the 1932 French translation of the book by V. Leconte and Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

In this case, the translation informed by machine learning does a better job than the human translator, who unnecessarily adds words to the original author's text for 'clarity.'

> What's happening here? Why is TextBlob so good at translation? Well, behind the scenes, it's using Google Translate, a sophisticated AI capable of analyzing millions of phrases to predict the best strings for the task at hand. There's nothing manual happening here, and you need an internet connection to use `blob.translate`.

✅ Try some more sentences. Which is better, machine learning or human translation? In which cases?

## Sentiment analysis

Another area where machine learning excels is sentiment analysis. A non-machine learning approach to sentiment analysis involves identifying words and phrases that are 'positive' or 'negative.' Then, given a new piece of text, the total value of positive, negative, and neutral words is calculated to determine the overall sentiment.

This approach can be easily fooled, as you may have seen in the Marvin task—the sentence `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` is sarcastic and negative, but the simple algorithm detects 'great,' 'wonderful,' and 'glad' as positive and 'waste,' 'lost,' and 'dark' as negative. The overall sentiment is skewed by these conflicting words.

✅ Pause for a moment and think about how we convey sarcasm as human speakers. Tone inflection plays a significant role. Try saying the phrase "Well, that film was awesome" in different ways to see how your voice conveys meaning.

### ML approaches

The machine learning approach involves manually gathering negative and positive bodies of text—tweets, movie reviews, or anything where a human has provided both a score *and* a written opinion. NLP techniques can then be applied to these opinions and scores, allowing patterns to emerge (e.g., positive movie reviews might frequently include the phrase 'Oscar worthy,' while negative reviews might use 'disgusting' more often).

> ⚖️ **Example**: Imagine you work in a politician's office, and a new law is being debated. Constituents might write emails either supporting or opposing the law. If you're tasked with reading and sorting these emails into two piles, *for* and *against*, you might feel overwhelmed by the sheer volume. Wouldn't it be helpful if a bot could read and sort them for you? 
> 
> One way to achieve this is by using machine learning. You would train the model with a sample of *for* and *against* emails. The model would associate certain phrases and words with the respective categories, *but it wouldn't understand the content itself*. You could test the model with emails it hadn't seen before and compare its conclusions to your own. Once satisfied with the model's accuracy, you could process future emails without reading each one.

✅ Does this process sound similar to methods you've used in previous lessons?

## Exercise - sentimental sentences

Sentiment is measured with a *polarity* score ranging from -1 to 1, where -1 represents the most negative sentiment and 1 represents the most positive. Sentiment is also measured with a score for objectivity (0) and subjectivity (1).

Take another look at Jane Austen's *Pride and Prejudice*. The text is available here at [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). The sample below shows a short program that analyzes the sentiment of the first and last sentences of the book and displays their sentiment polarity and subjectivity/objectivity scores.

You should use the `TextBlob` library (described above) to determine `sentiment` (you don't need to write your own sentiment calculator) for the following task.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

You see the following output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Challenge - check sentiment polarity

Your task is to determine, using sentiment polarity, whether *Pride and Prejudice* contains more absolutely positive sentences than absolutely negative ones. For this task, you may assume that a polarity score of 1 or -1 represents absolute positivity or negativity, respectively.

**Steps:**

1. Download a [copy of Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) from Project Gutenberg as a .txt file. Remove the metadata at the beginning and end of the file, leaving only the original text.
2. Open the file in Python and extract the contents as a string.
3. Create a TextBlob using the book string.
4. Analyze each sentence in the book in a loop:
   1. If the polarity is 1 or -1, store the sentence in an array or list of positive or negative messages.
5. At the end, print out all the positive sentences and negative sentences (separately) and the count of each.

Here is a sample [solution](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Knowledge Check

1. Sentiment is based on the words used in the sentence, but does the code *understand* the words?
2. Do you think the sentiment polarity is accurate? In other words, do you *agree* with the scores?
   1. Specifically, do you agree or disagree with the absolute **positive** polarity of the following sentences?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. The next 3 sentences were scored with an absolute positive sentiment, but upon closer reading, they are not positive sentences. Why did the sentiment analysis think they were positive sentences?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Do you agree or disagree with the absolute **negative** polarity of the following sentences?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Any fan of Jane Austen will understand that she often uses her books to critique the more absurd aspects of English Regency society. Elizabeth Bennett, the main character in *Pride and Prejudice*, is a sharp social observer (like the author), and her language is often heavily nuanced. Even Mr. Darcy (the love interest in the story) notes Elizabeth's playful and teasing use of language: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Challenge

Can you make Marvin even better by extracting other features from the user input?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study
There are many ways to determine sentiment in text. Consider the business applications that could benefit from this approach. Reflect on how it might fail or produce unintended results. Learn more about advanced, enterprise-grade systems for sentiment analysis, such as [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Try analyzing some of the sentences from Pride and Prejudice mentioned earlier and see if it can pick up on subtle nuances.

## Assignment

[Poetic license](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may include errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is advised. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.