### Translation

An important challenge in computational linguistics is accurate *translation* of a sentence from one spoken or written language to another. This is a very hard problem compounded by the fact that there are thousands of languages and each can have very different grammar rules. One approach is to convert the formal grammar rules for one language, such as English, into a non-language dependent structure, and then translate it by converting back to another language. This means that you would take the following steps:

1. Identify or tag the words in input language into nouns, verbs etc.
2. Produce a direct translation of each word in the target language format

Example: In **English**, the simple sentence `I feel happy` is 3 words in the order **subject** (I), **verb** (feel), **adjective** (happy). However, in the **Irish** language, the same sentence has a very different grammatical structure - emotions like "*happy*" or "*sad*" are expressed as being *upon* you. The English phrase `I feel happy` in Irish would be `Tá athas orm`. A *literal* translation would be `Happy is upon me`. Of course, an Irish speaker translating to English would say `I feel happy`, not `Happy is upon me`, because they understand the meaning of the sentence, even if the words and sentence structure are different. The formal order for the sentence in Irish are **verb** (Tá or is), **adjective** (athas, or happy), **subject** (orm, or upon me).

A simple (and poor) translation program might translate words only, ignoring the sentence structure. This leads to bad (and sometimes hilarious) mistranslations: `I feel happy` translates literally to `Mise bhraitheann athas`. In Irish that means (literally) `me feel happy` and not a valid Irish sentence. Even though English and Irish are languages spoken on two adjoining islands, they are very different languages with different grammar structures. 

### Machine Learning Approaches

So far, you've learned about the formal rules approach to natural language processing. Another approach is to ignore the meaning of the words, and instead use machine learning to detect patterns. This can work in translation if you have lots of text (a *corpus*) or texts (*corpora*) in both the origin and target languages. For instance, if you have *Pride and Prejudice* in English and a human translation of the book in *French*, you could detect phrases in one are idiomatically translated into the other. 

For instance, when an English phrase such as `John looked at the cake with a wolfish grin` is translated literally, to, say French, it might become `John regarda le gâteau avec un sourire de loup`. A reader of both languages would understand that the direct translation of `wolfish grin` is not the French translation `wolf smile` but a synonym - in this case for being very hungry or voracious. A better translation that a human might make would be `John regarda le gâteau avec voracité`, because it better conveys the meaning. If a ML model has enough human translations to build a model on, it can improve the accuracy of translations by identifying common patterns in texts that have been previously translated by expert human speakers of both languages. 

Another area where machine learning can work very well is sentiment analysis. A non-ML approach to sentiment is to identify words and phrases which are 'positive' and 'negative'. Then, given a new piece of text, calculate the total value of the positive, negative and neutral words to identify the overall sentiment. This approach is easily tricked as you may have seen in the Marvin task - the sentence `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` is a sarcastic, negative sentiment sentence, but the simple algorithm detects 'great', 'wonderful', 'glad' as positive and 'waste', 'lost' and 'dark' as negative. The overall sentiment is swayed by these conflicting words.

The ML approach would be to hand gather negative and positive bodies of text - tweets, or movie reviews, or anything where the human has given a score *and* a written opinion. Then NLP techniques can be applied to opinions and scores, so that patterns emerge (e.g., positive movie reviews tend to have the phrase 'Oscar worthy' more than negative movie reviews, or positive restaurant reviews say 'gourmet' much more than 'disgusting').

**Example**: If you worked in a politician's office and there was some new law being debated, constituents might write to the office with emails supporting or emails against the particular new law. Let's say you are tasked with reading the emails and sorting them in 2 piles, *for* and *against*. If there were a lot of emails, you might be overwhelmed attempting to read them all. Wouldn't it be nice if a bot could read them all for you, understand them and tell you which pile each email belonged? One way to achieve that is to use Machine Learning. You would train the model with a portion of the *against* emails and a portion of the *for* emails. The model would tend to associate phrases and words with the against side and the for side, *but it would not understand any of the content*, only that certain words and patterns were more likely to appear in an *against* or a *for* email. You could test it with some emails that you had not used to train the model, and see if it came to the same conclusion as you did. Then, once you were happy with the accuracy of the model, you could process future emails without having to read each one.

### Task: Sentimental Sentences

Sentiment is measured in with a *polarity* of -1 to 1, meaning -1 is the most negative sentiment, and 1 is the most positive. Sentiment is also measured with an 0 - 1 score for objectivity (0) and subjectivity (1).

**Pride and Prejudice**, by **Jane Austen** is an English novel with some famous lines. The text is available here at [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). The sample below shows a short program which analyses the sentiment of first and last sentences from the book and display its sentiment polarity and subjectivity/objectivity score. You should us the TextBlob library (described above) to determine sentiment (you do not have to write your own sentiment calculator) in the following task.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
# outputs:
# It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)
# Darcy, as well as Elizabeth, really loved them; and they were
#     both ever sensible of the warmest gratitude towards the persons
#      who, by bringing her into Derbyshire, had been the means of
#      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

Your task is to determine, using sentiment polarity, if *Pride and Prejudice* has more absolutely positive sentences than absolutely negative ones. For this task, you may assume that a polarity score of 1 or -1 is absolutely positive or negative respectively. 

Steps:

1. Download a copy of Pride and Prejudice from Project Gutenberg as a .txt file. Remove the metadata at the start and end of the file, leaving only the original text
2. Open the file in Python and extract the contents as a string
3. Create a TextBlob using the book string
4. Analyse each sentence in the book in a loop
   1. If the polarity is 1 or -1 store the sentence in an array or list of positive or negative messages
5. At the end, print out all the positive sentences and negative sentences (separately) and the number of each.

Here is a sample [solution](solutions/lesson1_task3.py).

✅ Knowledge Check 

1. The sentiment is based on words used in the sentence, but does it code *understand* the words?
2. Do you think the sentiment polarity is accurate, or in other words, do you *agree* with the scores?
   1. In particular, do you agree or disagree with the absolute **positive** polarity of the following sentences?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. The next 3 sentences were scored with an absolute positive sentiment, but on close reading, they are not positive sentences. Why did the sentiment analysis think they were positive sentences?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Do you agree or disagree with the absolute **negative** polarity of the following sentences?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!



🚀 Challenge: Can you make Marvin even better by extracting other features from the user input?
