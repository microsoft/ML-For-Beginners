# Natural Language Processing

Add a sketchnote if possible/appropriate

![Embed a video here if available](video-url)

## [Pre-lecture quiz](link-to-quiz-app)

## Introduction

This section covers a brief history and important concepts of *Computational Linguistics* focusing on *Natural Language Processing*. 
You will learn about how the ideas about languages developed and what the major areas of study have been. 
You will also learn definitions and concepts about how computers process text, including parsing, grammar, and identifying nouns and verbs. There are some coding tasks in this lesson, and several important concepts are introduced that you will learn to code later on in the next lessons. 

Computational linguistics is an area of research and development over many decades that studies how computers can work with, and even understand, translate, and communicate with languages. Natural Language Processing (NLP) is a related field focused on how computers can process 'natural', or human, languages. If you have ever dictated to your phone instead of typing, or asked a virtual assistant a question, your speech was converted into a text form and then processed or *parsed* from the language you spoke into a format that the phone or assistant could understand and act on. This is possible because a coder wrote a program to do this. A few decades ago, some science fiction writers predicted that people would mostly speak to their computers, and the computers would always understand exactly what they meant. Sadly, it turned out to be a harder problem that many imagined, and while it is a much better understood problem today, there are significant challenges in achieving 'perfect' natural language processing when it comes to understanding the meaning of a sentence. This is a particularly hard problem when it comes to understanding humour, or detecting emotions such as sarcasm in a sentence.

At this point, you may be remembering school classes where the teacher covered the parts of grammar in a sentence. In some countries, students are taught grammar and linguistics as a dedicated subject, but in many, these topics are included as part of learning a language: either your first language in primary school (learning to read and write) and perhaps a second language in post-primary, or high school. Don't  worry if you are not an expert at differentiating nouns from verbs, or adverbs from adjectives! If you struggle with the difference between the *simple present* and *present progressive*, you are not alone. This is a challenging thing for many people, even native speakers of a language. The good news is that computers are really good at applying formal rules, and you will learn to write code that can *parse* a sentence as well as a human. The greater challenge you will examine later is understanding the *meaning*, and *sentiment*, of a sentence.

## Prerequisites

For this lesson, the main prerequisite is being able to read and understand the language of this lesson. There are no maths or equations to handle. While I happen to be writing this lesson in English, it is also translated into other languages, so you could be reading a translation. There are examples where a number of different languages are used (to compare the different grammar rules of different languages). These are *not* translated, but the explanatory text is, so the meaning should be clear.

For the coding tasks, you will use Python and the examples are using Python 3.8.

In this section, you will need:
* Python 3 programming language 
  * this lesson uses input, loops, file reading, arrays
* Visual Studio Code & the Python extension
  * (*or the Python IDE of your choice*)
* [TextBlob](https://github.com/sloria/TextBlob) a simplified text processing library for Python
  * Follow the instructions on the TextBlob site to install it on your system (install the corpora as well as shown below)
    ```bash
    pip install -U textblob
    python -m textblob.download_corpora
    ```

## Conversing with Eliza

The history of trying to make computers understand human language goes back decades, and one of the earliest scientists to consider natural language processing was *Alan Turing*. When Turing was researching *Artificial Intelligence* in the 1950's, he considered if a conversational test could be given to a human and computer (via typed correspondence) where the human in the conversation was not sure if they were conversing with another human or a computer. If, after a certain length of conversation, the human could not determine that the answers were from a computer or not, then could the computer be said to be *thinking*? The idea for this came from a party game called *The Imitation Game* where an interrogator is alone in a room and tasked with determining which of two people (in another room) are male and female respectively. The interrogator can send notes, and must try to think of questions where the written answers reveal the gender of the mystery person. Of course, the players in the other room are trying to trick the interrogator by answering questions in such as way as to mislead or confuse the interrogator, whilst also giving the appearance of answering honestly.

In the 1960's an MIT scientist called *Joseph Weizenbaum* developed [*Eliza*](https://en.wikipedia.org/wiki/ELIZA), a computer 'therapist' that would ask the human questions and give the appearance of understanding their answers. However, while Eliza could parse a sentence and identify certain grammatical constructs and keywords so as to give a reasonable answer, it could not be said to *understand* the sentence. If Eliza was presented with a sentence following the format "**I am** <u>sad</u>" it might rearrange and substitute words in the sentence to form the response "How long have **you been** <u>sad</u>". This gave the impression that Eliza understood the statement and was asking a follow-on question, whereas in reality, it was changing the tense and adding some words. If Eliza could not identify a keyword that it had a response for, it would instead give a random response that could be applicable to many different statements. Eliza could be easily tricked, for instance if a user wrote "**You are** a <u>bicycle</u>" it might respond with "How long have **I been** a <u>bicycle</u>?", instead of a more reasoned response. 

> Note: You can read the original description of [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) published in 1966 if you have an ACM account

### Task: Coding a simple conversational bot

A conversational bot, like Eliza, is a program that elicits user input and seems to understand and respond intelligently. Unlike Eliza, our bot will not have several rules giving it the appearance of having an intelligent conversation. Instead, out bot will have one ability only, to keep the conversation going with random responses that might work in just about any trivial conversation.

Your steps when building a conversational bot:

1. Print instructions advising the user how to interact with the bot
2. Start a loop
   1. Accept user input
   2. If user has asked to exit, then exit
   3. Process user input and determine response (in this case, the response is a random choice from a list of possible generic responses)
   4. Print response and 
3. loop back to step 2

If you have some Python coding skills, attempt to write this bot yourself with the following random responses:

```python
random_responses = ["That is quite interesting, please tell me more.",
                    "I see. Do go on.",
                    "Why do you say that?",
                    "Funny weather we've been having, isn't it?",
                    "Let's change the subject.",
                    "Did you catch the game last night?"]
```

Here is some sample output to guide you (user input is on the lines with starting with >):

```
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

One possible solution to the task is [here](solutions/lesson1_task1.py)

✅ Knowledge Check
1. Do you think the random responses would 'trick' someone into thinking that the bot actually understood them?
2. What features would the bot need to be more effective?
3. If a bot could really 'understand' the meaning of a sentence, would it need to 'remember' the meaning of previous sentences in a conversation too?
NLP Techniques

In the next lesson, you'll learn about a number of other approaches to parsing natural language and machine learning, but there are a few concepts you should know here first.

For most *Natural Language Processing* tasks, the text to be processed must be broken down, examined, and the results stored or cross referenced with rules and data sets. This allows the programmer to derive the meaning or intent or even just the frequency of terms and words in a text. Here are a list of common techniques used in processing text. You should know these are because they are combined with machine learning techniques to analyse large amounts of text efficiently. In the next lesson, you'll learn how to code some of these.

### Tokenization

Probably the first thing most NLP algorithms have to do is split the text into tokens, or words. While this sounds simple, having to take punctuation and different language's word and sentence delimiters can make it tricky.

### Parsing & Part-of-speech Tagging

Every word that has been tokenised can be tagged as a part of speech - is the word a noun, a verb, or adjective etc. The sentence `the quick red fox jumped over the lazy brown dog` might be POS tagged as *fox* = noun, *jumped* = verb etc.

Parsing is recognising what words are related to each other in a sentence - for instance `the quick red fox jumped` is an adjective-noun-verb sequence that is is separate from `lazy brown dog` sequence.  

### Word and phrase Frequencies

A useful tool when analysing a large body of text is to build a dictionary of every word or phrase of interest and how often it appears. The phrase `the quick red fox jumped over the lazy brown dog` has a word frequency of 2 for `the`. The Rudyard Kipling poem *The Winners* has a verse:

```
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

As phrases frequencies can be case insensitive or case sensitive as required, the phrase `a friend` has a frequency of 2 and `the` has a frequency of 6, and `travels` is 2.

### n-grams

A text can be split into sequences of words of a set length, a single word (unigram), two words (bigrams), three words (trigrams) or any number of words (n-grams).

For instance `the quick red fox jumped over the lazy brown dog` with a n-gram score of 2 produces the following n-grams:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

It might be easier to visualise it as a sliding box over the sentence. Here it is for n-grams of 3 words, the n-gram is in bold in each sentence:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

### Noun phrase Extraction

In most sentences, there is a noun that is the subject, or object of the sentence. In English, it is often identifiable as it has 'a' or 'an' or the preceding it. Identifying the subject or object of a sentence by 'extracting the noun phrase' is a common task in NLP when attempting to understand the meaning of a sentence.

In the sentence `the quick red fox jumped over the lazy brown dog` there are 2 noun phrases: **quick red fox** and **lazy brown dog**.

### Sentiment analysis

A sentence or text can be analysed for sentiment, or how *positive* or *negative* it is. Sentiment is measured in *polarity* and *objectivity/subjectivity*. Polarity is measured from -1.0 to 1.0 (negative to positive) and 0.0 to 1.0 (most objective to most subjective). Later this lesson you'll learn there are different ways to determine sentiment using machine learning, but one way is to have a list of words and phrases that are categorised as positive or negative by a human expert and apply that model to text to calculate a polarity score.

### WordNet, Inflection and lemmatization

[WordNet](https://wordnet.princeton.edu/) is a database of words, synonyms, antonyms and many other details for every word in many different languages. It is incredibly useful when attempting to build translations, spell checkers, or language tools of any type.

Inflection enables you to take a word and get the singular, or plural of the word.

A *lemma* is the root or headword for a set of words, for instance *flew*, *flies*, *flying* have a lemma of the verb *fly*.  

### TextBlob & NLTK

Luckily, you don't have to build all of these techniques yourself, as there are excellent Python libraries available that make it much more accessible to developers who haven't specialised in natural language processing or machine learning. The next lesson includes more examples on these, but here you will learn some useful examples to help you with the next task.

> Note: A useful [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) guide is available for TextBlob that is recommended for experienced Python developers 

When attempting to identify *noun phrases*, the default extractor seems to miss quite a few, but there is the option to use a different one.

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

### Task: Improving your bot with a little NLP

Next, you'll make Marvin a bit more sympathetic by analysing our input for sentiment, and printing out a response to match the sentiment. You'll also need to identify a noun_phrase and ask about it.

Your steps when building a better conversational bot:

1. Print instructions advising the user how to interact with the bot
2. Start loop 
   1. Accept user input
   2. If user has asked to exit, then exit
   3. Process user input and determine appropriate sentiment response
   4. If a noun phrase is detected in the sentiment, pluralize it and ask for more input on that topic
   5. Print response
3. loop back to step 2

Here is the code snippet to determine sentiment using TextBlob, note I only have 4 *gradients* of sentiment response (you could have more if you like):

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

Here is some sample output to guide you (user input is on the lines with starting with >):

```
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

One possible solution to the task is [here](solutions/lesson1_task2.py)

✅ Knowledge Check

1. Do you think the sympathetic responses would 'trick' someone into thinking that the bot actually understood them?
2. Does identifying the noun phrase make the bot more 'believable'?
3. Why would extracting a 'noun phrase' from a sentence a useful thing to do?

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

### [Post-lesson quiz](link-to-quiz-app)

### References

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 
