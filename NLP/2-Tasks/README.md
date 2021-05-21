# Common Natural Language Processing Tasks and Techniques

Add a sketchnote if possible/appropriate

![Embed a video here if available](video-url)

## [Pre-lecture quiz](link-to-quiz-app)

For most *Natural Language Processing* tasks, the text to be processed must be broken down, examined, and the results stored or cross referenced with rules and data sets. This allows the programmer to derive the meaning or intent or even just the frequency of terms and words in a text. 

Let's discover common techniques used in processing text. Combined with machine learning, these techniques help you to analyse large amounts of text efficiently.
## The Tools of NLP

> 🎓 **Tokenization**
> 
> Probably the first thing most NLP algorithms have to do is split the text into tokens, or words. While this sounds simple, having to take punctuation and different language's word and sentence delimiters can make it tricky.

> 🎓 **Parsing & Part-of-speech Tagging**
>
> Every word that has been tokenised can be tagged as a part of speech - is the word a noun, a verb, or adjective etc. The sentence `the quick red fox jumped over the lazy brown dog` might be POS tagged as *fox* = noun, *jumped* = verb etc.
>
> Parsing is recognising what words are related to each other in a sentence - for instance `the quick red fox jumped` is an adjective-noun-verb sequence that is is separate from `lazy brown dog` sequence.  

> 🎓 **Word and Phrase Frequencies**
>
> A useful tool when analysing a large body of text is to build a dictionary of every word or phrase of interest and how often it appears. The phrase `the quick red fox jumped over the lazy brown dog` has a word frequency of 2 for `the`. 

### Example:

The Rudyard Kipling poem *The Winners* has a verse:

```
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

As phrase frequencies can be case insensitive or case sensitive as required, the phrase `a friend` has a frequency of 2 and `the` has a frequency of 6, and `travels` is 2.

> 🎓 **N-grams**
>
> A text can be split into sequences of words of a set length, a single word (unigram), two words (bigrams), three words (trigrams) or any number of words (n-grams).

### Example

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

> 🎓 **Noun phrase Extraction**
>
> In most sentences, there is a noun that is the subject, or object of the sentence. In English, it is often identifiable as having 'a' or 'an' or 'the' preceding it. Identifying the subject or object of a sentence by 'extracting the noun phrase' is a common task in NLP when attempting to understand the meaning of a sentence.
### Example

In the sentence `the quick red fox jumped over the lazy brown dog` there are 2 noun phrases: **quick red fox** and **lazy brown dog**.

> 🎓 **Sentiment analysis**
>
> A sentence or text can be analysed for sentiment, or how *positive* or *negative* it is. Sentiment is measured in *polarity* and *objectivity/subjectivity*. Polarity is measured from -1.0 to 1.0 (negative to positive) and 0.0 to 1.0 (most objective to most subjective). 

✅ Later you'll learn that there are different ways to determine sentiment using machine learning, but one way is to have a list of words and phrases that are categorised as positive or negative by a human expert and apply that model to text to calculate a polarity score. Can you see how this would work in some circumstances and less well in others?

> 🎓 **WordNet**
>
> [WordNet](https://wordnet.princeton.edu/) is a database of words, synonyms, antonyms and many other details for every word in many different languages. It is incredibly useful when attempting to build translations, spell checkers, or language tools of any type.

> 🎓 **Inflection**
>
> Inflection enables you to take a word and get the singular or plural of the word.

> 🎓 **Lemmatization**
>
> A *lemma* is the root or headword for a set of words, for instance *flew*, *flies*, *flying* have a lemma of the verb *fly*.  

## TextBlob & NLTK

Luckily, you don't have to build all of these techniques yourself, as there are excellent Python libraries available that make it much more accessible to developers who aren't specialised in natural language processing or machine learning. The next lesson includes more examples on these, but here you will learn some useful examples to help you with the next task.

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
## Task: Improving your bot with a little NLP

In the previous lesson you built a very simple Q&A bot. Now, you'll make Marvin a bit more sympathetic by analyzing your input for sentiment and printing out a response to match the sentiment. You'll also need to identify a `noun_phrase` and ask about it.

Your steps when building a better conversational bot:

1. Print instructions advising the user how to interact with the bot
2. Start loop 
   1. Accept user input
   2. If user has asked to exit, then exit
   3. Process user input and determine appropriate sentiment response
   4. If a noun phrase is detected in the sentiment, pluralize it and ask for more input on that topic
   5. Print response
3. loop back to step 2

Here is the code snippet to determine sentiment using TextBlob. Note there are only four *gradients* of sentiment response (you could have more if you like):

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
One possible solution to the task is [here](solution/bot.py)

✅ Knowledge Check

1. Do you think the sympathetic responses would 'trick' someone into thinking that the bot actually understood them?
2. Does identifying the noun phrase make the bot more 'believable'?
3. Why would extracting a 'noun phrase' from a sentence a useful thing to do?
## 🚀Challenge

TODO
## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

TODO

**Assignment**: [Assignment Name](assignment.md)
