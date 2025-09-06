<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-06T11:00:38+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "en"
}
-->
# Common natural language processing tasks and techniques

For most *natural language processing* tasks, the text to be processed must be broken down, analyzed, and the results stored or cross-referenced with rules and datasets. These tasks allow the programmer to derive the _meaning_, _intent_, or simply the _frequency_ of terms and words in a text.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Let's explore common techniques used in processing text. Combined with machine learning, these techniques help you analyze large amounts of text efficiently. Before applying ML to these tasks, however, let's understand the challenges faced by an NLP specialist.

## Tasks common to NLP

There are various ways to analyze a text you are working on. These tasks help you understand the text and draw conclusions. Typically, these tasks are performed in a sequence.

### Tokenization

The first step most NLP algorithms take is splitting the text into tokens, or words. While this sounds simple, accounting for punctuation and different languages' word and sentence delimiters can make it challenging. You may need to use different methods to determine boundaries.

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizing a sentence from **Pride and Prejudice**. Infographic by [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) are a way to convert text data into numerical form. Embeddings are designed so that words with similar meanings or words often used together are grouped closely.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings for a sentence in **Pride and Prejudice**. Infographic by [Jen Looper](https://twitter.com/jenlooper)

✅ Try [this interesting tool](https://projector.tensorflow.org/) to experiment with word embeddings. Clicking on one word shows clusters of similar words: 'toy' clusters with 'disney', 'lego', 'playstation', and 'console'.

### Parsing & Part-of-speech Tagging

Every tokenized word can be tagged as a part of speech, such as a noun, verb, or adjective. For example, the sentence `the quick red fox jumped over the lazy brown dog` might be POS tagged as fox = noun, jumped = verb.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing a sentence from **Pride and Prejudice**. Infographic by [Jen Looper](https://twitter.com/jenlooper)

Parsing involves identifying relationships between words in a sentence—for instance, `the quick red fox jumped` is an adjective-noun-verb sequence that is separate from the `lazy brown dog` sequence.

### Word and Phrase Frequencies

A useful technique for analyzing large texts is building a dictionary of every word or phrase of interest and tracking how often it appears. For example, in the phrase `the quick red fox jumped over the lazy brown dog`, the word "the" appears twice.

Consider an example text where we count word frequencies. Rudyard Kipling's poem *The Winners* contains the following verse:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Depending on whether phrase frequencies are case-sensitive or case-insensitive, the phrase `a friend` has a frequency of 2, `the` has a frequency of 6, and `travels` appears twice.

### N-grams

Text can be divided into sequences of words of a fixed length: single words (unigrams), pairs of words (bigrams), triplets (trigrams), or any number of words (n-grams).

For example, the phrase `the quick red fox jumped over the lazy brown dog` with an n-gram score of 2 produces the following n-grams:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

It can be visualized as a sliding box over the sentence. For n-grams of 3 words, the n-gram is highlighted in bold in each sentence:

1.   **the quick red** fox jumped over the lazy brown dog
2.   the **quick red fox** jumped over the lazy brown dog
3.   the quick **red fox jumped** over the lazy brown dog
4.   the quick red **fox jumped over** the lazy brown dog
5.   the quick red fox **jumped over the** lazy brown dog
6.   the quick red fox jumped **over the lazy** brown dog
7.   the quick red fox jumped over **the lazy brown** dog
8.   the quick red fox jumped over the **lazy brown dog**

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram value of 3: Infographic by [Jen Looper](https://twitter.com/jenlooper)

### Noun phrase Extraction

In most sentences, there is a noun that serves as the subject or object. In English, it is often preceded by 'a', 'an', or 'the'. Identifying the subject or object of a sentence by extracting the noun phrase is a common NLP task when trying to understand the meaning of a sentence.

✅ In the sentence "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", can you identify the noun phrases?

In the sentence `the quick red fox jumped over the lazy brown dog`, there are 2 noun phrases: **quick red fox** and **lazy brown dog**.

### Sentiment analysis

A sentence or text can be analyzed for sentiment, or how *positive* or *negative* it is. Sentiment is measured in terms of *polarity* and *objectivity/subjectivity*. Polarity ranges from -1.0 to 1.0 (negative to positive), while objectivity ranges from 0.0 to 1.0 (most objective to most subjective).

✅ Later you'll learn that there are different ways to determine sentiment using machine learning, but one approach is to use a list of words and phrases categorized as positive or negative by a human expert and apply that model to text to calculate a polarity score. Can you see how this would work in some cases but not in others?

### Inflection

Inflection allows you to take a word and determine its singular or plural form.

### Lemmatization

A *lemma* is the root or base form of a word. For example, *flew*, *flies*, and *flying* all have the lemma *fly*.

There are also useful databases available for NLP researchers, such as:

### WordNet

[WordNet](https://wordnet.princeton.edu/) is a database of words, synonyms, antonyms, and other details for many words in various languages. It is incredibly useful for building translations, spell checkers, or any type of language tool.

## NLP Libraries

Fortunately, you don't have to build all these techniques from scratch. There are excellent Python libraries available that make NLP more accessible to developers who aren't specialized in natural language processing or machine learning. The next lessons will include more examples, but here are some useful examples to help you with the next task.

### Exercise - using `TextBlob` library

Let's use a library called TextBlob, which contains helpful APIs for tackling these types of tasks. TextBlob "stands on the giant shoulders of [NLTK](https://nltk.org) and [pattern](https://github.com/clips/pattern), and plays nicely with both." It incorporates a significant amount of ML into its API.

> Note: A useful [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) guide is available for TextBlob and is recommended for experienced Python developers.

When attempting to identify *noun phrases*, TextBlob offers several extractors to find them.

1. Take a look at `ConllExtractor`.

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

    > What's happening here? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) is "A noun phrase extractor that uses chunk parsing trained with the ConLL-2000 training corpus." ConLL-2000 refers to the 2000 Conference on Computational Natural Language Learning. Each year, the conference hosted a workshop to tackle a challenging NLP problem, and in 2000, it focused on noun chunking. A model was trained on the Wall Street Journal, using "sections 15-18 as training data (211727 tokens) and section 20 as test data (47377 tokens)." You can review the procedures [here](https://www.clips.uantwerpen.be/conll2000/chunking/) and the [results](https://ifarm.nl/erikt/research/np-chunking.html).

### Challenge - improving your bot with NLP

In the previous lesson, you built a simple Q&A bot. Now, you'll make Marvin a bit more empathetic by analyzing your input for sentiment and printing a response to match the sentiment. You'll also need to identify a `noun_phrase` and ask about it.

Steps to build a better conversational bot:

1. Print instructions advising the user how to interact with the bot.
2. Start a loop:
   1. Accept user input.
   2. If the user asks to exit, then exit.
   3. Process user input and determine an appropriate sentiment response.
   4. If a noun phrase is detected in the sentiment, pluralize it and ask for more input on that topic.
   5. Print a response.
3. Loop back to step 2.

Here is the code snippet to determine sentiment using TextBlob. Note that there are only four *gradients* of sentiment response (you can add more if you like):

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

Here is some sample output to guide you (user input starts with >):

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

One possible solution to the task is [here](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Knowledge Check

1. Do you think the empathetic responses would 'trick' someone into thinking the bot actually understood them?
2. Does identifying the noun phrase make the bot more 'believable'?
3. Why would extracting a 'noun phrase' from a sentence be a useful thing to do?

---

Implement the bot in the prior knowledge check and test it on a friend. Can it trick them? Can you make your bot more 'believable'?

## 🚀Challenge

Take a task from the prior knowledge check and try to implement it. Test the bot on a friend. Can it trick them? Can you make your bot more 'believable'?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

In the next few lessons, you will learn more about sentiment analysis. Research this fascinating technique in articles such as those on [KDNuggets](https://www.kdnuggets.com/tag/nlp).

## Assignment 

[Make a bot talk back](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.