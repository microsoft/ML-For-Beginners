<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-06T11:02:08+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "en"
}
-->
# Introduction to natural language processing

This lesson provides a brief history and key concepts of *natural language processing*, a subfield of *computational linguistics*.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

NLP, as it is commonly called, is one of the most prominent areas where machine learning has been applied and integrated into production software.

✅ Can you think of software you use daily that likely incorporates NLP? Consider your word processing programs or mobile apps you use regularly.

You will learn about:

- **The concept of languages**. How languages evolved and the major areas of study.
- **Definitions and concepts**. You will also explore definitions and concepts related to how computers process text, including parsing, grammar, and identifying nouns and verbs. This lesson includes coding tasks and introduces several important concepts that you will learn to code in subsequent lessons.

## Computational linguistics

Computational linguistics is a field of research and development spanning decades, focused on how computers can work with, understand, translate, and communicate using languages. Natural language processing (NLP) is a related field that specifically examines how computers can process 'natural', or human, languages.

### Example - phone dictation

If you've ever dictated to your phone instead of typing or asked a virtual assistant a question, your speech was converted into text and then processed or *parsed* from the language you spoke. The identified keywords were then processed into a format the phone or assistant could understand and act upon.

![comprehension](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Real linguistic comprehension is hard! Image by [Jen Looper](https://twitter.com/jenlooper)

### How is this technology made possible?

This is possible because someone wrote a computer program to enable it. A few decades ago, science fiction writers predicted that people would primarily speak to their computers, and the computers would always understand exactly what they meant. Unfortunately, this turned out to be a much harder problem than many imagined. While the problem is better understood today, achieving 'perfect' natural language processing—especially when it comes to understanding the meaning of a sentence—remains a significant challenge. This is particularly difficult when trying to interpret humor or detect emotions like sarcasm in a sentence.

You might recall school lessons where teachers covered grammar components in a sentence. In some countries, grammar and linguistics are taught as a dedicated subject, while in others, these topics are integrated into language learning: either your first language in primary school (learning to read and write) or perhaps a second language in high school. Don't worry if you're not an expert at distinguishing nouns from verbs or adverbs from adjectives!

If you struggle with the difference between the *simple present* and *present progressive*, you're not alone. This is challenging for many people, even native speakers of a language. The good news is that computers excel at applying formal rules, and you'll learn to write code that can *parse* a sentence as well as a human. The greater challenge you'll explore later is understanding the *meaning* and *sentiment* of a sentence.

## Prerequisites

For this lesson, the main prerequisite is being able to read and understand the language of this lesson. There are no math problems or equations to solve. While the original author wrote this lesson in English, it has been translated into other languages, so you might be reading a translation. Examples include several different languages (to compare grammar rules across languages). These examples are *not* translated, but the explanatory text is, so the meaning should be clear.

For the coding tasks, you'll use Python, and the examples are based on Python 3.8.

In this section, you will need and use:

- **Python 3 comprehension**. Understanding the Python 3 programming language, including input, loops, file reading, and arrays.
- **Visual Studio Code + extension**. We'll use Visual Studio Code and its Python extension. You can also use a Python IDE of your choice.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) is a simplified text processing library for Python. Follow the instructions on the TextBlob site to install it on your system (install the corpora as well, as shown below):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: You can run Python directly in VS Code environments. Check the [docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) for more information.

## Talking to machines

The history of trying to make computers understand human language spans decades, and one of the earliest scientists to explore natural language processing was *Alan Turing*.

### The 'Turing test'

When Turing researched *artificial intelligence* in the 1950s, he proposed a conversational test where a human and computer (via typed correspondence) would interact, and the human participant would try to determine whether they were conversing with another human or a computer.

If, after a certain length of conversation, the human could not distinguish whether the responses came from a computer or another human, could the computer be said to be *thinking*?

### The inspiration - 'the imitation game'

This idea was inspired by a party game called *The Imitation Game*, where an interrogator in one room tries to determine which of two people (in another room) is male and which is female. The interrogator sends notes and attempts to ask questions that reveal the gender of the mystery person. Meanwhile, the players in the other room try to mislead or confuse the interrogator while appearing to answer honestly.

### Developing Eliza

In the 1960s, an MIT scientist named *Joseph Weizenbaum* developed [*Eliza*](https://wikipedia.org/wiki/ELIZA), a computer 'therapist' that asked humans questions and gave the impression of understanding their answers. However, while Eliza could parse a sentence and identify certain grammatical constructs and keywords to provide reasonable responses, it could not truly *understand* the sentence. For example, if Eliza was presented with a sentence like "**I am** <u>sad</u>", it might rearrange and substitute words to form the response "How long have **you been** <u>sad</u>?"

This gave the impression that Eliza understood the statement and was asking a follow-up question, but in reality, it was simply changing the tense and adding some words. If Eliza couldn't identify a keyword it had a response for, it would provide a random response applicable to many different statements. Eliza could be easily tricked; for instance, if a user wrote "**You are** a <u>bicycle</u>", it might respond with "How long have **I been** a <u>bicycle</u>?" instead of a more logical reply.

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 Click the image above for a video about the original ELIZA program

> Note: You can read the original description of [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) published in 1966 if you have an ACM account. Alternatively, read about Eliza on [Wikipedia](https://wikipedia.org/wiki/ELIZA).

## Exercise - coding a basic conversational bot

A conversational bot, like Eliza, is a program that elicits user input and appears to understand and respond intelligently. Unlike Eliza, our bot will not have multiple rules to simulate intelligent conversation. Instead, it will have one simple ability: to keep the conversation going with random responses that could fit almost any trivial conversation.

### The plan

Steps to build a conversational bot:

1. Print instructions advising the user how to interact with the bot.
2. Start a loop:
   1. Accept user input.
   2. If the user asks to exit, then exit.
   3. Process user input and determine a response (in this case, the response is a random choice from a list of generic responses).
   4. Print the response.
3. Loop back to step 2.

### Building the bot

Let's create the bot. We'll start by defining some phrases.

1. Create this bot yourself in Python with the following random responses:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Here is some sample output to guide you (user input is on the lines starting with `>`):

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

    One possible solution to the task is [here](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py).

    ✅ Stop and consider:

    1. Do you think the random responses would 'trick' someone into thinking the bot actually understood them?
    2. What features would the bot need to be more effective?
    3. If a bot could truly 'understand' the meaning of a sentence, would it also need to 'remember' the meaning of previous sentences in a conversation?

---

## 🚀Challenge

Choose one of the "stop and consider" elements above and either try to implement it in code or write a solution on paper using pseudocode.

In the next lesson, you'll explore other approaches to parsing natural language and machine learning.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Explore the references below for further reading.

### References

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.

## Assignment 

[Search for a bot](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.