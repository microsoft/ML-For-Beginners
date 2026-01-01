<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-11-18T18:28:07+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "pcm"
}
-->
# Introduction to natural language processing

Dis lesson go talk small about di history and di important tins wey dey for *natural language processing*, wey be one part of *computational linguistics*.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

NLP, as people dey call am, na one of di popular area wey machine learning don dey use for software wey people dey use.

✅ You fit tink of software wey you dey use everyday wey fit get NLP inside? Wetin of di word processing programs or di mobile apps wey you dey use steady?

You go learn about:

- **Di idea of languages**. How languages take start and di main areas wey people dey study.
- **Definition and concepts**. You go also sabi di definitions and concepts about how computers dey process text, like parsing, grammar, and how to identify nouns and verbs. Dis lesson get some coding tasks, and e go introduce some important concepts wey you go learn how to code later for di next lessons.

## Computational linguistics

Computational linguistics na research and development area wey don dey for many years wey dey study how computers fit work with, understand, translate, and even communicate with languages. Natural language processing (NLP) na di part wey dey focus on how computers fit process 'natural', wey be human languages.

### Example - phone dictation

If you don ever dictate for your phone instead of typing or ask virtual assistant question, di speech wey you talk go turn text and dem go process or *parse* di language wey you talk. Di keywords wey dem detect go then turn format wey di phone or assistant fit understand and act on.

![comprehension](../../../../translated_images/comprehension.619708fc5959b0f6.pcm.png)
> To really understand language no easy! Image by [Jen Looper](https://twitter.com/jenlooper)

### How dem take make dis technology possible?

E possible because person write computer program to do am. Some years back, some science fiction writers talk say people go dey talk to their computers and di computers go always understand wetin dem mean. But e no easy as dem think, and even though di problem don dey better understood today, e still get big wahala to achieve 'perfect' natural language processing, especially to understand di meaning of sentence. E hard well well to understand humor or detect emotions like sarcasm for sentence.

You fit dey remember di grammar lessons for school where teacher dey talk about di parts of grammar for sentence. For some countries, dem dey teach grammar and linguistics as separate subject, but for many, dem dey include am as part of learning language: either your first language for primary school (to learn how to read and write) and maybe second language for secondary school. No worry if you no sabi di difference between nouns and verbs or adverbs and adjectives!

If you dey struggle with di difference between *simple present* and *present progressive*, you no dey alone. E dey hard for many people, even people wey dey speak di language well. Di good news be say computers sabi apply formal rules well, and you go learn how to write code wey fit *parse* sentence like human. Di bigger wahala wey you go look later na how to understand di *meaning* and *sentiment* of sentence.

## Prerequisites

For dis lesson, di main prerequisite na make you fit read and understand di language wey dem use for di lesson. No math problems or equations dey solve. Di original author write dis lesson for English, but e don translate to other languages, so you fit dey read translation. Some examples dey use different languages (to compare di grammar rules of di languages). Dem no translate di examples, but di explanation dey translated, so di meaning go clear.

For di coding tasks, you go use Python and di examples dey use Python 3.8.

For dis section, you go need and use:

- **Python 3 comprehension**. Programming language comprehension for Python 3, dis lesson dey use input, loops, file reading, arrays.
- **Visual Studio Code + extension**. We go use Visual Studio Code and di Python extension. You fit also use any Python IDE wey you like.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) na simple text processing library for Python. Follow di instructions for di TextBlob site to install am for your system (install di corpora too, as dem show below):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: You fit run Python direct for VS Code environments. Check di [docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) for more info.

## Talking to machines

Di history of how people dey try make computers understand human language don dey for many years, and one of di first scientists wey think about natural language processing na *Alan Turing*.

### Di 'Turing test'

When Turing dey research *artificial intelligence* for di 1950's, e think say dem fit give human and computer one conversation test (via typed messages) where di human no go sure if na another human or computer e dey talk to.

If after some time di human no fit know if di answers dey come from computer or human, dem fit say di computer dey *think*?

### Di inspiration - 'di imitation game'

Di idea come from one party game wey dem call *Di Imitation Game* where one interrogator dey alone for one room and e go try know which of di two people (wey dey another room) be male and female. Di interrogator fit send notes, and e go try ask questions wey di answers go show di gender of di mystery person. Di people for di other room go dey try confuse di interrogator by answering di questions in way wey go mislead di interrogator, but still dey look like dem dey answer honestly.

### Developing Eliza

For di 1960's, one MIT scientist wey dem call *Joseph Weizenbaum* develop [*Eliza*](https://wikipedia.org/wiki/ELIZA), one computer 'therapist' wey dey ask human questions and e go look like e understand di answers. But Eliza fit parse sentence and identify some grammar constructs and keywords to give reasonable answer, e no fit say e *understand* di sentence. If Eliza see sentence like "**I am** <u>sad</u>", e fit change di tense and add some words to form response like "How long have **you been** <u>sad</u>".

Dis go make e look like Eliza understand di statement and dey ask follow-up question, but na just rearrange and substitute words e dey do. If Eliza no fit find keyword wey e get response for, e go give random response wey fit work for many different statements. Eliza fit dey tricked easily, for example if person write "**You are** a <u>bicycle</u>", e fit respond "How long have **I been** a <u>bicycle</u>?", instead of better response.

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 Click di image above for video about di original ELIZA program

> Note: You fit read di original description of [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) wey dem publish for 1966 if you get ACM account. Or read about Eliza for [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercise - coding basic conversational bot

Conversational bot, like Eliza, na program wey dey collect user input and e go look like e understand and dey respond well. Unlike Eliza, our bot no go get many rules wey go make e look like e dey talk intelligently. Instead, our bot go just dey give random responses wey fit work for almost any small talk.

### Di plan

Steps to build conversational bot:

1. Print instructions wey go tell di user how to talk to di bot
2. Start loop
   1. Collect user input
   2. If user talk say e wan exit, make e exit
   3. Process user input and decide response (for dis case, di response na random choice from list of possible generic responses)
   4. Print response
3. Go back to step 2

### Build di bot

Make we create di bot now. We go start by defining some phrases.

1. Create dis bot for Python with di random responses wey dey below:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Dis na sample output to guide you (user input dey for di lines wey start with `>`):

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

    One possible solution to di task dey [here](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Stop and think

    1. You think say di random responses fit 'trick' person to believe say di bot really understand dem?
    2. Wetin di bot go need to make e dey more effective?
    3. If bot fit really 'understand' di meaning of sentence, e go need to 'remember' di meaning of di sentences wey dem don talk before for di conversation?

---

## 🚀Challenge

Choose one of di "stop and think" tins wey dey above and try implement am for code or write solution for paper using pseudocode.

For di next lesson, you go learn about other ways to parse natural language and machine learning.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Check di references below for more reading.

### References

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Assignment 

[Search for a bot](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI transleshion service [Co-op Translator](https://github.com/Azure/co-op-translator) do di transleshion. Even as we dey try make am correct, abeg make you sabi say automatik transleshion fit get mistake or no dey accurate. Di original dokyument wey dey for im native language na di one wey you go take as di correct source. For important informashon, e good make you use professional human transleshion. We no go fit take blame for any misunderstanding or wrong interpretashon wey fit happen because you use dis transleshion.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->