<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-06T10:54:33+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "en"
}
-->
# History of Machine Learning

![Summary of History of Machine Learning in a sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - History of Machine Learning](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML for beginners - History of Machine Learning")

> 🎥 Click the image above for a short video covering this lesson.

In this lesson, we’ll explore the key milestones in the history of machine learning and artificial intelligence.

The history of artificial intelligence (AI) as a field is closely tied to the history of machine learning, as the algorithms and computational advancements that drive ML have contributed to the development of AI. It’s worth noting that while these fields began to take shape as distinct areas of study in the 1950s, significant [algorithmic, statistical, mathematical, computational, and technical discoveries](https://wikipedia.org/wiki/Timeline_of_machine_learning) predate and overlap this period. In fact, people have been pondering these ideas for [centuries](https://wikipedia.org/wiki/History_of_artificial_intelligence). This article delves into the intellectual foundations of the concept of a "thinking machine."

---
## Notable Discoveries

- 1763, 1812 [Bayes Theorem](https://wikipedia.org/wiki/Bayes%27_theorem) and its predecessors. This theorem and its applications form the basis of inference, describing the probability of an event occurring based on prior knowledge.
- 1805 [Least Square Theory](https://wikipedia.org/wiki/Least_squares) by French mathematician Adrien-Marie Legendre. This theory, which you’ll learn about in our Regression unit, aids in data fitting.
- 1913 [Markov Chains](https://wikipedia.org/wiki/Markov_chain), named after Russian mathematician Andrey Markov, are used to describe sequences of possible events based on a previous state.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron), a type of linear classifier invented by American psychologist Frank Rosenblatt, laid the groundwork for advances in deep learning.

---

- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor), originally designed for mapping routes, is used in ML to detect patterns.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) is employed to train [feedforward neural networks](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network), derived from feedforward neural networks, create temporal graphs.

✅ Do some research. What other dates stand out as pivotal in the history of ML and AI?

---
## 1950: Machines That Think

Alan Turing, an extraordinary individual who was voted [by the public in 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) as the greatest scientist of the 20th century, is credited with laying the foundation for the concept of a "machine that can think." He addressed skeptics and his own need for empirical evidence by creating the [Turing Test](https://www.bbc.com/news/technology-18475646), which you’ll explore in our NLP lessons.

---
## 1956: Dartmouth Summer Research Project

"The Dartmouth Summer Research Project on artificial intelligence was a landmark event for AI as a field," and it was here that the term "artificial intelligence" was coined ([source](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it.

---

The lead researcher, mathematics professor John McCarthy, aimed "to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." Participants included another prominent figure in the field, Marvin Minsky.

The workshop is credited with sparking discussions on topics such as "the rise of symbolic methods, systems focused on limited domains (early expert systems), and deductive systems versus inductive systems." ([source](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "The Golden Years"

From the 1950s to the mid-1970s, there was great optimism about AI’s potential to solve numerous problems. In 1967, Marvin Minsky confidently stated, "Within a generation ... the problem of creating 'artificial intelligence' will substantially be solved." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Research in natural language processing flourished, search algorithms were refined and made more powerful, and the concept of "micro-worlds" emerged, where simple tasks were completed using plain language instructions.

---

Government agencies provided generous funding, computational and algorithmic advancements were made, and prototypes of intelligent machines were developed. Some of these machines include:

* [Shakey the robot](https://wikipedia.org/wiki/Shakey_the_robot), which could navigate and decide how to perform tasks "intelligently."

    ![Shakey, an intelligent robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey in 1972

---

* Eliza, an early "chatterbot," could converse with people and act as a primitive "therapist." You’ll learn more about Eliza in the NLP lessons.

    ![Eliza, a bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > A version of Eliza, a chatbot

---

* "Blocks world" was an example of a micro-world where blocks could be stacked and sorted, allowing experiments in teaching machines to make decisions. Advances using libraries like [SHRDLU](https://wikipedia.org/wiki/SHRDLU) propelled language processing forward.

    [![blocks world with SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world with SHRDLU")

    > 🎥 Click the image above for a video: Blocks world with SHRDLU

---
## 1974 - 1980: "AI Winter"

By the mid-1970s, it became clear that the complexity of creating "intelligent machines" had been underestimated and that its promise, given the available computational power, had been overstated. Funding dried up, and confidence in the field waned. Some factors that contributed to this decline included:
---
- **Limitations**. Computational power was insufficient.
- **Combinatorial explosion**. The number of parameters required for training grew exponentially as more was demanded of computers, without a corresponding evolution in computational power and capability.
- **Lack of data**. A shortage of data hindered the testing, development, and refinement of algorithms.
- **Are we asking the right questions?**. Researchers began to question the very questions they were pursuing:
  - Turing tests faced criticism, including the "Chinese room theory," which argued that "programming a digital computer may make it appear to understand language but could not produce real understanding." ([source](https://plato.stanford.edu/entries/chinese-room/))
  - Ethical concerns arose about introducing artificial intelligences like the "therapist" ELIZA into society.

---

During this time, different schools of thought in AI emerged. A dichotomy developed between ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) approaches. _Scruffy_ labs tweaked programs extensively to achieve desired results, while _neat_ labs focused on logic and formal problem-solving. ELIZA and SHRDLU were well-known _scruffy_ systems. In the 1980s, as the demand for reproducible ML systems grew, the _neat_ approach gained prominence due to its more explainable results.

---
## 1980s: Expert Systems

As the field matured, its value to businesses became evident, leading to the proliferation of "expert systems" in the 1980s. "Expert systems were among the first truly successful forms of artificial intelligence (AI) software." ([source](https://wikipedia.org/wiki/Expert_system))

These systems were _hybrid_, combining a rules engine that defined business requirements with an inference engine that used the rules to deduce new facts.

This era also saw growing interest in neural networks.

---
## 1987 - 1993: AI 'Chill'

The rise of specialized expert systems hardware had the unintended consequence of becoming overly specialized. Meanwhile, the advent of personal computers competed with these large, centralized systems. The democratization of computing had begun, eventually paving the way for the modern explosion of big data.

---
## 1993 - 2011

This period marked a new chapter for ML and AI, enabling solutions to earlier challenges caused by limited data and computational power. Data availability grew rapidly, especially with the introduction of smartphones around 2007. Computational power expanded exponentially, and algorithms evolved in tandem. The field began to mature, transitioning from its freewheeling early days into a structured discipline.

---
## Now

Today, machine learning and AI influence nearly every aspect of our lives. This era demands a thoughtful understanding of the risks and potential impacts of these algorithms on human lives. As Microsoft's Brad Smith has noted, "Information technology raises issues that go to the heart of fundamental human-rights protections like privacy and freedom of expression. These issues heighten responsibility for tech companies that create these products. In our view, they also call for thoughtful government regulation and for the development of norms around acceptable uses" ([source](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

The future remains uncertain, but understanding these systems, their software, and algorithms is crucial. We hope this curriculum will help you gain the knowledge needed to form your own perspective.

[![The history of deep learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "The history of deep learning")
> 🎥 Click the image above for a video: Yann LeCun discusses the history of deep learning in this lecture

---
## 🚀Challenge

Dive deeper into one of these historical moments and learn more about the people behind them. The characters are fascinating, and no scientific discovery happens in isolation. What do you uncover?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---
## Review & Self Study

Here are some resources to watch and listen to:

[This podcast where Amy Boyd discusses the evolution of AI](http://runasradio.com/Shows/Show/739)

[![The history of AI by Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "The history of AI by Amy Boyd")

---

## Assignment

[Create a timeline](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.