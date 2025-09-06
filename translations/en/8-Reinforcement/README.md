<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-06T10:58:42+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "en"
}
-->
# Introduction to reinforcement learning

Reinforcement learning, or RL, is considered one of the fundamental paradigms of machine learning, alongside supervised learning and unsupervised learning. RL focuses on decision-making: making the right decisions or, at the very least, learning from them.

Imagine you have a simulated environment, like the stock market. What happens if you implement a specific regulation? Does it lead to positive or negative outcomes? If something negative occurs, you need to take this _negative reinforcement_, learn from it, and adjust your approach. If the outcome is positive, you should build on that _positive reinforcement_.

![peter and the wolf](../../../8-Reinforcement/images/peter.png)

> Peter and his friends need to escape the hungry wolf! Image by [Jen Looper](https://twitter.com/jenlooper)

## Regional topic: Peter and the Wolf (Russia)

[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) is a musical fairy tale written by the Russian composer [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). It tells the story of a young pioneer, Peter, who bravely ventures out of his house into a forest clearing to confront a wolf. In this section, we will train machine learning algorithms to help Peter:

- **Explore** the surrounding area and create an optimal navigation map.
- **Learn** how to use a skateboard and maintain balance on it to move around more quickly.

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Click the image above to listen to Peter and the Wolf by Prokofiev

## Reinforcement learning

In earlier sections, you encountered two types of machine learning problems:

- **Supervised learning**, where we have datasets that provide example solutions to the problem we aim to solve. [Classification](../4-Classification/README.md) and [regression](../2-Regression/README.md) are examples of supervised learning tasks.
- **Unsupervised learning**, where we lack labeled training data. A primary example of unsupervised learning is [Clustering](../5-Clustering/README.md).

In this section, we will introduce a new type of learning problem that does not rely on labeled training data. There are several types of such problems:

- **[Semi-supervised learning](https://wikipedia.org/wiki/Semi-supervised_learning)**, where we have a large amount of unlabeled data that can be used to pre-train the model.
- **[Reinforcement learning](https://wikipedia.org/wiki/Reinforcement_learning)**, where an agent learns how to behave by conducting experiments in a simulated environment.

### Example - computer game

Imagine you want to teach a computer to play a game, such as chess or [Super Mario](https://wikipedia.org/wiki/Super_Mario). For the computer to play the game, it needs to predict which move to make in each game state. While this might seem like a classification problem, it is not—because we do not have a dataset containing states and corresponding actions. Although we might have some data, like records of chess matches or gameplay footage of Super Mario, it is unlikely that this data will sufficiently cover the vast number of possible states.

Instead of relying on existing game data, **Reinforcement Learning** (RL) is based on the idea of *letting the computer play* the game repeatedly and observing the outcomes. To apply Reinforcement Learning, we need two key components:

- **An environment** and **a simulator** that allow the computer to play the game multiple times. This simulator defines all the game rules, as well as possible states and actions.

- **A reward function**, which evaluates how well the computer performed during each move or game.

The primary difference between RL and other types of machine learning is that in RL, we typically do not know whether we have won or lost until the game is over. Therefore, we cannot determine whether a specific move is good or bad on its own—we only receive feedback (a reward) at the end of the game. Our goal is to design algorithms that enable us to train a model under these uncertain conditions. In this section, we will explore one RL algorithm called **Q-learning**.

## Lessons

1. [Introduction to reinforcement learning and Q-Learning](1-QLearning/README.md)
2. [Using a gym simulation environment](2-Gym/README.md)

## Credits

"Introduction to Reinforcement Learning" was written with ♥️ by [Dmitry Soshnikov](http://soshnikov.com)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.