# Getting Started with Reinforcement Learning

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Click the image above to listen to Peter and the Wolf by Prokofiev
## Regional Topic: Peter and the Wolf (Russia)

[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) is a musical fairy tale written by a Russian composer [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). It is a story about young pioneer Peter, who bravely goes out of his house to the forest clearing to chase the wolf. In this section, we will train machine learning algorithms that will help Peter:

- **Explore** the surrounding area and build an optimal navigation map
- **Learn** how to use a skateboard and balance on it, in order to move around faster.

## Introduction to Reinforcement Learning

In previous sections, you have seen two example of machine learning problems:

* **Supervised**, where we had some datasets that show sample solutions to the problem we want to solve. [Classification](../4-Classification/README.md) and [Regression](../2-Regression/README.md) are supervised learning tasks.
* **Unsupervised**, in which we do not have training data. The main example of unsupervised learning is [Clustering](../5-Clustering/README.md).

In this section, we will introduce you to a new type of learning problems, which do not require labeled training data. There are a several types of such problems:

* **[Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning)**, where we have a lot of unlabeled data that can be used to pre-train the model.
* **[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)**, in which the agent learns how to behave by performing a lot of experiments in some simulated environment.

Suppose, you want to teach computer to play a game, such as chess, or [Super Mario](https://en.wikipedia.org/wiki/Super_Mario). For computer to play a game, we need it to predict which move to make in each of the game states. While this may seem like a classification problem, it is not - because we do not have a dataset with states and corresponding actions. While we may have some data like that (existing chess matches, or recording of players playing Super Mario), it is likely not to cover sufficiently large number of possible states.

Instead of looking for existing game data, **Reinforcement Learning** (RL) is based on the idea of *making the computer play* many times, observing the result. Thus, to apply Reinforcement Learning, we need two things:
1. **An environment** and **a simulator**, which would allow us to play a game many times. This simulator would define all game rules, possible states and actions.
2. **A reward function**, which would tell us how well we did during each move or game.

The main difference between supervised learning is that in RL we typically do not know whether we win or lose until we finish the game. Thus, we cannot say whether a certain move alone is good or now - we only receive reward at the end of the game. And our goal is to design such algorithms that will allow us to train a model under such uncertain conditions. We will learn about one RL algorithm called **Q-learning**.

## Lessons

1. [Introduction to Reinforcement Learning and Q-Learning](1-QLearning/README.md)
2. [Using gym simulation environment](2-Gym/README.md)

## Credits

"Introduction to Reinforcement Learning" was written with ♥️ by [Dmitry Soshnikov](http://soshnikov.com)
