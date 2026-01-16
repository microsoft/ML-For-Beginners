<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-11-18T18:12:51+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "pcm"
}
-->
# Introduction to reinforcement learning

Reinforcement learning, RL, na one of di main machine learning style, e dey follow supervised learning and unsupervised learning. RL na all about decision: how to make correct decision or at least learn from di one wey you don make.

Imagine say you get one simulated environment like stock market. Wetin go happen if you put one regulation? E go get positive or negative effect? If e get negative effect, you go need take di _negative reinforcement_, learn from am, and change wetin you dey do. If e get positive result, you go need build on top di _positive reinforcement_.

![peter and the wolf](../../../translated_images/pcm/peter.779730f9ba3a8a8d.webp)

> Peter and im friends wan run comot from di hungry wolf! Image by [Jen Looper](https://twitter.com/jenlooper)

## Regional topic: Peter and the Wolf (Russia)

[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) na one musical fairy tale wey Russian composer [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) write. Di story na about young pioneer Peter, wey get courage go forest clearing to chase wolf. For dis section, we go train machine learning algorithms wey go help Peter:

- **Explore** di area wey dey around and create one better navigation map
- **Learn** how to use skateboard and balance on top am, so e go fit waka fast.

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Click di image above to hear Peter and the Wolf by Prokofiev

## Reinforcement learning

For di previous sections, you don see two examples of machine learning problems:

- **Supervised**, wey we get datasets wey dey show sample solutions to di problem wey we wan solve. [Classification](../4-Classification/README.md) and [regression](../2-Regression/README.md) na supervised learning tasks.
- **Unsupervised**, wey we no get labeled training data. Di main example of unsupervised learning na [Clustering](../5-Clustering/README.md).

For dis section, we go show you one new type of learning problem wey no need labeled training data. Dis type get different kinds:

- **[Semi-supervised learning](https://wikipedia.org/wiki/Semi-supervised_learning)**, wey we get plenty unlabeled data wey fit help pre-train di model.
- **[Reinforcement learning](https://wikipedia.org/wiki/Reinforcement_learning)**, wey agent go learn how to behave by doing experiments for one simulated environment.

### Example - computer game

Imagine say you wan teach computer how to play game, like chess, or [Super Mario](https://wikipedia.org/wiki/Super_Mario). For di computer to play di game, e go need predict di move wey e go make for each game state. Even though e fit look like classification problem, e no be - because we no get dataset wey get states and di actions wey follow. Even if we get data like chess matches or recording of people wey dey play Super Mario, e no go cover enough possible states.

Instead of looking for existing game data, **Reinforcement Learning** (RL) dey base on di idea of *making di computer play* plenty times and check di result. So, to use Reinforcement Learning, we need two things:

- **Environment** and **simulator** wey go allow us play di game plenty times. Dis simulator go define all di game rules plus di possible states and actions.

- **Reward function**, wey go tell us how well we do for each move or game.

Di main difference between other types of machine learning and RL be say for RL, we no dey know whether we go win or lose until di game finish. So, we no fit talk whether one move alone good or bad - we go only get reward when di game finish. Our goal na to design algorithms wey go help us train model for uncertain conditions. We go learn about one RL algorithm wey dem dey call **Q-learning**.

## Lessons

1. [Introduction to reinforcement learning and Q-Learning](1-QLearning/README.md)
2. [Using a gym simulation environment](2-Gym/README.md)

## Credits

"Introduction to Reinforcement Learning" na work wey [Dmitry Soshnikov](http://soshnikov.com) write with ♥️

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis document don dey translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am accurate, abeg sabi say automated translations fit get mistake or no dey 100% correct. Di original document for im native language na di main correct source. For important information, e better make una use professional human translation. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because of dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->