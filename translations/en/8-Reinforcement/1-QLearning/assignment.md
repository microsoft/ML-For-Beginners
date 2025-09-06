<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-06T10:59:31+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "en"
}
-->
# A More Realistic World

In our scenario, Peter could move around almost endlessly without feeling tired or hungry. In a more realistic world, he would need to sit down and rest occasionally, as well as eat to sustain himself. Let's make our world more realistic by implementing the following rules:

1. Moving from one location to another causes Peter to lose **energy** and gain **fatigue**.
2. Peter can regain energy by eating apples.
3. Peter can reduce fatigue by resting under a tree or on the grass (i.e., stepping into a board location with a tree or grass - green field).
4. Peter needs to locate and defeat the wolf.
5. To defeat the wolf, Peter must have specific levels of energy and fatigue; otherwise, he will lose the battle.

## Instructions

Use the original [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) notebook as the starting point for your solution.

Modify the reward function described above according to the game's rules, run the reinforcement learning algorithm to determine the best strategy for winning the game, and compare the results of random walk with your algorithm in terms of the number of games won and lost.

> **Note**: In this new world, the state is more complex and includes not only Peter's position but also his fatigue and energy levels. You can choose to represent the state as a tuple (Board, energy, fatigue), define a class for the state (you may also want to derive it from `Board`), or even modify the original `Board` class inside [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

In your solution, ensure that the code responsible for the random walk strategy is retained, and compare the results of your algorithm with the random walk strategy at the end.

> **Note**: You may need to adjust hyperparameters to make the algorithm work, especially the number of epochs. Since the game's success (defeating the wolf) is a rare event, you should expect a much longer training time.

## Rubric

| Criteria | Exemplary                                                                                                                                                                                             | Adequate                                                                                                                                                                                | Needs Improvement                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | A notebook is presented with the definition of new world rules, Q-Learning algorithm, and some textual explanations. Q-Learning significantly improves results compared to random walk.                | A notebook is presented, Q-Learning is implemented and improves results compared to random walk, but not significantly; or the notebook is poorly documented and the code is not well-structured. | Some attempt to redefine the world's rules is made, but the Q-Learning algorithm does not work, or the reward function is not fully defined. |

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may include errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is advised. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.