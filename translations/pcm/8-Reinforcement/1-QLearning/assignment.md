<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-11-18T18:17:24+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "pcm"
}
-->
# A More Realistic World

For dis our situation, Peter fit waka around almost without dey tire or hungry. But for more realistic world, e go need siddon rest from time to time, and also chop food. Make we make our world more realistic, by wey we go add dis rules:

1. As Peter dey waka from one place go another, e go dey lose **energy** and e go dey gain **fatigue**.
2. Peter fit get more energy if e chop apple.
3. Peter fit remove fatigue if e rest under tree or for grass (like if e waka enter board location wey get tree or grass - green field).
4. Peter need find and kill wolf.
5. To fit kill wolf, Peter go need get certain level of energy and fatigue, if not e go lose the fight.

## Instructions

Use the original [notebook.ipynb](notebook.ipynb) notebook as starting point for your solution.

Change the reward function wey dey above to match the rules of the game, run the reinforcement learning algorithm to learn the best strategy wey go help win the game, and compare the results of random walk with your algorithm based on how many games e win and lose.

> **Note**: For dis new world, the state go dey more complex, and apart from human position, e go also include fatigue and energy levels. You fit decide to represent the state as tuple (Board, energy, fatigue), or you fit define class for the state (you fit also wan derive am from `Board`), or even change the original `Board` class inside [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

For your solution, abeg keep the code wey dey responsible for random walk strategy, and compare the results of your algorithm with random walk for the end.

> **Note**: You fit need adjust hyperparameters to make am work, especially the number of epochs. Because the success of the game (fighting the wolf) na rare event, you fit expect say e go take longer time to train.

## Rubric

| Criteria | Exemplary                                                                                                                                                                                             | Adequate                                                                                                                                                                                | Needs Improvement                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook dey show new world rules, Q-Learning algorithm and some explanation. Q-Learning fit improve results well well compared to random walk.                                                       | Notebook dey show Q-Learning wey e implement and e improve results compared to random walk, but e no too dey significant; or notebook no dey well documented and code no dey well arranged | Some try dey to re-define the rules of the world, but Q-Learning algorithm no work, or reward function no dey complete                                                          |

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI transle-shun service [Co-op Translator](https://github.com/Azure/co-op-translator) take do di transle-shun. Even though we dey try make am correct, abeg no forget say AI transle-shun fit get mistake or no dey 100% accurate. Di original dokyument for di language wey dem take write am first na di main correct one. If na somtin wey serious or important, e go beta make una use professional human transle-shun. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because of dis transle-shun.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->