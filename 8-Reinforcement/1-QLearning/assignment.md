# A More Realistic World

In our situation, Peter was able to move around almost without getting tired or hungry. In a more realistic world, we has to sit down and rest from time to time, and also to feed himself. Let's make our world more realistic, by implementing the following rules:

1. By moving from one place to another, Peter loses **energy** and gains some **fatigue**.
2. Peter can gain more energy by eating apples.
3. Peter can get rid of fatigue by resting under the tree or on the grass (i.e. walking into a board location with a tree or grass - green field)
4. Peter needs to find and kill the wolf
5. In order to kill the wolf, Peter needs to have certain levels of energy and fatigue, otherwise he loses the battle.
## Instructions

Use the original [notebook.ipynb](notebook.ipynb) notebook as a starting point for your solution.

Modify the reward function above according to the rules of the game, run the reinforcement learning algorithm to learn the best strategy for winning the game, and compare the results of random walk with your algorithm in terms of number of games won and lost.

> **Note**: In your new world, the state is more complex, and in addition to human position also includes fatigue and energy levels. You may chose to represent the state as a tuple (Board,energy,fatigue), or define a class for the state (you may also want to derive it from `Board`), or even modify the original `Board` class inside [rlboard.py](rlboard.py).

In your solution, please keep the code responsible for random walk strategy, and compare the results of your algorithm with random walk at the end.

> **Note**: You may need to adjust hyperparameters to make it work, especially the number of epochs. Because the success of the game (fighting the wolf) is a rare event, you can expect much longer training time.
## Rubric

| Criteria | Exemplary                                                                                                                                                                                             | Adequate                                                                                                                                                                                | Needs Improvement                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | A notebook is presented with the definition of new world rules, Q-Learning algorithm and some textual explanations. Q-Learning is able to significantly improve the results comparing to random walk. | Notebook is presented, Q-Learning is implemented and improves results comparing to random walk, but not significantly; or notebook is poorly documented and code is not well-structured | Some attempt to re-define the rules of the world are made, but Q-Learning algorithm does not work, or reward function is not fully defined |
