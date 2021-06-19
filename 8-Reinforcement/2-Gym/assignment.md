# Train Mountain Car

[OpenAI Gym](http://gym.openai.com) has been designed in such a way that all environments provide the same API - i.e. the same methods `reset`, `step` and `render`, and the same abstractions of **action space** and **observation space**. Thus is should be possible to adapt the same reinforcement learning algorithms to different environments with minimal code changes.

## A Mountain Car Environment

[Mountain Car environment](https://gym.openai.com/envs/MountainCar-v0/) contains a car stuck in a valley:

<img src="images/mountaincar.png" width="300"/>

The goal is to get out of the valley and capture the flag, by doing at each step one of the following actions:

| Value | Meaning |
|---|---|
| 0 | Accelerate to the left |
| 1 | Do not accelerate |
| 2 | Accelerate to the right |

The main trick of this problem is, however, that the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

Observation space consists of just two values:

| Num | Observation  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Car Position | -1.2| 0.6 |
|  1  | Car Velocity | -0.07 | 0.07 |

Reward system for the mountain car is rather tricky:

 * Reward of 0 is awarded if the agent reached the flag (position = 0.5) on top of the mountain.
 * Reward of -1 is awarded if the position of the agent is less than 0.5.

Episode terminates if the car position is more than 0.5, or episode length is greater than 200.
## Instructions

Adapt our reinforcement learning algorithm to solve the mountain car problem. Start with existing [notebook.ipynb](notebook.ipynb) code, substitute new environment, change state discretization functions, and try to make existing algorithm to train with minimal code modifications. Optimize the result by adjusting hyperparameters.

> **Note**: Hyperparameters adjustment is likely to be needed to make algorithm converge. 
## Rubric

| Criteria | Exemplary | Adequate | Needs Improvement |
| -------- | --------- | -------- | ----------------- |
|          | Q-Learning algorithm is successfully adapted from CartPole example, with minimal code modifications, which is able to solve the problem of capturing the flag under 200 steps. | A new Q-Learning algorithm has been adopted from the Internet, but is well-documented; or existing algorithm adopted, but does not reach desired results | Student was not able to successfully adopt any algorithm, but has mede substantial steps towards solution (implemented state discretization, Q-Table data structure, etc.) |
