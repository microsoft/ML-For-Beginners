<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-06T11:00:15+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "en"
}
-->
# Train Mountain Car

[OpenAI Gym](http://gym.openai.com) is designed so that all environments share the same API—i.e., the same methods `reset`, `step`, and `render`, as well as the same abstractions for **action space** and **observation space**. This makes it possible to adapt the same reinforcement learning algorithms to different environments with minimal code changes.

## A Mountain Car Environment

The [Mountain Car environment](https://gym.openai.com/envs/MountainCar-v0/) involves a car stuck in a valley:

The goal is to get out of the valley and reach the flag by performing one of the following actions at each step:

| Value | Meaning |
|---|---|
| 0 | Accelerate to the left |
| 1 | Do not accelerate |
| 2 | Accelerate to the right |

The main challenge of this problem is that the car's engine is not powerful enough to climb the mountain in a single attempt. Therefore, the only way to succeed is to drive back and forth to build up momentum.

The observation space consists of just two values:

| Num | Observation  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Car Position | -1.2| 0.6 |
|  1  | Car Velocity | -0.07 | 0.07 |

The reward system for the mountain car is somewhat tricky:

 * A reward of 0 is given if the agent reaches the flag (position = 0.5) at the top of the mountain.
 * A reward of -1 is given if the agent's position is less than 0.5.

The episode ends if the car's position exceeds 0.5 or if the episode length exceeds 200 steps.

## Instructions

Adapt our reinforcement learning algorithm to solve the mountain car problem. Start with the existing [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) code, substitute the new environment, modify the state discretization functions, and try to train the existing algorithm with minimal code changes. Optimize the results by adjusting hyperparameters.

> **Note**: Adjusting hyperparameters will likely be necessary to make the algorithm converge.

## Rubric

| Criteria | Exemplary | Adequate | Needs Improvement |
| -------- | --------- | -------- | ----------------- |
|          | The Q-Learning algorithm is successfully adapted from the CartPole example with minimal code modifications and is able to solve the problem of capturing the flag in under 200 steps. | A new Q-Learning algorithm is adopted from the Internet but is well-documented; or the existing algorithm is adapted but does not achieve the desired results. | The student was unable to successfully adopt any algorithm but made substantial progress toward a solution (e.g., implemented state discretization, Q-Table data structure, etc.). |

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.