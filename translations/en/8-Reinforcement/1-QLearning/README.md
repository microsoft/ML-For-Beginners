<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-06T10:59:08+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "en"
}
-->
## Visualizing the Learned Policy

After running the learning algorithm, we can visualize the Q-Table to see the learned policy. The arrows (or circles) in each cell will indicate the preferred direction of movement based on the Q-Table values. This visualization helps us understand how the agent has learned to navigate the environment.

For example, the updated Q-Table might look like this:

![Peter's Learned Policy](../../../../8-Reinforcement/1-QLearning/images/learned_policy.png)

In this visualization:
- The arrows point in the direction of the action with the highest Q-Table value for each state.
- The agent is more likely to follow these directions to reach the apple while avoiding the wolf and other obstacles.

## Testing the Learned Policy

Once the Q-Table is trained, we can test the learned policy by letting the agent navigate the environment using the Q-Table values. Instead of randomly choosing actions, the agent will now select the action with the highest Q-Table value at each state.

Run the following code to test the learned policy: (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

This code will simulate the agent's movement based on the learned policy. You can observe how efficiently the agent reaches the apple compared to the random walk strategy.

## Results and Observations

After training the agent using Q-Learning:
- The agent should be able to reach the apple in significantly fewer steps compared to the random walk strategy.
- The learned policy will guide the agent to avoid the wolf and other obstacles while maximizing the reward.

You can also visualize the agent's movement during the test run:

![Peter's Learned Movement](../../../../8-Reinforcement/1-QLearning/images/learned_movement.gif)

Notice how the agent's movements are more purposeful and efficient compared to the random walk.

## Summary

In this lesson, we explored the basics of reinforcement learning and implemented the Q-Learning algorithm to train an agent to navigate an environment. Here's what we covered:
- The concepts of states, actions, rewards, and policies in reinforcement learning.
- How to define a reward function to guide the agent's learning process.
- The Bellman equation and its role in updating the Q-Table.
- The balance between exploration and exploitation during training.
- How to implement and visualize the Q-Learning algorithm in Python.

By the end of this lesson, you should have a solid understanding of how reinforcement learning works and how Q-Learning can be used to solve problems in a simulated environment.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Assignment

1. Modify the reward function to include penalties for stepping into water or grass. How does this affect the agent's learning process and the resulting policy?
2. Experiment with different values of the discount factor (γ) and learning rate (α). How do these parameters influence the agent's behavior and the speed of learning?
3. Create a new environment with a different layout (e.g., more obstacles, multiple apples, or multiple wolves). Train the agent in this new environment and observe how it adapts.

By completing these assignments, you'll gain a deeper understanding of how to fine-tune reinforcement learning algorithms and apply them to various scenarios.

The learnings can be summarized as:

- **Average path length increases**. Initially, the average path length increases. This is likely because, when we know nothing about the environment, the agent is prone to getting stuck in unfavorable states, such as water or encountering a wolf. As the agent gathers more knowledge and begins to use it, it can explore the environment for longer periods, but it still doesn't have a clear understanding of where the apples are located.

- **Path length decreases as we learn more**. Once the agent has learned enough, it becomes easier to achieve the goal, and the path length starts to decrease. However, since the agent is still exploring, it occasionally deviates from the optimal path to investigate new possibilities, which can make the path longer than necessary.

- **Abrupt length increase**. Another observation from the graph is that, at some point, the path length increases abruptly. This highlights the stochastic nature of the process, where the Q-Table coefficients can be "spoiled" by being overwritten with new values. Ideally, this should be minimized by reducing the learning rate (e.g., toward the end of training, adjusting Q-Table values by only small amounts).

Overall, it’s important to note that the success and quality of the learning process depend heavily on parameters such as the learning rate, learning rate decay, and discount factor. These are often referred to as **hyperparameters**, to distinguish them from **parameters**, which are optimized during training (e.g., Q-Table coefficients). The process of finding the best hyperparameter values is called **hyperparameter optimization**, which is a topic worthy of its own discussion.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Assignment  
[A More Realistic World](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.