## Checking the policy

Since the Q-Table lists the "attractiveness" of each action at each state, it is quite easy to use it to define the efficient navigation in our world. In the simplest case, we can select the action corresponding to the highest Q-Table value: (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> If you try the code above several times, you may notice that sometimes it "hangs", and you need to press the STOP button in the notebook to interrupt it. This happens because there could be situations when two states "point" to each other in terms of optimal Q-Value, in which case the agents ends up moving between those states indefinitely.

## 🚀Challenge

> **Task 1:** Modify the `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where it has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape.

## Navigation

A better navigation policy would be the one that we used during training, which combines exploitation and exploration. In this policy, we will select each action with a certain probability, proportional to the values in the Q-Table. This strategy may still result in the agent returning back to a position it has already explored, but, as you can see from the code below, it results in a very short average path to the desired location (remember that `print_statistics` runs the simulation 100 times): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

After running this code, you should get a much smaller average path length than before, in the range of 3-6.

## Investigating the learning process

As we have mentioned, the learning process is a balance between exploration and exploitation of gained knowledge about the structure of problem space. We have seen that the results of learning (the ability to help an agent to find a short path to the goal) has improved, but it is also interesting to observe how the average path length behaves during the learning process:

The learnings can be summarized as:

- **Average path length increases**. What we see here is that at first, the average path length increases. This is probably due to the fact that when we know nothing about the environment, we are likely to get trapped in bad states, such as water or the wolf. As we learn more and start using this knowledge, we can explore the environment for longer, but we still do not know where the apples are very well.

- **Path length decreases, as we learn more**. Once we learn enough, it becomes easier for the agent to achieve the goal, and the path length starts to decrease. However, we are still open to exploration, so we often diverge away from the best path and explore new options, making the path longer than optimal.

- **Length increases abruptly**. What we also observe on this graph is that at some point, the length increased abruptly. This indicates the stochastic nature of the process, and that we can at some point "spoil" the Q-Table coefficients by overwriting them with new values. This ideally should be minimized by decreasing the learning rate (for example, towards the end of training, we only adjust Q-Table values by a small value).

Overall, it is important to remember that the success and quality of the learning process significantly depends on parameters such as learning rate, learning rate decay, and discount factor. Those are often called **hyperparameters**, to distinguish them from **parameters**, which we optimize during training (for example, Q-Table coefficients). The process of finding the best hyperparameter values is called **hyperparameter optimization**, and it deserves a separate topic.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Assignment 
[A More Realistic World](assignment.md)

I'm sorry, but I cannot translate text into "mo" as it is not a recognized language or dialect in my training data. If you meant a specific language or dialect, please clarify, and I'll be happy to assist you!