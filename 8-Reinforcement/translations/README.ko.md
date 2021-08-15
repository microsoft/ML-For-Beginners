# Reinforcement learning 소개하기

Reinforcement learning, 즉 RL은, supervised learning 과 unsupervised learning 다음의, 기초 머신러닝 페러다임 중 하나로 봅니다. RL은 모든 의사결정입니다: 올바른 결정을 하거나 최소한 배우게 됩니다.

주식 시장처럼 시뮬레이션된 환경을 상상해봅니다. 만약 규제시키면 어떤 일이 벌어질까요. 긍정적이거나 부정적인 영향을 주나요? 만약 부정적인 일이 생긴다면, 진로를 바꾸어, _negative reinforcement_ 을 배울 필요가 있습니다. 긍정적인 결과는, _positive reinforcement_ 로 만들 필요가 있습니다.

![peter and the wolf](../images/peter.png)

> Peter and his friends need to escape the hungry wolf! Image by [Jen Looper](https://twitter.com/jenlooper)

## 지역 토픽: 피터와 늑대 (러시아)

[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)는 러시아 작곡가 [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev)가 작성한 뮤지컬 동화입니다. 늑대를 쫓아내기 위해 용감하게 집 밖의 숲으로 떠나는, 젊은 개척가 피터의 이야기입니다. 이 섹션에서, Peter를 도와주기 위해서 머신러닝 알고리즘을 훈련해볼 예정입니다:

- 주변 영역을 **탐색**하고 최적의 길을 안내하는 지도 만들기
- 빨리 움직이기 위해서, 스케이트보드를 사용하고 밸런스잡는 방식을 **배우기**

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Peter and the Wolf by Prokofiev를 들으려면 이미지 클릭

## Reinforcement learning

이전 섹션에서, 머신러닝 문제의 예시를 보았습니다:

- **Supervised**, 해결하려는 문제에 대해서 예시 솔루션을 추천할 데이터셋이 있습니다. [Classification](../../4-Classification/translations/README.ko.md) 과 [regression](../../2-Regression/translations/README.ko.md)은 supervised learning 작업입니다.

- **Unsupervised**, 라벨링된 훈련 데이터가 없습니다. unsupervised learning의 주요 예시는 [Clustering](../../5-Clustering/translations/README.ko.md)입니다.

이 섹션에서, 라벨링된 훈련 데이터가 필요없는 학습 문제의 새로운 타입에 대하여 소개할 예정입니다. 여러 문제의 타입이 있습니다:

- **[Semi-supervised learning](https://wikipedia.org/wiki/Semi-supervised_learning)**, 모델을 사전-훈련하며 사용할 수 있던 라벨링하지 않은 데이터를 많이 가지고 있습니다.
- **[Reinforcement learning](https://wikipedia.org/wiki/Reinforcement_learning)**, 에이전트는 시뮬레이션된 환경에서 실험해서 행동하는 방식을 학습합니다.

### 예시 - 컴퓨터 게임

체스나, [Super Mario](https://wikipedia.org/wiki/Super_Mario) 같은, 게임 플레이를 컴퓨터에게 가르치고 싶다고 가정합니다. 컴퓨터가 게임을 플레이하려면, 게임 상태마다 어떻게 움직여야 되는지 예측할 필요가 있습니다. classification 문제처럼 보이겠지만, 아닙니다 - 상태와 일치하는 작업이 같이 있는 데이터셋은 없기 때문입니다. 기존 체스 경기 혹은 Super Mario를 즐기는 플레이어의 기록이 같은 소수의 데이터가 있을 수 있겠지만, 데이터가 가능한 상태의 충분히 많은 수를 커버할 수 없을 수 있습니다.

기존 게임 데이터를 찾는 대신에, **Reinforcement Learning** (RL)은  매번 *making the computer play* 하고 결과를 지켜보는 아이디어가 기반됩니다.  그래서, Reinforcement Learning을 적용하면, 2가지가 필요합니다:              

- 게임을 계속 플레이할 수 있는 **환경**과 **시뮬레이터**. 시뮬레이터는 모든 게임 규칙의 가능한 상태와 동작을 정의합니다.

- **Reward function**, 각자 움직이거나 게임이 진행되면서 얼마나 잘 했는지 알려줍니다.

다른 타입의 머신러닝과 RL 사이에 다른 주요 포인트는 RL에서 일반적으로 게임을 끝내기 전에 이기거나 지는 것을 알 수 없다는 점입니다. 그래서, 특정 동작이 좋을지 나쁠지 말할 수 없습니다 - 오직 게임의 끝에서 보상을 받습니다. 그리고 목표는 불확실 조건에서 모델을 훈련할 알고리즘을 만드는 것입니다. **Q-learning**이라고 불리는 RL 알고리즘에 대하여 배울 예정입니다.

## 강의

1. [Reinforcement learning과 Q-Learning 소개하기](../1-QLearning/translations/EADME.ko.md)
2. [헬스장 시뮬레이션 환경 사용하기](../2-Gym/translations/README.ko.md)

## 크레딧

"Introduction to Reinforcement Learning" was written with ♥️ by [Dmitry Soshnikov](http://soshnikov.com)
