<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T10:54:20+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "ko"
}
-->
## 사전 준비

이번 강의에서는 **OpenAI Gym**이라는 라이브러리를 사용하여 다양한 **환경**을 시뮬레이션할 것입니다. 이 강의의 코드는 로컬 환경(예: Visual Studio Code)에서 실행할 수 있으며, 이 경우 시뮬레이션이 새 창에서 열립니다. 온라인으로 코드를 실행할 경우, [여기](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7)에 설명된 대로 코드를 약간 수정해야 할 수도 있습니다.

## OpenAI Gym

이전 강의에서는 게임의 규칙과 상태가 우리가 직접 정의한 `Board` 클래스에 의해 제공되었습니다. 이번에는 **시뮬레이션 환경**을 사용하여 균형 잡힌 막대의 물리학을 시뮬레이션할 것입니다. 강화 학습 알고리즘을 훈련하기 위한 가장 인기 있는 시뮬레이션 환경 중 하나는 [Gym](https://gym.openai.com/)이며, 이는 [OpenAI](https://openai.com/)에서 관리합니다. 이 Gym을 사용하면 카트폴 시뮬레이션부터 아타리 게임까지 다양한 **환경**을 생성할 수 있습니다.

> **참고**: OpenAI Gym에서 제공하는 다른 환경은 [여기](https://gym.openai.com/envs/#classic_control)에서 확인할 수 있습니다.

먼저 Gym을 설치하고 필요한 라이브러리를 가져옵니다. (코드 블록 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## 연습 - 카트폴 환경 초기화하기

카트폴 균형 문제를 다루기 위해서는 해당 환경을 초기화해야 합니다. 각 환경은 다음과 관련됩니다:

- **관찰 공간**: 환경에서 얻는 정보의 구조를 정의합니다. 카트폴 문제에서는 막대의 위치, 속도 및 기타 값을 받습니다.

- **행동 공간**: 가능한 행동을 정의합니다. 우리의 경우 행동 공간은 이산적이며, **왼쪽**과 **오른쪽** 두 가지 행동으로 구성됩니다. (코드 블록 2)

1. 초기화를 위해 다음 코드를 입력하세요:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

환경이 어떻게 작동하는지 확인하려면 100단계 동안 짧은 시뮬레이션을 실행해 봅시다. 각 단계에서 `action_space`에서 무작위로 선택한 행동을 제공합니다.

1. 아래 코드를 실행하고 결과를 확인하세요.

    ✅ 이 코드는 로컬 Python 설치에서 실행하는 것이 좋습니다! (코드 블록 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    다음과 유사한 이미지를 볼 수 있어야 합니다:

    ![균형을 잡지 못하는 카트폴](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. 시뮬레이션 중에는 행동을 결정하기 위해 관찰값을 얻어야 합니다. 실제로 `step` 함수는 현재 관찰값, 보상 함수, 시뮬레이션을 계속할 가치가 있는지 여부를 나타내는 완료 플래그를 반환합니다: (코드 블록 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    노트북 출력에서 다음과 같은 결과를 볼 수 있습니다:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    시뮬레이션의 각 단계에서 반환되는 관찰 벡터는 다음 값을 포함합니다:
    - 카트의 위치
    - 카트의 속도
    - 막대의 각도
    - 막대의 회전 속도

1. 이러한 숫자의 최소값과 최대값을 확인하세요: (코드 블록 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    또한 각 시뮬레이션 단계에서 보상 값이 항상 1임을 알 수 있습니다. 이는 우리의 목표가 가능한 한 오랫동안 막대를 수직에 가깝게 유지하는 것이기 때문입니다.

    ✅ 실제로, 카트폴 시뮬레이션은 100번의 연속적인 시도에서 평균 보상이 195에 도달하면 해결된 것으로 간주됩니다.

## 상태 이산화

Q-Learning에서는 각 상태에서 무엇을 해야 할지를 정의하는 Q-테이블을 구축해야 합니다. 이를 위해 상태는 **이산적**이어야 하며, 더 정확히는 유한한 수의 이산 값으로 구성되어야 합니다. 따라서 관찰값을 **이산화**하여 유한한 상태 집합으로 매핑해야 합니다.

이를 수행하는 몇 가지 방법이 있습니다:

- **구간으로 나누기**: 특정 값의 범위를 알고 있다면 이 범위를 여러 **구간**으로 나누고, 해당 값이 속하는 구간 번호로 값을 대체할 수 있습니다. 이는 numpy의 [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) 메서드를 사용하여 수행할 수 있습니다. 이 경우 디지털화에 선택한 구간 수에 따라 상태 크기를 정확히 알 수 있습니다.
  
✅ 값을 특정 유한 범위(예: -20에서 20)로 선형 보간한 다음, 반올림하여 정수로 변환할 수 있습니다. 이는 입력 값의 정확한 범위를 모를 경우 상태 크기에 대한 제어가 약간 줄어듭니다. 예를 들어, 우리의 경우 4개의 값 중 2개는 상한/하한이 정의되어 있지 않아 상태 수가 무한해질 수 있습니다.

우리의 예제에서는 두 번째 접근법을 사용할 것입니다. 나중에 알 수 있듯이 상한/하한이 정의되지 않았음에도 불구하고, 이러한 값들은 특정 유한 범위를 벗어나는 경우가 드뭅니다. 따라서 극단적인 값의 상태는 매우 드물게 나타납니다.

1. 모델에서 관찰값을 받아 4개의 정수 값 튜플을 생성하는 함수는 다음과 같습니다: (코드 블록 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. 구간을 사용한 또 다른 이산화 방법을 탐색해 봅시다: (코드 블록 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. 짧은 시뮬레이션을 실행하고 이산화된 환경 값을 관찰해 봅시다. `discretize`와 `discretize_bins`를 모두 시도해 보고 차이가 있는지 확인하세요.

    ✅ `discretize_bins`는 0부터 시작하는 구간 번호를 반환합니다. 따라서 입력 변수 값이 0 근처일 경우 구간의 중간 값(10)을 반환합니다. `discretize`에서는 출력 값의 범위에 대해 신경 쓰지 않았기 때문에 값이 음수가 될 수 있으며, 0은 0에 해당합니다. (코드 블록 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ 환경 실행을 보고 싶다면 `env.render`로 시작하는 줄의 주석을 제거하세요. 그렇지 않으면 백그라운드에서 실행할 수 있으며, 이 방식이 더 빠릅니다. Q-Learning 과정에서는 이 "보이지 않는" 실행을 사용할 것입니다.

## Q-테이블 구조

이전 강의에서는 상태가 0에서 8까지의 숫자 쌍으로 간단했기 때문에 Q-테이블을 8x8x2 형태의 numpy 텐서로 표현하는 것이 편리했습니다. 구간 이산화를 사용하는 경우 상태 벡터의 크기도 알려져 있으므로 동일한 접근법을 사용하여 상태를 20x20x10x10x2 배열 형태로 표현할 수 있습니다(여기서 2는 행동 공간의 차원이며, 첫 번째 차원은 관찰 공간의 각 매개변수에 대해 선택한 구간 수에 해당합니다).

하지만 관찰 공간의 정확한 차원이 알려지지 않은 경우도 있습니다. `discretize` 함수의 경우, 일부 원래 값이 제한되지 않았기 때문에 상태가 특정 한계 내에 유지된다는 것을 확신할 수 없습니다. 따라서 약간 다른 접근법을 사용하여 Q-테이블을 딕셔너리로 표현할 것입니다.

1. *(state, action)* 쌍을 딕셔너리 키로 사용하고, 값은 Q-테이블 항목 값에 해당합니다. (코드 블록 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    여기서 `qvalues()`라는 함수를 정의하여 특정 상태에 대해 가능한 모든 행동에 해당하는 Q-테이블 값을 반환합니다. Q-테이블에 항목이 없으면 기본값으로 0을 반환합니다.

## Q-Learning 시작하기

이제 피터에게 균형 잡는 법을 가르칠 준비가 되었습니다!

1. 먼저 몇 가지 하이퍼파라미터를 설정합시다: (코드 블록 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    여기서 `alpha`는 **학습률**로, 각 단계에서 Q-테이블의 현재 값을 어느 정도 조정해야 할지를 정의합니다. 이전 강의에서는 1로 시작하여 훈련 중에 `alpha`를 낮은 값으로 줄였습니다. 이번 예제에서는 단순성을 위해 이를 일정하게 유지할 것이며, 나중에 `alpha` 값을 조정하는 실험을 할 수 있습니다.

    `gamma`는 **할인 계수**로, 현재 보상보다 미래 보상을 얼마나 우선시해야 하는지를 나타냅니다.

    `epsilon`은 **탐험/활용 계수**로, 탐험을 선호할지 활용을 선호할지를 결정합니다. 알고리즘에서는 `epsilon` 비율만큼 Q-테이블 값을 기준으로 다음 행동을 선택하고, 나머지 경우에는 무작위 행동을 실행합니다. 이를 통해 이전에 탐색하지 않은 검색 공간 영역을 탐험할 수 있습니다.

    ✅ 균형을 잡는 관점에서 보면, 무작위 행동(탐험)은 잘못된 방향으로의 무작위 펀치처럼 작용하며, 막대는 이러한 "실수"에서 균형을 회복하는 방법을 배워야 합니다.

### 알고리즘 개선

이전 강의의 알고리즘을 두 가지 방식으로 개선할 수 있습니다:

- **평균 누적 보상 계산**: 여러 시뮬레이션에 걸쳐 평균 누적 보상을 계산합니다. 5000번의 반복마다 진행 상황을 출력하며, 해당 기간 동안 누적 보상의 평균을 계산합니다. 만약 195점 이상을 얻는다면 문제를 해결했다고 간주할 수 있으며, 요구된 품질보다 더 높은 품질로 해결한 것입니다.
  
- **최대 평균 누적 결과 계산**: `Qmax`를 계산하고, 해당 결과에 해당하는 Q-테이블을 저장합니다. 훈련을 실행하면 평균 누적 결과가 때때로 감소하는 것을 알 수 있으며, 이는 상황을 악화시키는 값으로 이미 학습된 Q-테이블 값을 "파괴"할 수 있음을 의미합니다.

1. 각 시뮬레이션에서 누적 보상을 `rewards` 벡터에 수집하여 나중에 그래프를 그립니다. (코드 블록 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

이 결과에서 다음을 알 수 있습니다:

- **목표에 근접**: 100번 이상의 연속 실행에서 195 누적 보상을 얻는 목표에 매우 근접하거나 실제로 달성했을 수 있습니다! 더 작은 숫자를 얻더라도, 우리는 5000번의 실행에서 평균을 내기 때문에 공식 기준에서는 100번의 실행만 필요합니다.
  
- **보상이 감소하기 시작**: 때때로 보상이 감소하기 시작하는데, 이는 상황을 악화시키는 값으로 이미 학습된 Q-테이블 값을 "파괴"할 수 있음을 의미합니다.

이 관찰은 훈련 진행 상황을 그래프로 나타내면 더 명확하게 보입니다.

## 훈련 진행 상황 그래프 그리기

훈련 중에 각 반복에서 누적 보상 값을 `rewards` 벡터에 수집했습니다. 이를 반복 번호에 대해 그래프로 나타내면 다음과 같습니다:

```python
plt.plot(rewards)
```

![원시 진행 상황](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

이 그래프에서는 아무것도 알 수 없습니다. 이는 확률적 훈련 과정의 특성상 훈련 세션 길이가 크게 변하기 때문입니다. 이 그래프를 더 이해하기 쉽게 만들기 위해, 예를 들어 100번의 실험에 대해 **이동 평균**을 계산할 수 있습니다. 이는 `np.convolve`를 사용하여 편리하게 수행할 수 있습니다: (코드 블록 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![훈련 진행 상황](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## 하이퍼파라미터 조정

학습을 더 안정적으로 만들기 위해 훈련 중에 일부 하이퍼파라미터를 조정하는 것이 좋습니다. 특히:

- **학습률** `alpha`의 경우, 1에 가까운 값으로 시작한 다음 점차적으로 이 값을 줄이는 것이 좋습니다. 시간이 지나면 Q-테이블에서 좋은 확률 값을 얻을 수 있으며, 이를 약간만 조정하고 새 값으로 완전히 덮어쓰지 않아야 합니다.

- **epsilon 증가**: `epsilon`을 천천히 증가시켜 탐험을 줄이고 활용을 늘리는 것이 좋습니다. 낮은 `epsilon` 값으로 시작하여 거의 1에 가까운 값으로 이동하는 것이 합리적일 수 있습니다.
> **작업 1**: 하이퍼파라미터 값을 조정해 보며 더 높은 누적 보상을 얻을 수 있는지 확인하세요. 195점을 넘기고 있나요?
> **과제 2**: 문제를 공식적으로 해결하려면 100번의 연속 실행에서 평균 보상이 195에 도달해야 합니다. 훈련 중에 이를 측정하고 문제를 공식적으로 해결했는지 확인하세요!

## 결과 확인하기

훈련된 모델이 실제로 어떻게 작동하는지 보는 것은 흥미로울 것입니다. 시뮬레이션을 실행하고 훈련 중과 동일한 행동 선택 전략을 따라 Q-테이블의 확률 분포에 따라 샘플링해 봅시다: (코드 블록 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

다음과 같은 결과를 볼 수 있을 것입니다:

![균형 잡힌 카트폴](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀도전

> **과제 3**: 여기서는 최종 Q-테이블을 사용했지만, 이것이 최상의 결과를 제공하는 것은 아닐 수 있습니다. 우리가 가장 성능이 좋은 Q-테이블을 `Qbest` 변수에 저장했다는 것을 기억하세요! `Qbest`를 `Q`로 복사하여 최상의 Q-테이블을 사용해 동일한 예제를 실행하고 차이를 확인해 보세요.

> **과제 4**: 여기서는 각 단계에서 최상의 행동을 선택하지 않고, 해당 확률 분포에 따라 샘플링했습니다. 항상 Q-테이블 값이 가장 높은 최상의 행동을 선택하는 것이 더 합리적일까요? 이를 위해 `np.argmax` 함수를 사용하여 Q-테이블 값이 가장 높은 행동 번호를 찾을 수 있습니다. 이 전략을 구현하여 균형 잡기에 개선이 있는지 확인해 보세요.

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 과제
[Mountain Car 훈련하기](assignment.md)

## 결론

우리는 이제 에이전트가 게임의 원하는 상태를 정의하는 보상 함수만 제공받고, 검색 공간을 지능적으로 탐색할 기회를 통해 좋은 결과를 얻는 방법을 배웠습니다. 우리는 Q-러닝 알고리즘을 이산적 환경과 연속적 환경에서 성공적으로 적용했으며, 이산적 행동을 사용했습니다.

행동 상태도 연속적이고 관찰 공간이 훨씬 더 복잡한 상황, 예를 들어 아타리 게임 화면의 이미지를 다루는 경우를 연구하는 것도 중요합니다. 이러한 문제에서는 종종 신경망과 같은 더 강력한 머신러닝 기술을 사용해야 좋은 결과를 얻을 수 있습니다. 이러한 더 고급 주제는 앞으로 진행될 고급 AI 과정에서 다룰 예정입니다.

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  