<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0ffe994d1cc881bdeb49226a064116e5",
  "translation_date": "2025-09-04T00:18:28+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "ko"
}
-->
# 강화 학습과 Q-러닝 소개

![기계 학습에서 강화 학습 요약을 스케치노트로 표현](../../../../translated_images/ml-reinforcement.94024374d63348dbb3571c343ca7ddabef72adac0b8086d47164b769ba3a8a1d.ko.png)
> 스케치노트 제공: [Tomomi Imura](https://www.twitter.com/girlie_mac)

강화 학습은 세 가지 중요한 개념을 포함합니다: 에이전트, 상태, 그리고 각 상태에서의 행동 집합. 특정 상태에서 행동을 실행하면 에이전트는 보상을 받습니다. 컴퓨터 게임 슈퍼 마리오를 다시 상상해 보세요. 당신은 마리오이고, 게임 레벨에서 절벽 가장자리 옆에 서 있습니다. 당신 위에는 동전이 있습니다. 당신이 마리오로서 특정 위치에 있는 게임 레벨에 있다는 것이 바로 당신의 상태입니다. 오른쪽으로 한 걸음 이동하는 행동은 절벽 아래로 떨어지게 만들고 낮은 점수를 받게 됩니다. 하지만 점프 버튼을 누르면 점수를 얻고 살아남을 수 있습니다. 이는 긍정적인 결과이며, 높은 점수를 받아야 합니다.

강화 학습과 시뮬레이터(게임)를 사용하면 살아남고 최대한 많은 점수를 얻는 보상을 극대화하는 방법을 배울 수 있습니다.

[![강화 학습 소개](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 위 이미지를 클릭하여 Dmitry가 강화 학습에 대해 설명하는 영상을 확인하세요.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/45/)

## 사전 준비 및 설정

이 강의에서는 Python으로 코드를 실험해 볼 것입니다. 이 강의의 Jupyter Notebook 코드를 컴퓨터나 클라우드에서 실행할 수 있어야 합니다.

[강의 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)을 열어 강의를 따라가며 실습을 진행하세요.

> **참고:** 클라우드에서 코드를 열 경우, 노트북 코드에서 사용되는 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 파일도 가져와야 합니다. 해당 파일을 노트북과 동일한 디렉토리에 추가하세요.

## 소개

이 강의에서는 러시아 작곡가 [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev)의 음악 동화 **[피터와 늑대](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**에서 영감을 받아 피터가 환경을 탐험하고 맛있는 사과를 모으며 늑대를 피하도록 하는 **강화 학습**을 탐구할 것입니다.

**강화 학습**(RL)은 여러 실험을 실행하여 **에이전트**가 특정 **환경**에서 최적의 행동을 학습할 수 있도록 하는 학습 기법입니다. 이 환경에서 에이전트는 **보상 함수**로 정의된 **목표**를 가져야 합니다.

## 환경

간단히 하기 위해, 피터의 세계를 `width` x `height` 크기의 정사각형 보드로 간주해 봅시다:

![피터의 환경](../../../../translated_images/environment.40ba3cb66256c93fa7e92f6f7214e1d1f588aafa97d266c11d108c5c5d101b6c.ko.png)

이 보드의 각 셀은 다음 중 하나일 수 있습니다:

* **땅**: 피터와 다른 생물이 걸을 수 있는 곳.
* **물**: 걸을 수 없는 곳.
* **나무** 또는 **풀**: 쉴 수 있는 장소.
* **사과**: 피터가 먹고 싶어하는 것.
* **늑대**: 위험하며 피해야 하는 존재.

이 환경을 다루는 코드는 별도의 Python 모듈 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)에 포함되어 있습니다. 이 코드는 개념을 이해하는 데 중요하지 않으므로 모듈을 가져와 샘플 보드를 생성하는 데 사용합니다(코드 블록 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

이 코드는 위 그림과 유사한 환경을 출력해야 합니다.

## 행동과 정책

이 예제에서 피터의 목표는 늑대와 장애물을 피하면서 사과를 찾는 것입니다. 이를 위해 그는 사과를 찾을 때까지 걸어다닐 수 있습니다.

따라서 어떤 위치에서든 그는 다음 행동 중 하나를 선택할 수 있습니다: 위, 아래, 왼쪽, 오른쪽.

이 행동을 사전으로 정의하고, 해당 좌표 변화에 매핑합니다. 예를 들어, 오른쪽으로 이동(`R`)은 `(1,0)`에 해당합니다(코드 블록 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

이 시나리오의 전략과 목표를 요약하면 다음과 같습니다:

- **전략**: 에이전트(피터)의 전략은 **정책**이라고 불리는 함수로 정의됩니다. 정책은 주어진 상태에서 행동을 반환합니다. 이 경우, 문제의 상태는 플레이어의 현재 위치를 포함한 보드로 표현됩니다.

- **목표**: 강화 학습의 목표는 문제를 효율적으로 해결할 수 있는 좋은 정책을 학습하는 것입니다. 하지만 기본적으로 가장 간단한 정책인 **랜덤 워크**를 고려해 봅시다.

## 랜덤 워크

먼저 랜덤 워크 전략을 구현하여 문제를 해결해 봅시다. 랜덤 워크에서는 허용된 행동 중에서 다음 행동을 무작위로 선택하여 사과에 도달할 때까지 이동합니다(코드 블록 3).

1. 아래 코드를 사용하여 랜덤 워크를 구현하세요:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    `walk` 호출은 해당 경로의 길이를 반환해야 하며, 실행마다 결과가 달라질 수 있습니다.

1. 워크 실험을 여러 번(예: 100회) 실행하고 결과 통계를 출력하세요(코드 블록 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    평균 경로 길이가 약 30-40 단계로 나타나며, 이는 평균적으로 가장 가까운 사과까지의 거리가 약 5-6 단계인 점을 고려할 때 꽤 많은 단계입니다.

    랜덤 워크 동안 피터의 움직임을 확인할 수도 있습니다:

    ![피터의 랜덤 워크](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 보상 함수

정책을 더 지능적으로 만들기 위해 어떤 움직임이 다른 움직임보다 "더 나은지"를 이해해야 합니다. 이를 위해 목표를 정의해야 합니다.

목표는 **보상 함수**로 정의될 수 있으며, 각 상태에 대해 점수를 반환합니다. 숫자가 높을수록 보상이 더 좋습니다(코드 블록 5).

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

보상 함수의 흥미로운 점은 대부분의 경우 *게임이 끝날 때* 실질적인 보상을 받는다는 것입니다. 이는 알고리즘이 긍정적인 보상으로 이어지는 "좋은" 단계를 기억하고 그 중요성을 높여야 하며, 나쁜 결과로 이어지는 모든 움직임은 억제해야 한다는 것을 의미합니다.

## Q-러닝

여기서 논의할 알고리즘은 **Q-러닝**이라고 합니다. 이 알고리즘에서 정책은 **Q-테이블**이라는 함수(또는 데이터 구조)로 정의됩니다. 이는 주어진 상태에서 각 행동의 "좋음"을 기록합니다.

Q-테이블은 종종 테이블 또는 다차원 배열로 표현하기 편리하기 때문에 이렇게 불립니다. 보드의 크기가 `width` x `height`인 경우, Q-테이블은 `width` x `height` x `len(actions)` 형태의 numpy 배열로 표현할 수 있습니다(코드 블록 6):

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Q-테이블의 모든 값을 동일한 값으로 초기화합니다. 여기서는 0.25로 초기화합니다. 이는 모든 상태에서 모든 움직임이 동일하게 좋은 "랜덤 워크" 정책에 해당합니다. Q-테이블을 `plot` 함수에 전달하여 보드에서 테이블을 시각화할 수 있습니다: `m.plot(Q)`.

![피터의 환경](../../../../translated_images/env_init.04e8f26d2d60089e128f21d22e5fef57d580e559f0d5937b06c689e5e7cdd438.ko.png)

각 셀의 중앙에는 이동 방향을 나타내는 "화살표"가 있습니다. 모든 방향이 동일하기 때문에 점이 표시됩니다.

이제 시뮬레이션을 실행하고 환경을 탐색하며 Q-테이블 값을 더 나은 분포로 학습해야 합니다. 이를 통해 사과를 찾는 경로를 훨씬 더 빠르게 찾을 수 있습니다.

## Q-러닝의 핵심: 벨만 방정식

움직이기 시작하면 각 행동은 해당 보상을 가지게 됩니다. 즉, 이론적으로 가장 높은 즉각적인 보상을 기반으로 다음 행동을 선택할 수 있습니다. 하지만 대부분의 상태에서는 움직임이 사과에 도달하는 목표를 달성하지 못하므로 어떤 방향이 더 나은지 즉시 결정할 수 없습니다.

> 즉각적인 결과가 중요한 것이 아니라, 시뮬레이션 끝에서 얻을 최종 결과가 중요하다는 점을 기억하세요.

이 지연된 보상을 고려하기 위해 **[동적 프로그래밍](https://en.wikipedia.org/wiki/Dynamic_programming)**의 원칙을 사용해야 합니다. 이는 문제를 재귀적으로 생각할 수 있도록 합니다.

현재 상태 *s*에 있다고 가정하고 다음 상태 *s'*로 이동하려고 합니다. 이렇게 하면 보상 함수로 정의된 즉각적인 보상 *r(s,a)*를 받게 되며, 추가적인 미래 보상도 받게 됩니다. Q-테이블이 각 행동의 "매력"을 올바르게 반영한다고 가정하면, 상태 *s'*에서 *Q(s',a')* 값이 최대인 행동 *a'*를 선택할 것입니다. 따라서 상태 *s*에서 얻을 수 있는 최상의 미래 보상은 `max`

## 정책 확인하기

Q-Table은 각 상태에서 각 행동의 "매력도"를 나열하고 있으므로, 이를 사용하여 우리 세계에서 효율적인 탐색을 정의하는 것은 매우 간단합니다. 가장 간단한 경우, Q-Table 값이 가장 높은 행동을 선택하면 됩니다: (코드 블록 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> 위 코드를 여러 번 실행해보면, 가끔 코드가 "멈추는" 현상을 경험할 수 있습니다. 이 경우 노트북에서 STOP 버튼을 눌러 중단해야 합니다. 이는 두 상태가 최적의 Q-Value 관점에서 서로를 "가리키는" 상황이 발생할 수 있기 때문입니다. 이 경우 에이전트는 두 상태 사이를 무한히 이동하게 됩니다.

## 🚀도전 과제

> **과제 1:** `walk` 함수를 수정하여 경로의 최대 길이를 특정 단계 수(예: 100)로 제한하고, 위 코드가 이 값을 반환하는 것을 확인하세요.

> **과제 2:** `walk` 함수가 이전에 방문했던 장소로 다시 돌아가지 않도록 수정하세요. 이렇게 하면 `walk`가 루프에 빠지는 것을 방지할 수 있지만, 에이전트가 탈출할 수 없는 위치에 "갇히는" 상황은 여전히 발생할 수 있습니다.

## 탐색

더 나은 탐색 정책은 학습 중에 사용했던 정책으로, 탐색과 활용을 결합한 것입니다. 이 정책에서는 Q-Table 값에 비례하여 각 행동을 특정 확률로 선택합니다. 이 전략은 여전히 에이전트가 이미 탐색한 위치로 돌아가는 결과를 초래할 수 있지만, 아래 코드에서 볼 수 있듯이 목표 위치까지의 평균 경로가 매우 짧아지는 결과를 가져옵니다(참고로 `print_statistics`는 시뮬레이션을 100번 실행합니다): (코드 블록 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

이 코드를 실행한 후, 이전보다 훨씬 짧은 평균 경로 길이를 얻을 수 있습니다. 이는 보통 3~6 범위에 속합니다.

## 학습 과정 조사

앞서 언급했듯이, 학습 과정은 문제 공간 구조에 대한 지식을 탐색하고 활용하는 균형입니다. 학습 결과(에이전트가 목표로 가는 짧은 경로를 찾는 능력)가 개선된 것을 확인했지만, 학습 과정 중 평균 경로 길이가 어떻게 변화하는지 관찰하는 것도 흥미롭습니다:

학습 내용을 요약하면 다음과 같습니다:

- **평균 경로 길이 증가**. 처음에는 평균 경로 길이가 증가합니다. 이는 환경에 대해 아무것도 모를 때, 나쁜 상태(물이나 늑대)에 갇히기 쉽기 때문입니다. 더 많은 것을 배우고 이 지식을 활용하기 시작하면, 환경을 더 오래 탐색할 수 있지만, 여전히 사과가 어디에 있는지 잘 모르는 상태입니다.

- **학습이 진행됨에 따라 경로 길이 감소**. 충분히 학습한 후에는 에이전트가 목표를 달성하기 쉬워지고, 경로 길이가 줄어들기 시작합니다. 하지만 여전히 탐색을 열어두고 있기 때문에, 종종 최적 경로에서 벗어나 새로운 옵션을 탐색하며 경로가 최적보다 길어지기도 합니다.

- **경로 길이의 급격한 증가**. 그래프에서 관찰할 수 있는 또 다른 점은, 어느 순간 경로 길이가 급격히 증가한다는 것입니다. 이는 과정의 확률적 특성을 나타내며, Q-Table 계수를 새로운 값으로 덮어쓰면서 "망칠" 수 있음을 의미합니다. 이는 학습 후반부에 학습률을 줄여 Q-Table 값을 소폭으로만 조정하도록 함으로써 최소화할 수 있습니다.

전반적으로, 학습 과정의 성공과 품질은 학습률, 학습률 감소, 할인 계수와 같은 매개변수에 크게 의존합니다. 이러한 매개변수는 **하이퍼파라미터**라고 불리며, 학습 중에 최적화되는 **파라미터**(예: Q-Table 계수)와 구분됩니다. 최적의 하이퍼파라미터 값을 찾는 과정을 **하이퍼파라미터 최적화**라고 하며, 이는 별도의 주제로 다룰 가치가 있습니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## 과제 
[더 현실적인 세계](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  