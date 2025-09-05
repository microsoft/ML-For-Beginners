<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T20:20:38+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "vi"
}
-->
# Trượt ván CartPole

Bài toán mà chúng ta đã giải trong bài học trước có vẻ như là một vấn đề đơn giản, không thực sự áp dụng được vào các tình huống thực tế. Nhưng thực tế không phải vậy, vì nhiều vấn đề trong thế giới thực cũng có kịch bản tương tự - bao gồm chơi cờ vua hoặc cờ vây. Chúng tương tự nhau vì chúng ta cũng có một bàn cờ với các quy tắc nhất định và một **trạng thái rời rạc**.

## [Câu hỏi trước bài học](https://ff-quizzes.netlify.app/en/ml/)

## Giới thiệu

Trong bài học này, chúng ta sẽ áp dụng các nguyên tắc của Q-Learning vào một bài toán với **trạng thái liên tục**, tức là trạng thái được biểu diễn bằng một hoặc nhiều số thực. Chúng ta sẽ giải quyết bài toán sau:

> **Bài toán**: Nếu Peter muốn thoát khỏi con sói, cậu ấy cần phải di chuyển nhanh hơn. Chúng ta sẽ xem cách Peter học cách trượt ván, đặc biệt là giữ thăng bằng, bằng cách sử dụng Q-Learning.

![Cuộc chạy trốn vĩ đại!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Peter và các bạn của cậu ấy sáng tạo để thoát khỏi con sói! Hình ảnh bởi [Jen Looper](https://twitter.com/jenlooper)

Chúng ta sẽ sử dụng một phiên bản đơn giản của việc giữ thăng bằng được gọi là bài toán **CartPole**. Trong thế giới CartPole, chúng ta có một thanh trượt ngang có thể di chuyển sang trái hoặc phải, và mục tiêu là giữ thăng bằng một cây cột thẳng đứng trên thanh trượt.

## Yêu cầu trước

Trong bài học này, chúng ta sẽ sử dụng một thư viện gọi là **OpenAI Gym** để mô phỏng các **môi trường** khác nhau. Bạn có thể chạy mã của bài học này trên máy tính cá nhân (ví dụ: từ Visual Studio Code), trong trường hợp đó, mô phỏng sẽ mở trong một cửa sổ mới. Khi chạy mã trực tuyến, bạn có thể cần thực hiện một số điều chỉnh, như được mô tả [ở đây](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Trong bài học trước, các quy tắc của trò chơi và trạng thái được cung cấp bởi lớp `Board` mà chúng ta tự định nghĩa. Ở đây, chúng ta sẽ sử dụng một **môi trường mô phỏng** đặc biệt, mô phỏng vật lý đằng sau cây cột giữ thăng bằng. Một trong những môi trường mô phỏng phổ biến nhất để huấn luyện các thuật toán học tăng cường được gọi là [Gym](https://gym.openai.com/), được duy trì bởi [OpenAI](https://openai.com/). Bằng cách sử dụng Gym này, chúng ta có thể tạo ra các **môi trường** khác nhau từ mô phỏng CartPole đến các trò chơi Atari.

> **Lưu ý**: Bạn có thể xem các môi trường khác có sẵn từ OpenAI Gym [tại đây](https://gym.openai.com/envs/#classic_control).

Đầu tiên, hãy cài đặt Gym và nhập các thư viện cần thiết (mã khối 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Bài tập - khởi tạo môi trường CartPole

Để làm việc với bài toán giữ thăng bằng CartPole, chúng ta cần khởi tạo môi trường tương ứng. Mỗi môi trường được liên kết với:

- **Không gian quan sát** định nghĩa cấu trúc thông tin mà chúng ta nhận được từ môi trường. Đối với bài toán CartPole, chúng ta nhận được vị trí của cây cột, vận tốc và một số giá trị khác.

- **Không gian hành động** định nghĩa các hành động có thể thực hiện. Trong trường hợp của chúng ta, không gian hành động là rời rạc và bao gồm hai hành động - **trái** và **phải**. (mã khối 2)

1. Để khởi tạo, hãy nhập mã sau:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Để xem cách môi trường hoạt động, hãy chạy một mô phỏng ngắn trong 100 bước. Tại mỗi bước, chúng ta cung cấp một hành động để thực hiện - trong mô phỏng này, chúng ta chỉ chọn ngẫu nhiên một hành động từ `action_space`.

1. Chạy mã dưới đây và xem kết quả.

    ✅ Nhớ rằng nên chạy mã này trên cài đặt Python cục bộ! (mã khối 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Bạn sẽ thấy một hình ảnh tương tự như hình này:

    ![CartPole không giữ thăng bằng](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Trong quá trình mô phỏng, chúng ta cần nhận các quan sát để quyết định cách hành động. Thực tế, hàm bước trả về các quan sát hiện tại, một hàm thưởng, và cờ `done` cho biết liệu có nên tiếp tục mô phỏng hay không: (mã khối 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Bạn sẽ thấy kết quả tương tự như thế này trong đầu ra của notebook:

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

    Vector quan sát được trả về tại mỗi bước của mô phỏng chứa các giá trị sau:
    - Vị trí của xe đẩy
    - Vận tốc của xe đẩy
    - Góc của cây cột
    - Tốc độ quay của cây cột

1. Lấy giá trị nhỏ nhất và lớn nhất của các số này: (mã khối 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Bạn cũng có thể nhận thấy rằng giá trị thưởng tại mỗi bước mô phỏng luôn là 1. Điều này là vì mục tiêu của chúng ta là tồn tại lâu nhất có thể, tức là giữ cây cột ở vị trí thẳng đứng hợp lý trong thời gian dài nhất.

    ✅ Thực tế, mô phỏng CartPole được coi là đã giải quyết nếu chúng ta đạt được phần thưởng trung bình là 195 trong 100 lần thử liên tiếp.

## Rời rạc hóa trạng thái

Trong Q-Learning, chúng ta cần xây dựng Q-Table để xác định hành động tại mỗi trạng thái. Để làm được điều này, trạng thái cần phải **rời rạc**, cụ thể hơn, nó phải chứa một số lượng hữu hạn các giá trị rời rạc. Vì vậy, chúng ta cần **rời rạc hóa** các quan sát, ánh xạ chúng thành một tập hợp hữu hạn các trạng thái.

Có một vài cách để làm điều này:

- **Chia thành các khoảng**. Nếu chúng ta biết khoảng của một giá trị nhất định, chúng ta có thể chia khoảng này thành một số **khoảng nhỏ**, và sau đó thay thế giá trị bằng số thứ tự của khoảng mà nó thuộc về. Điều này có thể được thực hiện bằng phương pháp [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) của numpy. Trong trường hợp này, chúng ta sẽ biết chính xác kích thước trạng thái, vì nó sẽ phụ thuộc vào số lượng khoảng mà chúng ta chọn để số hóa.

✅ Chúng ta có thể sử dụng nội suy tuyến tính để đưa các giá trị về một khoảng hữu hạn (ví dụ, từ -20 đến 20), và sau đó chuyển đổi các số thành số nguyên bằng cách làm tròn. Điều này cho chúng ta ít kiểm soát hơn về kích thước của trạng thái, đặc biệt nếu chúng ta không biết chính xác phạm vi của các giá trị đầu vào. Ví dụ, trong trường hợp của chúng ta, 2 trong số 4 giá trị không có giới hạn trên/dưới, điều này có thể dẫn đến số lượng trạng thái vô hạn.

Trong ví dụ của chúng ta, chúng ta sẽ sử dụng cách tiếp cận thứ hai. Như bạn có thể nhận thấy sau này, mặc dù không có giới hạn trên/dưới, những giá trị này hiếm khi vượt ra ngoài một khoảng hữu hạn nhất định, do đó những trạng thái với giá trị cực đoan sẽ rất hiếm.

1. Đây là hàm sẽ lấy quan sát từ mô hình của chúng ta và tạo ra một bộ giá trị nguyên gồm 4 giá trị: (mã khối 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Hãy khám phá một phương pháp rời rạc hóa khác sử dụng các khoảng: (mã khối 7)

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

1. Bây giờ hãy chạy một mô phỏng ngắn và quan sát các giá trị môi trường rời rạc. Hãy thử cả `discretize` và `discretize_bins` để xem có sự khác biệt nào không.

    ✅ `discretize_bins` trả về số thứ tự của khoảng, bắt đầu từ 0. Vì vậy, đối với các giá trị của biến đầu vào xung quanh 0, nó trả về số từ giữa khoảng (10). Trong `discretize`, chúng ta không quan tâm đến phạm vi của các giá trị đầu ra, cho phép chúng là số âm, do đó các giá trị trạng thái không bị dịch chuyển, và 0 tương ứng với 0. (mã khối 8)

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

    ✅ Bỏ chú thích dòng bắt đầu bằng `env.render` nếu bạn muốn xem cách môi trường thực thi. Nếu không, bạn có thể thực thi nó trong nền, điều này nhanh hơn. Chúng ta sẽ sử dụng cách thực thi "ẩn" này trong quá trình Q-Learning.

## Cấu trúc Q-Table

Trong bài học trước, trạng thái là một cặp số đơn giản từ 0 đến 8, và do đó rất tiện lợi để biểu diễn Q-Table bằng một tensor numpy với kích thước 8x8x2. Nếu chúng ta sử dụng rời rạc hóa bằng khoảng, kích thước của vector trạng thái cũng được biết, vì vậy chúng ta có thể sử dụng cách tiếp cận tương tự và biểu diễn trạng thái bằng một mảng có kích thước 20x20x10x10x2 (ở đây 2 là kích thước của không gian hành động, và các kích thước đầu tiên tương ứng với số lượng khoảng mà chúng ta đã chọn để sử dụng cho mỗi tham số trong không gian quan sát).

Tuy nhiên, đôi khi kích thước chính xác của không gian quan sát không được biết. Trong trường hợp của hàm `discretize`, chúng ta có thể không bao giờ chắc chắn rằng trạng thái của chúng ta nằm trong một giới hạn nhất định, vì một số giá trị ban đầu không bị giới hạn. Do đó, chúng ta sẽ sử dụng một cách tiếp cận hơi khác và biểu diễn Q-Table bằng một từ điển.

1. Sử dụng cặp *(state,action)* làm khóa của từ điển, và giá trị sẽ tương ứng với giá trị của mục trong Q-Table. (mã khối 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Ở đây chúng ta cũng định nghĩa một hàm `qvalues()`, trả về danh sách các giá trị Q-Table cho một trạng thái nhất định tương ứng với tất cả các hành động có thể. Nếu mục không có trong Q-Table, chúng ta sẽ trả về 0 làm giá trị mặc định.

## Bắt đầu Q-Learning

Bây giờ chúng ta đã sẵn sàng để dạy Peter cách giữ thăng bằng!

1. Đầu tiên, hãy đặt một số siêu tham số: (mã khối 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Ở đây, `alpha` là **tốc độ học** xác định mức độ chúng ta nên điều chỉnh các giá trị hiện tại của Q-Table tại mỗi bước. Trong bài học trước, chúng ta bắt đầu với giá trị 1, và sau đó giảm `alpha` xuống các giá trị thấp hơn trong quá trình huấn luyện. Trong ví dụ này, chúng ta sẽ giữ nó cố định chỉ để đơn giản, và bạn có thể thử nghiệm với việc điều chỉnh giá trị `alpha` sau.

    `gamma` là **hệ số chiết khấu** cho biết mức độ chúng ta nên ưu tiên phần thưởng trong tương lai so với phần thưởng hiện tại.

    `epsilon` là **yếu tố khám phá/khai thác** xác định liệu chúng ta nên ưu tiên khám phá hay khai thác. Trong thuật toán của chúng ta, chúng ta sẽ chọn hành động tiếp theo theo giá trị Q-Table trong `epsilon` phần trăm trường hợp, và trong số trường hợp còn lại, chúng ta sẽ thực hiện một hành động ngẫu nhiên. Điều này sẽ cho phép chúng ta khám phá các khu vực của không gian tìm kiếm mà chúng ta chưa từng thấy trước đây.

    ✅ Về mặt giữ thăng bằng - chọn hành động ngẫu nhiên (khám phá) sẽ giống như một cú đẩy ngẫu nhiên sai hướng, và cây cột sẽ phải học cách phục hồi thăng bằng từ những "sai lầm" đó.

### Cải thiện thuật toán

Chúng ta cũng có thể thực hiện hai cải tiến cho thuật toán từ bài học trước:

- **Tính phần thưởng tích lũy trung bình**, qua một số lần mô phỏng. Chúng ta sẽ in tiến trình mỗi 5000 lần lặp, và chúng ta sẽ tính trung bình phần thưởng tích lũy qua khoảng thời gian đó. Điều này có nghĩa là nếu chúng ta đạt được hơn 195 điểm - chúng ta có thể coi bài toán đã được giải quyết, với chất lượng thậm chí cao hơn yêu cầu.

- **Tính kết quả tích lũy trung bình tối đa**, `Qmax`, và chúng ta sẽ lưu trữ Q-Table tương ứng với kết quả đó. Khi bạn chạy quá trình huấn luyện, bạn sẽ nhận thấy rằng đôi khi kết quả tích lũy trung bình bắt đầu giảm, và chúng ta muốn giữ lại các giá trị của Q-Table tương ứng với mô hình tốt nhất được quan sát trong quá trình huấn luyện.

1. Thu thập tất cả phần thưởng tích lũy tại mỗi lần mô phỏng vào vector `rewards` để vẽ biểu đồ sau này. (mã khối 11)

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

Những gì bạn có thể nhận thấy từ các kết quả này:

- **Gần đạt mục tiêu**. Chúng ta rất gần đạt được mục tiêu là đạt 195 phần thưởng tích lũy qua hơn 100 lần chạy liên tiếp của mô phỏng, hoặc chúng ta có thể đã đạt được nó! Ngay cả khi chúng ta đạt được các số nhỏ hơn, chúng ta vẫn không biết, vì chúng ta tính trung bình qua 5000 lần chạy, và chỉ cần 100 lần chạy là đủ theo tiêu chí chính thức.

- **Phần thưởng bắt đầu giảm**. Đôi khi phần thưởng bắt đầu giảm, điều này có nghĩa là chúng ta có thể "phá hủy" các giá trị đã học trong Q-Table bằng các giá trị làm tình hình trở nên tệ hơn.

Quan sát này rõ ràng hơn nếu chúng ta vẽ biểu đồ tiến trình huấn luyện.

## Vẽ biểu đồ tiến trình huấn luyện

Trong quá trình huấn luyện, chúng ta đã thu thập giá trị phần thưởng tích lũy tại mỗi lần lặp vào vector `rewards`. Đây là cách nó trông khi chúng ta vẽ biểu đồ so với số lần lặp:

```python
plt.plot(rewards)
```

![Tiến trình thô](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Từ biểu đồ này, không thể nói được điều gì, vì do bản chất của quá trình huấn luyện ngẫu nhiên, độ dài của các phiên huấn luyện thay đổi rất nhiều. Để làm cho biểu đồ này có ý nghĩa hơn, chúng ta có thể tính **trung bình chạy** qua một loạt các thí nghiệm, giả sử là 100. Điều này có thể được thực hiện một cách thuận tiện bằng cách sử dụng `np.convolve`: (mã khối 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![Tiến trình huấn luyện](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Thay đổi siêu tham số

Để làm cho việc học ổn định hơn, có ý nghĩa khi điều chỉnh một số siêu tham số của chúng ta trong quá trình huấn luyện. Cụ thể:

- **Đối với tốc độ học**, `alpha`, chúng ta có thể bắt đầu với các giá trị gần 1, và sau đó tiếp tục giảm tham số này. Theo thời gian, chúng ta sẽ nhận được các giá trị xác suất tốt trong Q-Table, và do đó chúng ta nên điều chỉnh chúng một cách nhẹ nhàng, thay vì ghi đè hoàn toàn bằng các giá trị mới.

- **Tăng epsilon**. Chúng ta có thể muốn tăng `epsilon` từ từ, để khám phá ít hơn và khai thác nhiều hơn. Có lẽ hợp lý khi bắt đầu với giá trị thấp của `epsilon`, và tăng lên gần 1.
> **Nhiệm vụ 1**: Thử thay đổi các giá trị siêu tham số và xem liệu bạn có thể đạt được tổng phần thưởng cao hơn không. Bạn có đạt trên 195 không?
> **Nhiệm vụ 2**: Để giải quyết vấn đề một cách chính thức, bạn cần đạt được mức thưởng trung bình 195 qua 100 lần chạy liên tiếp. Đo lường điều này trong quá trình huấn luyện và đảm bảo rằng bạn đã giải quyết vấn đề một cách chính thức!

## Xem kết quả hoạt động

Sẽ rất thú vị khi thực sự thấy mô hình đã huấn luyện hoạt động như thế nào. Hãy chạy mô phỏng và áp dụng chiến lược chọn hành động giống như trong quá trình huấn luyện, bằng cách lấy mẫu theo phân phối xác suất trong Q-Table: (khối mã 13)

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

Bạn sẽ thấy điều gì đó như thế này:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Thử thách

> **Nhiệm vụ 3**: Ở đây, chúng ta đang sử dụng bản sao cuối cùng của Q-Table, nhưng có thể nó không phải là bản tốt nhất. Hãy nhớ rằng chúng ta đã lưu Q-Table có hiệu suất tốt nhất vào biến `Qbest`! Thử ví dụ tương tự với Q-Table có hiệu suất tốt nhất bằng cách sao chép `Qbest` sang `Q` và xem liệu bạn có nhận thấy sự khác biệt không.

> **Nhiệm vụ 4**: Ở đây chúng ta không chọn hành động tốt nhất ở mỗi bước, mà thay vào đó lấy mẫu theo phân phối xác suất tương ứng. Liệu có hợp lý hơn không nếu luôn chọn hành động tốt nhất, với giá trị cao nhất trong Q-Table? Điều này có thể thực hiện bằng cách sử dụng hàm `np.argmax` để tìm số hành động tương ứng với giá trị cao nhất trong Q-Table. Hãy triển khai chiến lược này và xem liệu nó có cải thiện khả năng cân bằng không.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Bài tập
[Huấn luyện Mountain Car](assignment.md)

## Kết luận

Chúng ta đã học cách huấn luyện các tác nhân để đạt được kết quả tốt chỉ bằng cách cung cấp cho chúng một hàm thưởng định nghĩa trạng thái mong muốn của trò chơi, và cho chúng cơ hội khám phá không gian tìm kiếm một cách thông minh. Chúng ta đã áp dụng thành công thuật toán Q-Learning trong các trường hợp môi trường rời rạc và liên tục, nhưng với các hành động rời rạc.

Điều quan trọng là cũng cần nghiên cứu các tình huống mà trạng thái hành động cũng liên tục, và khi không gian quan sát phức tạp hơn nhiều, chẳng hạn như hình ảnh từ màn hình trò chơi Atari. Trong những vấn đề này, chúng ta thường cần sử dụng các kỹ thuật học máy mạnh mẽ hơn, chẳng hạn như mạng nơ-ron, để đạt được kết quả tốt. Những chủ đề nâng cao này sẽ là nội dung của khóa học AI nâng cao sắp tới của chúng ta.

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.