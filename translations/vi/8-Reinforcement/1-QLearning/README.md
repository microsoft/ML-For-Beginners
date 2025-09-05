<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T20:13:42+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về Học Tăng Cường và Q-Learning

![Tóm tắt về học tăng cường trong học máy qua sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote bởi [Tomomi Imura](https://www.twitter.com/girlie_mac)

Học tăng cường liên quan đến ba khái niệm quan trọng: tác nhân, các trạng thái, và một tập hợp các hành động cho mỗi trạng thái. Bằng cách thực hiện một hành động trong một trạng thái cụ thể, tác nhân sẽ nhận được phần thưởng. Hãy tưởng tượng trò chơi điện tử Super Mario. Bạn là Mario, bạn đang ở một cấp độ trong trò chơi, đứng cạnh mép vực. Phía trên bạn là một đồng xu. Bạn là Mario, ở một cấp độ trò chơi, tại một vị trí cụ thể... đó là trạng thái của bạn. Di chuyển một bước sang phải (một hành động) sẽ khiến bạn rơi xuống vực, và điều đó sẽ cho bạn một điểm số thấp. Tuy nhiên, nhấn nút nhảy sẽ giúp bạn ghi điểm và bạn sẽ sống sót. Đó là một kết quả tích cực và điều đó sẽ thưởng cho bạn một điểm số cao.

Bằng cách sử dụng học tăng cường và một trình mô phỏng (trò chơi), bạn có thể học cách chơi trò chơi để tối đa hóa phần thưởng, đó là sống sót và ghi được càng nhiều điểm càng tốt.

[![Giới thiệu về Học Tăng Cường](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Nhấn vào hình ảnh trên để nghe Dmitry thảo luận về Học Tăng Cường

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Yêu cầu và Cài đặt

Trong bài học này, chúng ta sẽ thử nghiệm một số đoạn mã trong Python. Bạn cần có khả năng chạy mã Jupyter Notebook từ bài học này, trên máy tính của bạn hoặc trên đám mây.

Bạn có thể mở [notebook bài học](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) và đi qua bài học này để xây dựng.

> **Lưu ý:** Nếu bạn mở mã này từ đám mây, bạn cũng cần lấy tệp [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), được sử dụng trong mã notebook. Thêm nó vào cùng thư mục với notebook.

## Giới thiệu

Trong bài học này, chúng ta sẽ khám phá thế giới của **[Peter và con sói](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, lấy cảm hứng từ câu chuyện cổ tích âm nhạc của nhà soạn nhạc người Nga, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Chúng ta sẽ sử dụng **Học Tăng Cường** để giúp Peter khám phá môi trường của mình, thu thập những quả táo ngon và tránh gặp con sói.

**Học Tăng Cường** (RL) là một kỹ thuật học cho phép chúng ta học hành vi tối ưu của một **tác nhân** trong một **môi trường** bằng cách thực hiện nhiều thử nghiệm. Một tác nhân trong môi trường này cần có một **mục tiêu**, được định nghĩa bởi một **hàm phần thưởng**.

## Môi trường

Để đơn giản, hãy xem thế giới của Peter là một bảng vuông có kích thước `width` x `height`, như sau:

![Môi trường của Peter](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Mỗi ô trong bảng này có thể là:

* **mặt đất**, nơi Peter và các sinh vật khác có thể đi lại.
* **nước**, nơi bạn rõ ràng không thể đi lại.
* một **cây** hoặc **cỏ**, nơi bạn có thể nghỉ ngơi.
* một **quả táo**, thứ mà Peter sẽ rất vui khi tìm thấy để ăn.
* một **con sói**, thứ nguy hiểm và cần tránh.

Có một module Python riêng biệt, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), chứa mã để làm việc với môi trường này. Vì mã này không quan trọng để hiểu các khái niệm của chúng ta, chúng ta sẽ nhập module và sử dụng nó để tạo bảng mẫu (code block 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Mã này sẽ in ra hình ảnh của môi trường tương tự như hình trên.

## Hành động và chính sách

Trong ví dụ của chúng ta, mục tiêu của Peter là tìm được quả táo, đồng thời tránh con sói và các chướng ngại vật khác. Để làm điều này, anh ta có thể đi lại cho đến khi tìm thấy quả táo.

Do đó, tại bất kỳ vị trí nào, anh ta có thể chọn một trong các hành động sau: lên, xuống, trái và phải.

Chúng ta sẽ định nghĩa các hành động đó dưới dạng một từ điển và ánh xạ chúng tới các cặp thay đổi tọa độ tương ứng. Ví dụ, di chuyển sang phải (`R`) sẽ tương ứng với cặp `(1,0)`. (code block 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Tóm lại, chiến lược và mục tiêu của kịch bản này như sau:

- **Chiến lược**, của tác nhân (Peter) được định nghĩa bởi một cái gọi là **chính sách**. Chính sách là một hàm trả về hành động tại bất kỳ trạng thái nào. Trong trường hợp của chúng ta, trạng thái của vấn đề được biểu diễn bởi bảng, bao gồm vị trí hiện tại của người chơi.

- **Mục tiêu**, của học tăng cường là cuối cùng học được một chính sách tốt cho phép chúng ta giải quyết vấn đề một cách hiệu quả. Tuy nhiên, như một cơ sở, hãy xem xét chính sách đơn giản nhất gọi là **đi bộ ngẫu nhiên**.

## Đi bộ ngẫu nhiên

Hãy giải quyết vấn đề của chúng ta bằng cách triển khai chiến lược đi bộ ngẫu nhiên. Với đi bộ ngẫu nhiên, chúng ta sẽ chọn ngẫu nhiên hành động tiếp theo từ các hành động được phép, cho đến khi chúng ta đạt được quả táo (code block 3).

1. Triển khai đi bộ ngẫu nhiên với mã dưới đây:

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

    Lệnh gọi `walk` sẽ trả về độ dài của đường đi tương ứng, có thể thay đổi từ lần chạy này sang lần chạy khác.

1. Thực hiện thử nghiệm đi bộ một số lần (ví dụ, 100 lần), và in ra thống kê kết quả (code block 4):

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

    Lưu ý rằng độ dài trung bình của một đường đi là khoảng 30-40 bước, khá nhiều, trong khi khoảng cách trung bình đến quả táo gần nhất là khoảng 5-6 bước.

    Bạn cũng có thể xem chuyển động của Peter trong quá trình đi bộ ngẫu nhiên:

    ![Đi bộ ngẫu nhiên của Peter](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Hàm phần thưởng

Để làm cho chính sách của chúng ta thông minh hơn, chúng ta cần hiểu hành động nào "tốt hơn" hành động khác. Để làm điều này, chúng ta cần định nghĩa mục tiêu của mình.

Mục tiêu có thể được định nghĩa dưới dạng một **hàm phần thưởng**, hàm này sẽ trả về một giá trị điểm cho mỗi trạng thái. Số càng cao, hàm phần thưởng càng tốt. (code block 5)

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

Một điều thú vị về các hàm phần thưởng là trong hầu hết các trường hợp, *chúng ta chỉ nhận được phần thưởng đáng kể vào cuối trò chơi*. Điều này có nghĩa là thuật toán của chúng ta cần phải nhớ các bước "tốt" dẫn đến phần thưởng tích cực ở cuối, và tăng tầm quan trọng của chúng. Tương tự, tất cả các bước dẫn đến kết quả xấu cần bị giảm giá trị.

## Q-Learning

Thuật toán mà chúng ta sẽ thảo luận ở đây được gọi là **Q-Learning**. Trong thuật toán này, chính sách được định nghĩa bởi một hàm (hoặc cấu trúc dữ liệu) gọi là **Q-Table**. Nó ghi lại "mức độ tốt" của mỗi hành động trong một trạng thái nhất định.

Nó được gọi là Q-Table vì thường tiện lợi để biểu diễn nó dưới dạng một bảng, hoặc mảng đa chiều. Vì bảng của chúng ta có kích thước `width` x `height`, chúng ta có thể biểu diễn Q-Table bằng một mảng numpy với hình dạng `width` x `height` x `len(actions)`: (code block 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Lưu ý rằng chúng ta khởi tạo tất cả các giá trị của Q-Table với một giá trị bằng nhau, trong trường hợp của chúng ta là 0.25. Điều này tương ứng với chính sách "đi bộ ngẫu nhiên", vì tất cả các hành động trong mỗi trạng thái đều tốt như nhau. Chúng ta có thể truyền Q-Table vào hàm `plot` để trực quan hóa bảng trên bảng: `m.plot(Q)`.

![Môi trường của Peter](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Ở trung tâm của mỗi ô có một "mũi tên" chỉ hướng di chuyển ưu tiên. Vì tất cả các hướng đều bằng nhau, một dấu chấm được hiển thị.

Bây giờ chúng ta cần chạy mô phỏng, khám phá môi trường của mình, và học một phân phối giá trị Q-Table tốt hơn, điều này sẽ cho phép chúng ta tìm đường đến quả táo nhanh hơn nhiều.

## Bản chất của Q-Learning: Phương trình Bellman

Khi chúng ta bắt đầu di chuyển, mỗi hành động sẽ có một phần thưởng tương ứng, tức là chúng ta có thể lý thuyết chọn hành động tiếp theo dựa trên phần thưởng ngay lập tức cao nhất. Tuy nhiên, trong hầu hết các trạng thái, hành động sẽ không đạt được mục tiêu của chúng ta là đến quả táo, và do đó chúng ta không thể ngay lập tức quyết định hướng nào tốt hơn.

> Hãy nhớ rằng điều quan trọng không phải là kết quả ngay lập tức, mà là kết quả cuối cùng, mà chúng ta sẽ đạt được vào cuối mô phỏng.

Để tính đến phần thưởng bị trì hoãn này, chúng ta cần sử dụng các nguyên tắc của **[lập trình động](https://en.wikipedia.org/wiki/Dynamic_programming)**, cho phép chúng ta suy nghĩ về vấn đề của mình một cách đệ quy.

Giả sử chúng ta đang ở trạng thái *s*, và chúng ta muốn di chuyển đến trạng thái tiếp theo *s'*. Bằng cách làm như vậy, chúng ta sẽ nhận được phần thưởng ngay lập tức *r(s,a)*, được định nghĩa bởi hàm phần thưởng, cộng với một phần thưởng tương lai. Nếu chúng ta giả sử rằng Q-Table của chúng ta phản ánh chính xác "sự hấp dẫn" của mỗi hành động, thì tại trạng thái *s'* chúng ta sẽ chọn một hành động *a* tương ứng với giá trị tối đa của *Q(s',a')*. Do đó, phần thưởng tương lai tốt nhất có thể mà chúng ta có thể nhận được tại trạng thái *s* sẽ được định nghĩa là `max`

## Kiểm tra chính sách

Vì Q-Table liệt kê "mức độ hấp dẫn" của mỗi hành động tại mỗi trạng thái, nên rất dễ sử dụng nó để xác định cách điều hướng hiệu quả trong thế giới của chúng ta. Trong trường hợp đơn giản nhất, chúng ta có thể chọn hành động tương ứng với giá trị Q-Table cao nhất: (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Nếu bạn thử đoạn mã trên nhiều lần, bạn có thể nhận thấy rằng đôi khi nó "bị treo", và bạn cần nhấn nút STOP trong notebook để dừng lại. Điều này xảy ra vì có thể có những tình huống khi hai trạng thái "chỉ" vào nhau về mặt giá trị Q tối ưu, dẫn đến việc agent di chuyển qua lại giữa các trạng thái đó vô thời hạn.

## 🚀Thử thách

> **Nhiệm vụ 1:** Sửa đổi hàm `walk` để giới hạn độ dài tối đa của đường đi bằng một số bước nhất định (ví dụ, 100), và xem đoạn mã trên trả về giá trị này theo thời gian.

> **Nhiệm vụ 2:** Sửa đổi hàm `walk` để không quay lại những nơi mà nó đã từng đi qua trước đó. Điều này sẽ ngăn `walk` lặp lại, tuy nhiên, agent vẫn có thể bị "mắc kẹt" ở một vị trí mà nó không thể thoát ra.

## Điều hướng

Một chính sách điều hướng tốt hơn sẽ là chính sách mà chúng ta đã sử dụng trong quá trình huấn luyện, kết hợp giữa khai thác và khám phá. Trong chính sách này, chúng ta sẽ chọn mỗi hành động với một xác suất nhất định, tỷ lệ thuận với các giá trị trong Q-Table. Chiến lược này vẫn có thể dẫn đến việc agent quay lại một vị trí mà nó đã khám phá, nhưng như bạn có thể thấy từ đoạn mã dưới đây, nó dẫn đến một đường đi trung bình rất ngắn đến vị trí mong muốn (hãy nhớ rằng `print_statistics` chạy mô phỏng 100 lần): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Sau khi chạy đoạn mã này, bạn sẽ nhận được độ dài đường đi trung bình nhỏ hơn nhiều so với trước, trong khoảng từ 3-6.

## Khám phá quá trình học

Như chúng ta đã đề cập, quá trình học là sự cân bằng giữa việc khám phá và khai thác kiến thức đã thu được về cấu trúc không gian vấn đề. Chúng ta đã thấy rằng kết quả của việc học (khả năng giúp agent tìm đường ngắn đến mục tiêu) đã được cải thiện, nhưng cũng rất thú vị khi quan sát cách độ dài đường đi trung bình thay đổi trong quá trình học:

## Tóm tắt bài học:

- **Độ dài đường đi trung bình tăng lên**. Điều chúng ta thấy ở đây là ban đầu, độ dài đường đi trung bình tăng lên. Điều này có thể là do khi chúng ta chưa biết gì về môi trường, chúng ta có khả năng bị mắc kẹt ở các trạng thái xấu, như nước hoặc sói. Khi chúng ta học được nhiều hơn và bắt đầu sử dụng kiến thức này, chúng ta có thể khám phá môi trường lâu hơn, nhưng vẫn chưa biết rõ vị trí của những quả táo.

- **Độ dài đường đi giảm khi học được nhiều hơn**. Khi chúng ta học đủ, việc đạt được mục tiêu trở nên dễ dàng hơn đối với agent, và độ dài đường đi bắt đầu giảm. Tuy nhiên, chúng ta vẫn mở rộng khám phá, vì vậy chúng ta thường đi lệch khỏi đường đi tốt nhất và khám phá các lựa chọn mới, làm cho đường đi dài hơn mức tối ưu.

- **Độ dài tăng đột ngột**. Điều chúng ta cũng quan sát được trên biểu đồ này là tại một số thời điểm, độ dài tăng đột ngột. Điều này cho thấy tính ngẫu nhiên của quá trình, và rằng chúng ta có thể "làm hỏng" các hệ số Q-Table bằng cách ghi đè chúng với các giá trị mới. Điều này lý tưởng nên được giảm thiểu bằng cách giảm tốc độ học (ví dụ, về cuối quá trình huấn luyện, chúng ta chỉ điều chỉnh các giá trị Q-Table bằng một giá trị nhỏ).

Nhìn chung, điều quan trọng cần nhớ là sự thành công và chất lượng của quá trình học phụ thuộc đáng kể vào các tham số, như tốc độ học, sự giảm tốc độ học, và hệ số chiết khấu. Những tham số này thường được gọi là **siêu tham số**, để phân biệt với **tham số**, mà chúng ta tối ưu trong quá trình huấn luyện (ví dụ, các hệ số Q-Table). Quá trình tìm giá trị siêu tham số tốt nhất được gọi là **tối ưu hóa siêu tham số**, và nó xứng đáng là một chủ đề riêng.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Bài tập 
[Một thế giới thực tế hơn](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.