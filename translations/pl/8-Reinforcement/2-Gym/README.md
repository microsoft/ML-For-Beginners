<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T08:28:22+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "pl"
}
-->
# CartPole Skating

Problem, który rozwiązywaliśmy w poprzedniej lekcji, może wydawać się zabawkowy i mało przydatny w rzeczywistych scenariuszach. Jednak tak nie jest, ponieważ wiele problemów w prawdziwym świecie również ma podobny charakter – na przykład gra w szachy czy Go. Są one podobne, ponieważ również mamy planszę z określonymi zasadami i **dyskretny stan**.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

## Wprowadzenie

W tej lekcji zastosujemy te same zasady Q-Learningu do problemu z **ciągłym stanem**, czyli stanem opisanym przez jedną lub więcej liczb rzeczywistych. Zajmiemy się następującym problemem:

> **Problem**: Jeśli Piotr chce uciec przed wilkiem, musi nauczyć się poruszać szybciej. Zobaczymy, jak Piotr może nauczyć się jeździć na łyżwach, a w szczególności utrzymywać równowagę, korzystając z Q-Learningu.

![Wielka ucieczka!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Piotr i jego przyjaciele wykazują się kreatywnością, aby uciec przed wilkiem! Obraz autorstwa [Jen Looper](https://twitter.com/jenlooper)

Użyjemy uproszczonej wersji problemu utrzymywania równowagi, znanej jako problem **CartPole**. W świecie CartPole mamy poziomy suwak, który może poruszać się w lewo lub w prawo, a celem jest utrzymanie pionowego słupka na górze suwaka.

## Wymagania wstępne

W tej lekcji będziemy korzystać z biblioteki **OpenAI Gym**, aby symulować różne **środowiska**. Możesz uruchomić kod z tej lekcji lokalnie (np. w Visual Studio Code), w takim przypadku symulacja otworzy się w nowym oknie. Jeśli uruchamiasz kod online, może być konieczne wprowadzenie pewnych zmian w kodzie, jak opisano [tutaj](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

W poprzedniej lekcji zasady gry i stan były określone przez klasę `Board`, którą sami zdefiniowaliśmy. Tutaj użyjemy specjalnego **środowiska symulacyjnego**, które zasymuluje fizykę stojącą za balansującym słupkiem. Jednym z najpopularniejszych środowisk symulacyjnych do trenowania algorytmów uczenia ze wzmocnieniem jest [Gym](https://gym.openai.com/), utrzymywany przez [OpenAI](https://openai.com/). Dzięki Gym możemy tworzyć różne **środowiska**, od symulacji CartPole po gry Atari.

> **Uwaga**: Inne środowiska dostępne w OpenAI Gym możesz zobaczyć [tutaj](https://gym.openai.com/envs/#classic_control).

Najpierw zainstalujmy Gym i zaimportujmy wymagane biblioteki (blok kodu 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Ćwiczenie – inicjalizacja środowiska CartPole

Aby pracować z problemem balansowania CartPole, musimy zainicjalizować odpowiednie środowisko. Każde środowisko jest związane z:

- **Przestrzenią obserwacji**, która definiuje strukturę informacji, jakie otrzymujemy ze środowiska. W przypadku problemu CartPole otrzymujemy pozycję słupka, prędkość i inne wartości.

- **Przestrzenią akcji**, która definiuje możliwe działania. W naszym przypadku przestrzeń akcji jest dyskretna i składa się z dwóch działań – **lewo** i **prawo**. (blok kodu 2)

1. Aby zainicjalizować środowisko, wpisz następujący kod:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Aby zobaczyć, jak działa środowisko, uruchommy krótką symulację na 100 kroków. Na każdym kroku podajemy jedną z akcji do wykonania – w tej symulacji losowo wybieramy akcję z `action_space`.

1. Uruchom poniższy kod i zobacz, co się stanie.

    ✅ Pamiętaj, że preferowane jest uruchamianie tego kodu na lokalnej instalacji Pythona! (blok kodu 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Powinieneś zobaczyć coś podobnego do tego obrazu:

    ![CartPole bez równowagi](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Podczas symulacji musimy uzyskać obserwacje, aby zdecydować, jak działać. W rzeczywistości funkcja `step` zwraca bieżące obserwacje, funkcję nagrody oraz flagę `done`, która wskazuje, czy symulacja powinna być kontynuowana, czy nie: (blok kodu 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    W notatniku powinieneś zobaczyć coś takiego:

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

    Wektor obserwacji zwracany na każdym kroku symulacji zawiera następujące wartości:
    - Pozycja wózka
    - Prędkość wózka
    - Kąt słupka
    - Prędkość obrotowa słupka

1. Uzyskaj minimalne i maksymalne wartości tych liczb: (blok kodu 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Możesz również zauważyć, że wartość nagrody na każdym kroku symulacji wynosi zawsze 1. Dzieje się tak, ponieważ naszym celem jest przetrwanie jak najdłużej, tj. utrzymanie słupka w możliwie pionowej pozycji przez najdłuższy czas.

    ✅ W rzeczywistości symulacja CartPole jest uznawana za rozwiązaną, jeśli uda nam się uzyskać średnią nagrodę 195 w 100 kolejnych próbach.

## Dyskretyzacja stanu

W Q-Learningu musimy zbudować Q-Table, która określa, co robić w każdym stanie. Aby to zrobić, stan musi być **dyskretny**, a dokładniej, powinien zawierać skończoną liczbę wartości dyskretnych. Dlatego musimy w jakiś sposób **zdyskretyzować** nasze obserwacje, mapując je na skończony zbiór stanów.

Istnieje kilka sposobów, aby to zrobić:

- **Podział na przedziały**. Jeśli znamy zakres danej wartości, możemy podzielić ten zakres na liczbę **przedziałów**, a następnie zastąpić wartość numerem przedziału, do którego należy. Można to zrobić za pomocą metody numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). W tym przypadku dokładnie znamy rozmiar stanu, ponieważ zależy on od liczby przedziałów, które wybierzemy do digitalizacji.

✅ Możemy użyć interpolacji liniowej, aby sprowadzić wartości do pewnego skończonego zakresu (np. od -20 do 20), a następnie przekonwertować liczby na liczby całkowite przez zaokrąglenie. Daje nam to nieco mniej kontroli nad rozmiarem stanu, szczególnie jeśli nie znamy dokładnych zakresów wartości wejściowych. Na przykład w naszym przypadku 2 z 4 wartości nie mają górnych/dolnych ograniczeń, co może skutkować nieskończoną liczbą stanów.

W naszym przykładzie wybierzemy drugie podejście. Jak zauważysz później, mimo nieokreślonych górnych/dolnych ograniczeń, te wartości rzadko przyjmują wartości poza pewnymi skończonymi zakresami, więc stany z ekstremalnymi wartościami będą bardzo rzadkie.

1. Oto funkcja, która pobiera obserwację z naszego modelu i zwraca krotkę 4 wartości całkowitych: (blok kodu 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Przyjrzyjmy się również innej metodzie dyskretyzacji za pomocą przedziałów: (blok kodu 7)

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

1. Uruchommy teraz krótką symulację i zaobserwujmy te zdyskretyzowane wartości środowiska. Możesz wypróbować zarówno `discretize`, jak i `discretize_bins`, aby zobaczyć, czy istnieje różnica.

    ✅ `discretize_bins` zwraca numer przedziału, który zaczyna się od 0. Dlatego dla wartości zmiennej wejściowej w okolicach 0 zwraca numer ze środka zakresu (10). W `discretize` nie przejmowaliśmy się zakresem wartości wyjściowych, pozwalając im być ujemnymi, więc wartości stanu nie są przesunięte, a 0 odpowiada 0. (blok kodu 8)

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

    ✅ Odkomentuj linię zaczynającą się od `env.render`, jeśli chcesz zobaczyć, jak środowisko działa. W przeciwnym razie możesz uruchomić je w tle, co jest szybsze. Tę "niewidzialną" egzekucję wykorzystamy podczas procesu Q-Learningu.

## Struktura Q-Table

W poprzedniej lekcji stan był prostą parą liczb od 0 do 8, więc wygodnie było reprezentować Q-Table jako tensor numpy o kształcie 8x8x2. Jeśli używamy dyskretyzacji za pomocą przedziałów, rozmiar naszego wektora stanu jest również znany, więc możemy użyć tego samego podejścia i reprezentować stan jako tablicę o kształcie 20x20x10x10x2 (gdzie 2 to wymiar przestrzeni akcji, a pierwsze wymiary odpowiadają liczbie przedziałów, które wybraliśmy dla każdej z wartości w przestrzeni obserwacji).

Jednak czasami dokładne wymiary przestrzeni obserwacji nie są znane. W przypadku funkcji `discretize` nigdy nie możemy być pewni, że nasz stan pozostaje w określonych granicach, ponieważ niektóre z oryginalnych wartości nie są ograniczone. Dlatego użyjemy nieco innego podejścia i przedstawimy Q-Table jako słownik.

1. Użyj pary *(stan, akcja)* jako klucza słownika, a wartość będzie odpowiadać wartości wpisu w Q-Table. (blok kodu 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Tutaj definiujemy również funkcję `qvalues()`, która zwraca listę wartości Q-Table dla danego stanu, odpowiadającą wszystkim możliwym akcjom. Jeśli wpis nie jest obecny w Q-Table, zwrócimy 0 jako wartość domyślną.

## Zaczynamy Q-Learning

Teraz jesteśmy gotowi, aby nauczyć Piotra balansowania!

1. Najpierw ustawmy kilka hiperparametrów: (blok kodu 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Tutaj `alpha` to **współczynnik uczenia się**, który określa, w jakim stopniu powinniśmy dostosowywać bieżące wartości Q-Table na każdym kroku. W poprzedniej lekcji zaczynaliśmy od 1, a następnie zmniejszaliśmy `alpha` do niższych wartości podczas treningu. W tym przykładzie utrzymamy go na stałym poziomie dla uproszczenia, ale możesz eksperymentować z dostosowywaniem wartości `alpha` później.

    `gamma` to **współczynnik dyskontowy**, który pokazuje, w jakim stopniu powinniśmy priorytetyzować przyszłą nagrodę nad bieżącą.

    `epsilon` to **współczynnik eksploracji/eksploatacji**, który określa, czy powinniśmy preferować eksplorację czy eksploatację. W naszym algorytmie w `epsilon` procentach przypadków wybierzemy następną akcję zgodnie z wartościami Q-Table, a w pozostałych przypadkach wykonamy losową akcję. Pozwoli nam to eksplorować obszary przestrzeni poszukiwań, których wcześniej nie widzieliśmy.

    ✅ W kontekście balansowania – wybór losowej akcji (eksploracja) działałby jak przypadkowe "pchnięcie" w złą stronę, a słupek musiałby nauczyć się, jak odzyskać równowagę po tych "błędach".

### Ulepszanie algorytmu

Możemy również wprowadzić dwa ulepszenia do naszego algorytmu z poprzedniej lekcji:

- **Obliczanie średniej skumulowanej nagrody** w serii symulacji. Będziemy drukować postęp co 5000 iteracji i uśredniać naszą skumulowaną nagrodę w tym okresie. Oznacza to, że jeśli uzyskamy więcej niż 195 punktów – możemy uznać problem za rozwiązany, i to z jeszcze wyższą jakością niż wymagana.

- **Obliczanie maksymalnego średniego wyniku skumulowanego**, `Qmax`, i przechowywanie Q-Table odpowiadającej temu wynikowi. Podczas treningu zauważysz, że czasami średni wynik skumulowany zaczyna spadać, a my chcemy zachować wartości Q-Table odpowiadające najlepszemu modelowi zaobserwowanemu podczas treningu.

1. Zbierz wszystkie skumulowane nagrody z każdej symulacji w wektorze `rewards` do dalszego wykreślania. (blok kodu 11)

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

Co możesz zauważyć z tych wyników:

- **Blisko naszego celu**. Jesteśmy bardzo blisko osiągnięcia celu uzyskania 195 skumulowanych nagród w ponad 100 kolejnych uruchomieniach symulacji, lub możemy już go osiągnęliśmy! Nawet jeśli uzyskamy mniejsze liczby, nadal nie wiemy, ponieważ uśredniamy wyniki z 5000 uruchomień, a formalne kryterium wymaga tylko 100 uruchomień.

- **Nagroda zaczyna spadać**. Czasami nagroda zaczyna spadać, co oznacza, że możemy "zniszczyć" już wyuczone wartości w Q-Table, zastępując je tymi, które pogarszają sytuację.

To zjawisko jest bardziej widoczne, jeśli wykreślimy postęp treningu.

## Wykres postępu treningu

Podczas treningu zbieraliśmy wartość skumulowanej nagrody na każdej iteracji w wektorze `rewards`. Oto jak wygląda, gdy wykreślimy ją względem numeru iteracji:

```python
plt.plot(rewards)
```

![surowy postęp](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Z tego wykresu trudno coś wywnioskować, ponieważ ze względu na charakter stochastycznego procesu treningowego długość sesji treningowych znacznie się różni. Aby lepiej zrozumieć ten wykres, możemy obliczyć **średnią kroczącą** dla serii eksperymentów, na przykład 100. Można to wygodnie zrobić za pomocą `np.convolve`: (blok kodu 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![postęp treningu](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Zmienianie hiperparametrów

Aby uczynić proces uczenia bardziej stabilnym, warto dostosować niektóre z naszych hiperparametrów podczas treningu. W szczególności:

- **Dla współczynnika uczenia się**, `alpha`, możemy zacząć od wartości bliskich 1, a następnie stopniowo zmniejszać ten parametr. Z czasem będziemy uzyskiwać dobre wartości prawdopodobieństwa w Q-Table, więc powinniśmy je dostosowywać delikatnie, a nie całkowicie nadpisywać nowymi wartościami.

- **Zwiększanie epsilon**. Możemy chcieć stopniowo zwiększać `epsilon`, aby mniej eksplorować, a bardziej eksploatować. Prawdopodobnie warto zacząć od niższej wartości `epsilon` i stopniowo zwiększać ją do prawie 1.
> **Zadanie 1**: Pobaw się wartościami hiperparametrów i sprawdź, czy możesz osiągnąć wyższą skumulowaną nagrodę. Czy udaje Ci się przekroczyć 195?
> **Zadanie 2**: Aby formalnie rozwiązać problem, musisz osiągnąć średnią nagrodę na poziomie 195 w 100 kolejnych próbach. Mierz to podczas treningu i upewnij się, że formalnie rozwiązałeś problem!

## Zobaczenie wyników w praktyce

Ciekawie byłoby zobaczyć, jak zachowuje się wytrenowany model. Uruchommy symulację i zastosujmy tę samą strategię wyboru akcji, co podczas treningu, próbkując zgodnie z rozkładem prawdopodobieństwa w Q-Table: (blok kodu 13)

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

Powinieneś zobaczyć coś takiego:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Wyzwanie

> **Zadanie 3**: Tutaj korzystaliśmy z ostatecznej wersji Q-Table, która może nie być najlepsza. Pamiętaj, że zapisaliśmy najlepiej działającą Q-Table w zmiennej `Qbest`! Wypróbuj ten sam przykład, używając najlepiej działającej Q-Table, kopiując `Qbest` do `Q`, i sprawdź, czy zauważysz różnicę.

> **Zadanie 4**: W tym przypadku nie wybieraliśmy najlepszej akcji na każdym kroku, lecz próbkowaliśmy zgodnie z odpowiadającym rozkładem prawdopodobieństwa. Czy miałoby więcej sensu zawsze wybierać najlepszą akcję, z najwyższą wartością w Q-Table? Można to zrobić, używając funkcji `np.argmax`, aby znaleźć numer akcji odpowiadający najwyższej wartości w Q-Table. Zaimplementuj tę strategię i sprawdź, czy poprawia to balansowanie.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Zadanie
[Wytrenuj Mountain Car](assignment.md)

## Podsumowanie

Nauczyliśmy się, jak trenować agentów, aby osiągali dobre wyniki, dostarczając im jedynie funkcję nagrody, która definiuje pożądany stan gry, oraz dając im możliwość inteligentnego eksplorowania przestrzeni poszukiwań. Z powodzeniem zastosowaliśmy algorytm Q-Learning w przypadkach środowisk dyskretnych i ciągłych, ale z dyskretnymi akcjami.

Ważne jest również badanie sytuacji, w których przestrzeń akcji jest również ciągła, a przestrzeń obserwacji jest znacznie bardziej złożona, na przykład obraz z ekranu gry Atari. W takich problemach często musimy korzystać z bardziej zaawansowanych technik uczenia maszynowego, takich jak sieci neuronowe, aby osiągnąć dobre wyniki. Te bardziej zaawansowane tematy będą przedmiotem naszego kolejnego, bardziej zaawansowanego kursu AI.

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego języku źródłowym powinien być uznawany za wiarygodne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.