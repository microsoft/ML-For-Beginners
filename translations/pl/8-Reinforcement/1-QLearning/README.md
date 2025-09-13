<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T08:27:35+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "pl"
}
-->
# Wprowadzenie do uczenia ze wzmocnieniem i Q-Learningu

![Podsumowanie uczenia ze wzmocnieniem w uczeniu maszynowym w formie sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

Uczenie ze wzmocnieniem opiera się na trzech kluczowych pojęciach: agencie, stanach oraz zestawie akcji dla każdego stanu. Wykonując akcję w określonym stanie, agent otrzymuje nagrodę. Wyobraź sobie grę komputerową Super Mario. Jesteś Mario, znajdujesz się na poziomie gry, stojąc obok krawędzi klifu. Nad tobą jest moneta. Ty, jako Mario, w poziomie gry, w określonej pozycji... to twój stan. Przesunięcie się o krok w prawo (akcja) spowoduje, że spadniesz z klifu, co da ci niską wartość punktową. Jednak naciśnięcie przycisku skoku pozwoli ci zdobyć punkt i pozostać przy życiu. To pozytywny wynik, który powinien nagrodzić cię dodatnią wartością punktową.

Korzystając z uczenia ze wzmocnieniem i symulatora (gry), możesz nauczyć się grać w grę, aby maksymalizować nagrodę, czyli pozostawać przy życiu i zdobywać jak najwięcej punktów.

[![Wprowadzenie do uczenia ze wzmocnieniem](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Kliknij obrazek powyżej, aby posłuchać Dmitry'ego omawiającego uczenie ze wzmocnieniem

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

## Wymagania wstępne i konfiguracja

W tej lekcji będziemy eksperymentować z kodem w Pythonie. Powinieneś być w stanie uruchomić kod z Jupyter Notebook z tej lekcji, zarówno na swoim komputerze, jak i w chmurze.

Możesz otworzyć [notebook lekcji](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) i przejść przez tę lekcję, aby ją zbudować.

> **Uwaga:** Jeśli otwierasz ten kod z chmury, musisz również pobrać plik [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), który jest używany w kodzie notebooka. Dodaj go do tego samego katalogu co notebook.

## Wprowadzenie

W tej lekcji zagłębimy się w świat **[Piotrusia i Wilka](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirowany muzyczną bajką rosyjskiego kompozytora, [Siergieja Prokofiewa](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Wykorzystamy **uczenie ze wzmocnieniem**, aby pozwolić Piotrusiowi eksplorować swoje otoczenie, zbierać smaczne jabłka i unikać spotkania z wilkiem.

**Uczenie ze wzmocnieniem** (RL) to technika uczenia, która pozwala nam nauczyć się optymalnego zachowania **agenta** w określonym **środowisku** poprzez przeprowadzanie wielu eksperymentów. Agent w tym środowisku powinien mieć jakiś **cel**, zdefiniowany przez **funkcję nagrody**.

## Środowisko

Dla uproszczenia, wyobraźmy sobie świat Piotrusia jako kwadratową planszę o rozmiarze `szerokość` x `wysokość`, jak poniżej:

![Środowisko Piotrusia](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Każda komórka na tej planszy może być:

* **ziemią**, po której Piotruś i inne stworzenia mogą chodzić.
* **wodą**, po której oczywiście nie można chodzić.
* **drzewem** lub **trawą**, miejscem, gdzie można odpocząć.
* **jabłkiem**, które Piotruś chętnie znajdzie, aby się nakarmić.
* **wilkiem**, który jest niebezpieczny i należy go unikać.

Istnieje osobny moduł Pythona, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), który zawiera kod do pracy z tym środowiskiem. Ponieważ ten kod nie jest istotny dla zrozumienia naszych koncepcji, zaimportujemy moduł i użyjemy go do stworzenia przykładowej planszy (blok kodu 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Ten kod powinien wydrukować obraz środowiska podobny do powyższego.

## Akcje i polityka

W naszym przykładzie celem Piotrusia będzie znalezienie jabłka, unikając wilka i innych przeszkód. Aby to zrobić, może zasadniczo chodzić po planszy, aż znajdzie jabłko.

Dlatego w dowolnej pozycji może wybrać jedną z następujących akcji: góra, dół, lewo i prawo.

Zdefiniujemy te akcje jako słownik i przypiszemy je do par odpowiadających zmian współrzędnych. Na przykład, ruch w prawo (`R`) odpowiada parze `(1,0)`. (blok kodu 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Podsumowując, strategia i cel tego scenariusza są następujące:

- **Strategia** naszego agenta (Piotrusia) jest zdefiniowana przez tzw. **politykę**. Polityka to funkcja, która zwraca akcję w dowolnym stanie. W naszym przypadku stan problemu jest reprezentowany przez planszę, w tym aktualną pozycję gracza.

- **Cel** uczenia ze wzmocnieniem to ostatecznie nauczenie się dobrej polityki, która pozwoli nam efektywnie rozwiązać problem. Jednak jako punkt odniesienia rozważmy najprostszą politykę zwaną **losowym spacerem**.

## Losowy spacer

Najpierw rozwiążmy nasz problem, implementując strategię losowego spaceru. W przypadku losowego spaceru będziemy losowo wybierać następną akcję z dozwolonych akcji, aż dotrzemy do jabłka (blok kodu 3).

1. Zaimplementuj losowy spacer za pomocą poniższego kodu:

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

    Wywołanie `walk` powinno zwrócić długość odpowiadającej ścieżki, która może się różnić w zależności od uruchomienia.

1. Uruchom eksperyment spaceru kilka razy (np. 100) i wydrukuj wynikowe statystyki (blok kodu 4):

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

    Zauważ, że średnia długość ścieżki wynosi około 30-40 kroków, co jest dość dużo, biorąc pod uwagę fakt, że średnia odległość do najbliższego jabłka wynosi około 5-6 kroków.

    Możesz również zobaczyć, jak wygląda ruch Piotrusia podczas losowego spaceru:

    ![Losowy spacer Piotrusia](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funkcja nagrody

Aby nasza polityka była bardziej inteligentna, musimy zrozumieć, które ruchy są "lepsze" od innych. Aby to zrobić, musimy zdefiniować nasz cel.

Cel można zdefiniować w kategoriach **funkcji nagrody**, która zwróci pewną wartość punktową dla każdego stanu. Im wyższa liczba, tym lepsza funkcja nagrody. (blok kodu 5)

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

Interesującą rzeczą dotyczącą funkcji nagrody jest to, że w większości przypadków *otrzymujemy znaczącą nagrodę dopiero na końcu gry*. Oznacza to, że nasz algorytm powinien jakoś zapamiętać "dobre" kroki, które prowadzą do pozytywnej nagrody na końcu, i zwiększyć ich znaczenie. Podobnie, wszystkie ruchy prowadzące do złych wyników powinny być zniechęcane.

## Q-Learning

Algorytm, który omówimy tutaj, nazywa się **Q-Learning**. W tym algorytmie polityka jest definiowana przez funkcję (lub strukturę danych) zwaną **Q-Tablicą**. Rejestruje ona "dobroć" każdej z akcji w danym stanie.

Nazywa się ją Q-Tablicą, ponieważ często wygodnie jest ją reprezentować jako tablicę lub wielowymiarową macierz. Ponieważ nasza plansza ma wymiary `szerokość` x `wysokość`, możemy reprezentować Q-Tablicę za pomocą tablicy numpy o kształcie `szerokość` x `wysokość` x `len(actions)`: (blok kodu 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Zauważ, że inicjalizujemy wszystkie wartości Q-Tablicy równą wartością, w naszym przypadku - 0.25. Odpowiada to polityce "losowego spaceru", ponieważ wszystkie ruchy w każdym stanie są równie dobre. Możemy przekazać Q-Tablicę do funkcji `plot`, aby zwizualizować tablicę na planszy: `m.plot(Q)`.

![Środowisko Piotrusia](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

W centrum każdej komórki znajduje się "strzałka", która wskazuje preferowany kierunek ruchu. Ponieważ wszystkie kierunki są równe, wyświetlany jest punkt.

Teraz musimy uruchomić symulację, zbadać nasze środowisko i nauczyć się lepszego rozkładu wartości Q-Tablicy, który pozwoli nam znacznie szybciej znaleźć drogę do jabłka.

## Istota Q-Learningu: Równanie Bellmana

Gdy zaczniemy się poruszać, każda akcja będzie miała odpowiadającą jej nagrodę, tj. teoretycznie możemy wybrać następną akcję na podstawie najwyższej natychmiastowej nagrody. Jednak w większości stanów ruch nie osiągnie naszego celu, jakim jest dotarcie do jabłka, i dlatego nie możemy od razu zdecydować, który kierunek jest lepszy.

> Pamiętaj, że nie liczy się natychmiastowy wynik, ale raczej ostateczny wynik, który uzyskamy na końcu symulacji.

Aby uwzględnić tę opóźnioną nagrodę, musimy skorzystać z zasad **[programowania dynamicznego](https://en.wikipedia.org/wiki/Dynamic_programming)**, które pozwalają nam myśleć o naszym problemie w sposób rekurencyjny.

Załóżmy, że teraz znajdujemy się w stanie *s*, i chcemy przejść do następnego stanu *s'*. Wykonując to, otrzymamy natychmiastową nagrodę *r(s,a)*, zdefiniowaną przez funkcję nagrody, plus jakąś przyszłą nagrodę. Jeśli założymy, że nasza Q-Tablica poprawnie odzwierciedla "atrakcyjność" każdej akcji, to w stanie *s'* wybierzemy akcję *a*, która odpowiada maksymalnej wartości *Q(s',a')*. Tak więc najlepsza możliwa przyszła nagroda, jaką możemy uzyskać w stanie *s*, będzie zdefiniowana jako `max`

## Sprawdzanie polityki

Ponieważ Q-Table zawiera "atrakcyjność" każdej akcji w każdym stanie, łatwo jest wykorzystać ją do zdefiniowania efektywnej nawigacji w naszym świecie. W najprostszym przypadku możemy wybrać akcję odpowiadającą najwyższej wartości w Q-Table: (blok kodu 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Jeśli uruchomisz powyższy kod kilka razy, możesz zauważyć, że czasami "zawiesza się" i musisz nacisnąć przycisk STOP w notebooku, aby go przerwać. Dzieje się tak, ponieważ mogą wystąpić sytuacje, w których dwa stany "wskazują" na siebie nawzajem pod względem optymalnej wartości Q-Value, w wyniku czego agent porusza się między tymi stanami w nieskończoność.

## 🚀Wyzwanie

> **Zadanie 1:** Zmodyfikuj funkcję `walk`, aby ograniczyć maksymalną długość ścieżki do określonej liczby kroków (np. 100) i zobacz, jak powyższy kod czasami zwraca tę wartość.

> **Zadanie 2:** Zmodyfikuj funkcję `walk`, aby nie wracała do miejsc, w których już wcześniej była. Zapobiegnie to zapętleniu `walk`, jednak agent nadal może utknąć w miejscu, z którego nie może się wydostać.

## Nawigacja

Lepszą polityką nawigacji byłaby ta, którą stosowaliśmy podczas treningu, łącząca eksploatację i eksplorację. W tej polityce wybieramy każdą akcję z określonym prawdopodobieństwem, proporcjonalnym do wartości w Q-Table. Ta strategia może nadal prowadzić do powrotu agenta do pozycji, którą już eksplorował, ale, jak widać w poniższym kodzie, skutkuje bardzo krótką średnią ścieżką do pożądanej lokalizacji (pamiętaj, że `print_statistics` uruchamia symulację 100 razy): (blok kodu 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Po uruchomieniu tego kodu powinieneś uzyskać znacznie krótszą średnią długość ścieżki niż wcześniej, w zakresie 3-6.

## Badanie procesu uczenia

Jak wspomnieliśmy, proces uczenia to balans między eksploracją a wykorzystaniem zdobytej wiedzy o strukturze przestrzeni problemowej. Widzieliśmy, że wyniki uczenia (zdolność do pomocy agentowi w znalezieniu krótkiej ścieżki do celu) poprawiły się, ale interesujące jest również obserwowanie, jak średnia długość ścieżki zachowuje się podczas procesu uczenia:

## Podsumowanie wniosków:

- **Średnia długość ścieżki rośnie**. Na początku średnia długość ścieżki rośnie. Wynika to prawdopodobnie z faktu, że gdy nic nie wiemy o środowisku, łatwo jest utknąć w złych stanach, takich jak woda czy wilk. Gdy uczymy się więcej i zaczynamy korzystać z tej wiedzy, możemy eksplorować środowisko dłużej, ale nadal nie znamy dobrze lokalizacji jabłek.

- **Długość ścieżki maleje, gdy uczymy się więcej**. Gdy nauczymy się wystarczająco dużo, agentowi łatwiej jest osiągnąć cel, a długość ścieżki zaczyna się zmniejszać. Jednak nadal jesteśmy otwarci na eksplorację, więc często odbiegamy od najlepszej ścieżki i eksplorujemy nowe opcje, co wydłuża ścieżkę ponad optymalną.

- **Długość nagle wzrasta**. Na wykresie można również zauważyć, że w pewnym momencie długość nagle wzrasta. Wskazuje to na stochastyczny charakter procesu i na to, że w pewnym momencie możemy "zepsuć" współczynniki Q-Table, nadpisując je nowymi wartościami. Idealnie powinno się to minimalizować, zmniejszając współczynnik uczenia (na przykład pod koniec treningu dostosowujemy wartości Q-Table tylko o niewielką wartość).

Ogólnie rzecz biorąc, ważne jest, aby pamiętać, że sukces i jakość procesu uczenia w dużej mierze zależą od parametrów, takich jak współczynnik uczenia, jego zmniejszanie oraz współczynnik dyskontowy. Często nazywa się je **hiperparametrami**, aby odróżnić je od **parametrów**, które optymalizujemy podczas treningu (na przykład współczynniki Q-Table). Proces znajdowania najlepszych wartości hiperparametrów nazywa się **optymalizacją hiperparametrów** i zasługuje na osobny temat.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Zadanie 
[Bardziej realistyczny świat](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.